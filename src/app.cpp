#include "app.hpp"

#include <SGE/engine.hpp>
#include <SGE/renderer/renderer.hpp>
#include <SGE/renderer/camera.hpp>
#include <SGE/input.hpp>
#include <SGE/time/time.hpp>
#include <SGE/types/anchor.hpp>
#include <SGE/types/color.hpp>
#include <SGE/types/window_settings.hpp>
#include <SGE/types/blend_mode.hpp>
#include <SGE/log.hpp>
#include <SGE/time/stopwatch.hpp>

#include <glm/trigonometric.hpp>
#include <glm/gtc/random.hpp>

#include "glm/ext/scalar_constants.hpp"
#include "thread_pool.hpp"

using namespace sge;

static constexpr size_t PARTICLE_COUNT = 4000;
static constexpr float PARTICLE_SIZE = 8.0f;
static constexpr float PARTICLE_SPACING = 5.0f;
static constexpr float SMOOTHING_RADIUS = 15.0f;
static constexpr float KERNEL_H = 2.0f;
static constexpr float GRAVITY = 150.0f;
static constexpr float TARGET_DENSITY = 6.0f;
static constexpr float PRESSURE_MULTIPLIER = 150.0f;
static constexpr float NEAR_PRESSURE_MULTIPLIER = 100.0f;
static constexpr float COLLISION_DAMPING = 0.9f;
static constexpr float VISCOSITY_STRENGTH = 0.1f;
static constexpr float INTERACTION_RADIUS = 250.0f;
static constexpr float INTERACTION_STRENGTH = 500.0f;

struct Particle {
    glm::vec2 position;
    glm::vec2 velocity;
    float property;
    float density;
};

static struct GameState {
    Batch batch;
    Camera camera = Camera(CameraOrigin::TopLeft);
    bool paused = false;
    std::vector<glm::vec2> positions;
    std::vector<glm::vec2> predicted_positions;
    std::vector<glm::vec2> velocities;
    std::vector<float> densities;
    std::vector<float> near_densities;
    std::vector<LinearRgba> colors;
    BS::thread_pool<> pool;
} g;

float ExampleFunction(glm::vec2 pos) {
    return glm::cos((pos.y / 80.0f) - 3.0f + glm::sin(pos.x / 80.0f)) * 0.5f + 0.5f;
    // return glm::cos(pos.x / 100.0f) * 0.5f + 0.5f;
}

float DensityKernel(float x, float radius) {
    if (x >= radius) return 0.0f;

    float v = radius - x;
    float scale = 6.0f / (glm::pi<float>() * glm::pow(radius, 4.0f));
    return v * v * scale;
}

float NearDensityKernel(float x, float radius) {
    if (x >= radius) return 0.0f;

    float v = radius - x;
    float scale = 10.0f / (glm::pi<float>() * glm::pow(radius, 5.0f));
    return v * v * v * scale;
}

float DensityKernelDerivative(float x, float radius) {
    if (x >= radius) return 0.0f;

    float v = radius - x;
    float scale = 12.0f / (glm::pow(radius, 4.0f) * glm::pi<float>());
    return -v * scale;
}

float NearDensityKernelDerivative(float x, float radius) {
    if (x >= radius) return 0.0f;

    float v = radius - x;
    float scale = 30.0f / (glm::pow(radius, 5.0f) * glm::pi<float>());
    return -v * v * scale;
}

float ViscosityKernel(float x, float radius) {
    if (x >= radius) return 0.0f;

    float v = radius * radius - x * x;
    float q = SMOOTHING_RADIUS;
    float q6 = glm::pow(q, 6.0f);
    float q4 = glm::pow(q, 4.0f);
    float q2 = glm::pow(q, 2.0f);
    float scale = (4.0f * q6) / (glm::pi<float>() * (4.0f*q6 - 6.0f * q4 + 4.0f * q2 - 1.0f) * glm::pow(radius, 8.0f));
    return v * v * v * scale;
}

std::tuple<float, float> CalculateDensity(size_t index) {
    float density = 0.0f;
    float near_density = 0.0f;

    const glm::vec2 point = g.predicted_positions[index];

    for (size_t i = 0; i < PARTICLE_COUNT; ++i) {
        glm::vec2 offset = g.predicted_positions[i] - point;
        float sqr_dst = glm::dot(offset, offset);
        if (sqr_dst > SMOOTHING_RADIUS * SMOOTHING_RADIUS) continue;

        float dst = glm::sqrt(sqr_dst);

        density += DensityKernel(dst, SMOOTHING_RADIUS);
        near_density += NearDensityKernel(dst, SMOOTHING_RADIUS);
    }

    return std::make_tuple(density, near_density);
}

static void InitParticles() {
    const float width = g.camera.viewport().x - PARTICLE_SIZE;
    const float height = g.camera.viewport().y - PARTICLE_SIZE;

    float x = width / 2.0f;
    float y = height / 2.0f;

    // Spawn in spiral order
    size_t s = 1;
    int direction = 0;
    size_t index = 0;
    while (index < PARTICLE_COUNT) {
        for (int i = 0; i < glm::min(PARTICLE_COUNT, index+s)-index; ++i) {
            glm::vec2 pos = glm::vec2(x, y);
            g.positions[index] = pos;
            g.velocities[index] = glm::vec2(0.0);
            index += 1;

            switch (direction) {
                case 0: x += PARTICLE_SIZE;
                break;
                case 1: y += PARTICLE_SIZE;
                break;
                case 2: x -= PARTICLE_SIZE;
                break;
                case 3: y -= PARTICLE_SIZE;
                break;
            }
        }

        direction = (direction + 1) % 4;
        if (direction % 2 == 0)
            s++;
    }
}

static float ConvertDensityToPressure(float density) {
    float density_error = TARGET_DENSITY - density;
    float pressure = density_error * PRESSURE_MULTIPLIER;
    return pressure;
}

static float CalculateSharedPressure(float densityA, float densityB) {
    float pressureA = ConvertDensityToPressure(densityA);
    float pressureB = ConvertDensityToPressure(densityB);
    return (pressureA + pressureB) / 2.0f;
}

static float ConvertNearDensityToNearPressure(float density) {
    return density * NEAR_PRESSURE_MULTIPLIER;
}

static float CalculateSharedNearPressure(float densityA, float densityB) {
    float pressureA = ConvertNearDensityToNearPressure(densityA);
    float pressureB = ConvertNearDensityToNearPressure(densityB);
    return (pressureA + pressureB) / 2.0f;
}

static glm::vec2 CalculatePressureForce(size_t index) {
    glm::vec2 pressure_force = glm::vec2(0.0f);

    const glm::vec2 point = g.predicted_positions[index];

    for (size_t i = 0; i < PARTICLE_COUNT; ++i) {
        if (i == index) continue;

        glm::vec2 offset = g.predicted_positions[i] - point;
        float sqr_dst = glm::dot(offset, offset);

        if (sqr_dst > SMOOTHING_RADIUS * SMOOTHING_RADIUS) continue;

        float dst = glm::sqrt(sqr_dst);
        glm::vec2 dir = dst > glm::epsilon<float>() ? (g.predicted_positions[i] - point) / dst : glm::linearRand(glm::vec2(-1.0f), glm::vec2(1.0f));

        float density = g.densities[i];
        float near_density = g.near_densities[i];

        float shared_pressure = CalculateSharedPressure(g.densities[index], density);
        float shared_near_pressure = CalculateSharedNearPressure(g.near_densities[index], near_density);

        pressure_force += shared_pressure * dir * DensityKernelDerivative(dst, SMOOTHING_RADIUS) / density;
        pressure_force += shared_near_pressure * dir * NearDensityKernelDerivative(dst, SMOOTHING_RADIUS) / near_density;
    }

    return pressure_force;
}

static glm::vec2 CalculateExternalForces(size_t index) {
    glm::vec2 gravity_force = glm::vec2(0.0f, GRAVITY);

    float interaction_strength = 0.0f;
    if (Input::Pressed(MouseButton::Left)) {
        interaction_strength = -INTERACTION_STRENGTH;
    } else if (Input::Pressed(MouseButton::Right)) {
        interaction_strength = INTERACTION_STRENGTH;
    }

    if (!approx_equals(interaction_strength, 0.0f)) {
        const glm::vec2 input_pos = Input::MouseScreenPosition();
        const glm::vec2 offset = input_pos - g.positions[index];
        const float sqr_dst = glm::dot(offset, offset);

        if (sqr_dst < INTERACTION_RADIUS * INTERACTION_RADIUS) {
            const float dst = glm::sqrt(sqr_dst);
            const glm::vec2 dir = dst > glm::epsilon<float>() ? offset / dst : glm::vec2(0.0f);
            float scale = 1.0f - dst / INTERACTION_RADIUS;
            return (dir * interaction_strength - g.velocities[index]) * scale;
        }
    }

    return gravity_force;
}

static glm::vec2 CalculateViscosityForce(size_t index) {
    glm::vec2 viscosity_force = glm::vec2(0.0f);

    const glm::vec2 point = g.positions[index];

    for (size_t i = 0; i < PARTICLE_COUNT; ++i) {
        glm::vec2 offset = g.positions[i] - point;
        float sqr_dst = glm::dot(offset, offset);

        if (sqr_dst > SMOOTHING_RADIUS * SMOOTHING_RADIUS) continue;

        float dst = glm::sqrt(sqr_dst);
        float influence = ViscosityKernel(dst / SMOOTHING_RADIUS, KERNEL_H);
        viscosity_force += (g.velocities[i] - g.velocities[index]) * influence;
    }

    return viscosity_force * VISCOSITY_STRENGTH;
}

static void ResolveCollisions(glm::vec2& position, glm::vec2& velocity) {
    const float width = g.camera.viewport().x;
    const float height = g.camera.viewport().y;

    if (position.x + PARTICLE_SIZE > width) {
        position.x = width - PARTICLE_SIZE;
        velocity.x *= -1.0f * COLLISION_DAMPING;
    }

    if (position.y + PARTICLE_SIZE > height) {
        position.y = height - PARTICLE_SIZE;
        velocity.y *= -1.0f * COLLISION_DAMPING;
    }

    if (position.x < 0) {
        position.x = 0;
        velocity.x *= -1.0f * COLLISION_DAMPING;
    }

    if (position.y < 0) {
        position.y = 0;
        velocity.y *= -1.0f * COLLISION_DAMPING;
    }
}

void pre_update() {

}


template <size_t SIZE>
LinearRgba MultiColorLerp(const LinearRgba (&colors)[SIZE], float t) {
    t = glm::clamp(t, 0.0f, 1.0f);

    float delta = 1.0f / (SIZE - 1);
    int startIndex = (int)(t / delta);

    if (startIndex == SIZE - 1) {
        return colors[SIZE - 1];
    }

    float localT = std::fmod(t, delta) / delta;

    return (colors[startIndex] * (1.0f - localT)) + (colors[startIndex + 1] * localT);
}

struct GradientKey {
    LinearRgba color;
    float position;
};

template <size_t SIZE>
LinearRgba GradientEvaluate(const GradientKey (&keys)[SIZE], float position) {
    if (SIZE == 0)
        return LinearRgba::black();

    if (position <= keys[0].position)
        return keys[0].color;

    if (position >= keys[SIZE - 1].position)
        return keys[SIZE - 1].color;

    for (size_t i = 0; i < SIZE - 1; ++i) {
        if (position >= keys[i].position && position <= keys[i + 1].position) {
            float t = (position - keys[i].position) / (keys[i + 1].position - keys[i].position);
            return keys[i].color.lerp(keys[i + 1].color, t);
        }
    }

    return LinearRgba::black();
}

static constexpr GradientKey GRADIENT[] = {
    GradientKey{LinearRgba(13, 72, 209), 0.0f},
    GradientKey{LinearRgba(73, 214, 153), 0.45f},
    GradientKey{LinearRgba(235, 215, 66), 0.68f},
    GradientKey{LinearRgba(222, 33, 33), 1.0f},
};

void fixed_update() {
    const float dt = Time::FixedDeltaSeconds();

    g.pool.submit_loop(0, PARTICLE_COUNT, [dt](size_t i) {
        g.velocities[i] += CalculateExternalForces(i) * dt;
        g.predicted_positions[i] = g.positions[i] + g.velocities[i] * dt;
        auto [density, near_density] = CalculateDensity(i);
        g.densities[i] = density;
        g.near_densities[i] = near_density;
    }).wait();

    g.pool.submit_loop(0, PARTICLE_COUNT, [dt](size_t i) {
        glm::vec2 pressure_force = CalculatePressureForce(i);
        glm::vec2 pressure_accel = pressure_force / g.densities[i];
        g.velocities[i] += pressure_accel * dt;
    }).wait();

    g.pool.submit_loop(0, PARTICLE_COUNT, [dt](size_t i) {
        glm::vec2 viscosity_force = CalculateViscosityForce(i);
        g.velocities[i] += viscosity_force * dt;
    }).wait();

    float velocity_max = 8.0f;
    for (size_t i = 0; i < PARTICLE_COUNT; ++i) {
        const glm::vec2& v = g.velocities[i];
        velocity_max = std::max(velocity_max, glm::dot(v, v));
    }

    g.pool.submit_loop(0, PARTICLE_COUNT, [dt, velocity_max](size_t i) {
        ResolveCollisions(g.positions[i], g.velocities[i]);
        const glm::vec2 v = g.velocities[i];
        float t = glm::dot(v, v) / velocity_max;
        g.colors[i] = GradientEvaluate(GRADIENT, t);
        g.positions[i] += g.velocities[i] * dt;
    }).wait();
}

void update() {

}

void post_update() {

}

void render() {
    Renderer& renderer = Engine::Renderer();

    renderer.Begin(g.camera);

    g.batch.BeginOrderMode();
    {
        for (size_t i = 0; i < PARTICLE_COUNT; ++i) {
            g.batch.DrawCircle(g.positions[i], {
                .radius = PARTICLE_SIZE / 2.0f,
                .color = g.colors[i],
                .anchor = Anchor::TopLeft
            });
        }
    }
    g.batch.EndOrderMode();

    renderer.BeginMainPass();
        renderer.Clear(LLGL::ClearValue(0.0f, 0.0f, 0.0f, 1.0f));

        renderer.PrepareBatch(g.batch);
        renderer.UploadBatchData();
        renderer.RenderBatch(g.batch);

        g.batch.Reset();
    renderer.EndPass();

    renderer.End();
}

void post_render() {
#if SGE_DEBUG
    if (Input::Pressed(Key::C)) {
        Engine::Renderer().PrintDebugInfo();
    }
#endif
}

void window_resized(uint32_t width, uint32_t height, uint32_t w , uint32_t h) {
    g.camera.set_viewport(glm::uvec2(width, height));
    g.camera.update();
    InitParticles();
    render();
}

bool App::Init(RenderBackend backend, AppConfig config) {
    Engine::SetPreUpdateCallback(pre_update);
    Engine::SetUpdateCallback(update);
    Engine::SetPostUpdateCallback(post_update);
    Engine::SetFixedUpdateCallback(fixed_update);
    Engine::SetRenderCallback(render);
    Engine::SetPostRenderCallback(post_render);
    Engine::SetWindowResizeCallback(window_resized);

    glm::uvec2 window_size = glm::uvec2(1280, 720);

    WindowSettings settings;
    settings.width = window_size.x;
    settings.height = window_size.y;
    settings.fullscreen = config.fullscreen;
    settings.vsync = config.vsync;
    settings.hidden = true;
    settings.samples = 8;

    LLGL::Extent2D resolution;
    if (!Engine::Init(backend, settings, resolution)) return false;

    Time::SetFixedTimestepSeconds(1.0 / 120.0);

    g.camera.set_viewport({resolution.width, resolution.height});
    g.camera.set_zoom(1.0f);
    g.camera.update();

    g.batch.SetIsUi(true);

    Engine::ShowWindow();

    g.positions.resize(PARTICLE_COUNT);
    g.predicted_positions.resize(PARTICLE_COUNT);
    g.velocities.resize(PARTICLE_COUNT);
    g.densities.resize(PARTICLE_COUNT);
    g.near_densities.resize(PARTICLE_COUNT);
    g.colors.resize(PARTICLE_COUNT);

    InitParticles();

    return true;
}

void App::Run() {
    Engine::Run();
}

void App::Destroy() {
    Engine::Destroy();
}

