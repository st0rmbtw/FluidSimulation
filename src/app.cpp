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
#include <SGE/utils/gradient.hpp>

#include <glm/trigonometric.hpp>
#include <glm/gtc/random.hpp>
#include <glm/ext/scalar_constants.hpp>

#include "thread_pool.hpp"

using namespace sge;

static constexpr float PI = glm::pi<float>();
static constexpr float PIXELS_IN_METER = 5.0f;
static constexpr float PIXEL_TO_METER = 1.0f / PIXELS_IN_METER;
static constexpr float METER_TO_PIXEL = PIXELS_IN_METER;

static constexpr size_t PARTICLE_COUNT = 6000;
static constexpr float PARTICLE_SIZE = 7.0f * PIXEL_TO_METER; // m
static constexpr float GRAVITY = 9.8f * 100.0f * PIXEL_TO_METER;
static constexpr float MASS = 18.0f * (PARTICLE_SIZE * PARTICLE_SIZE);
static constexpr float SMOOTHING_RADIUS = 3.0f * PARTICLE_SIZE * 0.5f; // m
static constexpr float COLLISION_DAMPING = 0.9f;
static constexpr float VISCOSITY_STRENGTH = 0.1f;
static constexpr float TARGET_DENSITY = 1000.0f * (PARTICLE_SIZE * PARTICLE_SIZE);

static constexpr float SPEED_OF_SOUND = 20.0f; // m/s
static constexpr float PRESSURE_MULTIPLIER = 100.0f * PIXELS_IN_METER * MASS;

static constexpr float INTERACTION_RADIUS = 200.0f * PIXEL_TO_METER;
static constexpr float PULL_INTERACTION_STRENGTH = PRESSURE_MULTIPLIER * 3.5f;
static constexpr float PUSH_INTERACTION_STRENGTH = PULL_INTERACTION_STRENGTH * 2.0f;

static constexpr float LOOKUP_RADIUS = SMOOTHING_RADIUS;

static constexpr glm::ivec2 CELL_OFFSETS[] = {
    glm::ivec2(0, 0),
    glm::ivec2(-1, 0),
    glm::ivec2(-1, -1),
    glm::ivec2(0, -1),
    glm::ivec2(1, -1),
    glm::ivec2(1, 0),
    glm::ivec2(1, 1),
    glm::ivec2(0, 1),
    glm::ivec2(-1, 1),
};

static constexpr GradientKey GRADIENT[] = {
    GradientKey{LinearRgba(13, 72, 209), 0.0f},
    GradientKey{LinearRgba(73, 214, 153), 0.45f},
    GradientKey{LinearRgba(235, 215, 66), 0.68f},
    GradientKey{LinearRgba(222, 33, 33), 1.0f},
};

struct Cell {
    size_t index;
    size_t cell_key;
};

static struct AppState {
    Camera camera = Camera(CameraOrigin::TopLeft);
    BS::thread_pool<> pool;
    Batch batch;
    std::vector<glm::vec2> positions;
    std::vector<glm::vec2> predicted_positions;
    std::vector<glm::vec2> velocities;
    std::vector<glm::vec2> forces;
    std::vector<float> densities;
    std::vector<LinearRgba> colors;
    std::vector<Rect> obstacles;
    std::vector<Cell> spatial_lookup;
    std::vector<size_t> start_indices;
    Rect selection_rect;
    float interaction_strength = 0.0f;
    bool paused = false;
    bool gravity = false;
    bool selection = false;
    bool show_debug_info = false;
} g;

static inline float Poly6KernelScale(float radius) {
    return 4.0f / (PI * glm::pow(radius, 8.0f));
}

static float Poly6Kernel(float dst, float radius) {
    if (dst >= radius) return 0.0f;

    float v = (radius*radius) - (dst*dst);
    return v * v * v;
}

static float SpikyKernelDerivativeScale(float radius) {
    return -30.0f / (PI * glm::pow(radius, 5.0f));
}

static float SpikyKernelDerivative(float dst, float radius) {
    if (dst >= radius) return 0.0f;

    float v = radius - dst;
    return v * v;
}

static float Poly6KernelDerivative(float dst, float radius) {
    // const float ratio = dst / radius;
    // if (ratio > 2.0f) {
    //     return 0.0f;
    // }
    // const float normalization = 15.0f / (14.0f * PI * radius*radius);
    // if (0.0f <= ratio && ratio < 1.0f) {
    //     return normalization * (9.0f * ratio*ratio - 12.0 * ratio);
    // }
    // return 3.0f * normalization * (2.0f - ratio) * (2.0f - ratio);

    if (dst >= radius) return 0.0f;

    float v = radius - dst;
    float scale = 12.0f / (glm::pow(radius, 4.0f) * glm::pi<float>());
    return -v * scale;
}

// float ViscosityKernel(float dst, float radius) {
//     if (dst >= radius) return 0.0f;

//     float v = radius - dst;
//     float scale = 40.0f / (PI * glm::pow(radius, 5.0f));
//     return v * scale;
// }

static float SpikySmoothingKernel(float dst, float radius) {
    if (dst >= radius) {
        return 0.0f;
    }
    const float volume = PI * glm::pow(radius, 6.0f) / 15.0f;
    const float v = (radius - dst);
    return v * v * v / volume;
}

static float SbSmoothingKernel(float dst, float radius) {
    if (dst >= radius) {
        return 0.0f;
    }
    const float v = radius - dst;

    return v * v * 6.0f / (PI * glm::pow(radius, 4.0f));
}

static float sb_smoothing_kernel_derivative(float dst, float radius) {
    if (dst >= radius) {
        return 0.0f;
    }
    return (dst - radius) * 12.0 / (PI * glm::pow(radius, 4.0f));
}

static float CubicSplineKernel(float dst, float h) {
    float coeff = 40.0f / 7.0f / PI;
    coeff /= (h * h);
    const float q = dst / h;
    float kernel_val = 0.0;
    if (q <= 1.0) {
        if (q <= 0.5)
            kernel_val = coeff * (1.0f - 6.0f * (q * q) + 6.0f * (q * q * q));
        else
            kernel_val = coeff * (2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q));
    }
    return kernel_val;
}

static glm::vec2 CubicSplineKernelDerivative(glm::vec2 r, float dst, float h) {
    float coeff = 80.0f / 7.0f / PI;
    coeff /= (h * h * h);
    glm::vec2 derivative = glm::vec2(0.0f);
    float q = dst / h;
    glm::vec2 r_hat = dst > 1e-7f ? r / dst : r / (dst + 1e-7f);
    if (q <= 1.0) {
        if (q <= 0.5)
            derivative = coeff * (9.0f * q * q - 6.0f * q) * r_hat;
        else
            derivative = coeff * (-3.0f * (1.0f - q) * (1.0f - q)) * r_hat;
    }
    return derivative;
}

static float Poly6ScalingFactor(float radius) {
    return 4.0f / (PI * glm::pow(radius, 8.0f));
}
static float SpikyPow3ScalingFactor(float radius) {
    return 10.0f / (PI * glm::pow(radius, 5.0f));
}
static float SpikyPow2ScalingFactor(float radius) {
    return 6.0f / (PI * glm::pow(radius, 4.0f));
}
static float SpikyPow3DerivativeScalingFactor(float radius) {
    return 30.0f / (glm::pow(radius, 5.0f) * PI);
}

static float SpikyPow2DerivativeScalingFactor(float radius) {
    return 12.0f / (glm::pow(radius, 4.0f) * PI);
}

static float SmoothingKernelPoly6(float dst, float radius) {
	if (dst >= radius) return 0.0f;

    float v = radius * radius - dst * dst;
    return v * v * v * Poly6ScalingFactor(radius);
}

static float SpikyKernelPow3(float dst, float radius) {
    if (dst >= radius) return 0.0f;

    float v = radius - dst;
    return v * v * v * SpikyPow3ScalingFactor(radius);
}

static float SpikyKernelPow2(float dst, float radius) {
	if (dst >= radius) return 0.0f;

    float v = radius - dst;
    return v * v * SpikyPow2ScalingFactor(radius);
}

static float DerivativeSpikyPow3(float dst, float radius) {
	if (dst >= radius) return 0.0f;

    float v = radius - dst;
    return -v * v * SpikyPow3DerivativeScalingFactor(radius);
}

static float DerivativeSpikyPow2(float dst, float radius) {
	if (dst >= radius) return 0.0f;

    float v = radius - dst;
    return -v * SpikyPow2DerivativeScalingFactor(radius);
}

static inline float DensityKernel(float dst, float radius)
{
	return Poly6Kernel(dst, radius);
}

static inline float DensityKernelScale(float radius)
{
	return Poly6KernelScale(radius);
}

static inline float DensityKernelDerivative(float dst, float radius)
{
	return SpikyKernelDerivative(dst, radius);
}

static inline float DensityKernelDerivativeScale(float radius)
{
	return SpikyKernelDerivativeScale(radius);
}

static inline float ViscosityKernel(float dst, float radius)
{
	return SmoothingKernelPoly6(dst, radius);
}

static glm::ivec2 PositionToCellCoord(glm::vec2 position, float radius) {
    int x = position.x / radius;
    int y = position.y / radius;
    return {x, y};
}

static size_t HashCell(glm::ivec2 pos) {
    size_t a = pos.x * 15823;
    size_t b = pos.y * 9737333;
    return a + b;
}

static size_t GetKeyFromHash(size_t hash) {
    return hash % g.spatial_lookup.size();
}

template <typename F>
static void ForEachNeighbor(glm::vec2 position, const std::vector<glm::vec2>& positions, F&& func) {
    static constexpr float SQR_RADIUS = SMOOTHING_RADIUS * SMOOTHING_RADIUS;

    const glm::ivec2 center = PositionToCellCoord(position, LOOKUP_RADIUS);

    for (const glm::ivec2& offset : CELL_OFFSETS) {
        size_t key = GetKeyFromHash(HashCell(center + offset));
        size_t start_index = g.start_indices[key];

        for (size_t i = start_index; i < g.spatial_lookup.size(); ++i) {
            if (g.spatial_lookup[i].cell_key != key) break;

            size_t particle_index = g.spatial_lookup[i].index;
            glm::vec2 offset = positions[particle_index] - position;
            float sqr_dst = glm::dot(offset, offset);

            if (sqr_dst < SQR_RADIUS) {
                float dst = glm::sqrt(sqr_dst);
                std::forward<F>(func)(particle_index, offset, dst);
            }
        }
    }
}

// template <typename F>
// static void ForEachNeighbor(glm::vec2 position, const std::vector<glm::vec2>& positions, F&& func) {
//     static constexpr float SQR_RADIUS = SMOOTHING_RADIUS * SMOOTHING_RADIUS;

//     for (size_t j = 0; j < PARTICLE_COUNT; ++j) {
//         glm::vec2 offset = positions[j] - position;
//         float sqr_dst = glm::dot(offset, offset);

//         if (sqr_dst < SQR_RADIUS) {
//             float dst = glm::sqrt(sqr_dst);
//             std::forward<F>(func)(j, offset, dst);
//         }
//     }
// }

static inline glm::vec2 RandomDir() {
    return glm::linearRand(glm::vec2(-1.0f), glm::vec2(1.0f));
}

static float CalculateDensity(size_t index) {
    float density = DensityKernel(0.0f, SMOOTHING_RADIUS) * MASS;

    const glm::vec2 point = g.predicted_positions[index];

    ForEachNeighbor(point, g.predicted_positions, [index, &density](size_t i, glm::vec2 offset, float dst) {
        if (i != index) {
            density += MASS * DensityKernel(dst, SMOOTHING_RADIUS);
        }
    });

    return density * DensityKernelScale(SMOOTHING_RADIUS);
}

static void InitParticles() {
    const float width = g.camera.viewport().x;
    const float height = g.camera.viewport().y;

    float x = (width / 2.0f) * PIXEL_TO_METER;
    float y = (height / 2.0f) * PIXEL_TO_METER;

    // Spawn in spiral order
    size_t s = 1;
    int direction = 0;
    size_t index = 0;
    while (index < PARTICLE_COUNT) {
        for (int i = 0; i < glm::min(PARTICLE_COUNT, index+s)-index; ++i) {
            glm::vec2 pos = glm::vec2(x, y);
            g.positions[index] = pos;
            g.predicted_positions[index] = pos;
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

    g.pool.submit_loop(0, PARTICLE_COUNT, [](size_t i) {
        g.densities[i] = CalculateDensity(i);
    }).wait();
}

static float ConvertDensityToPressure(float density) {
    // density = glm::max(density, TARGET_DENSITY);
    // float ratio = density / TARGET_DENSITY;
    // float pressure = PRESSURE_MULTIPLIER * (glm::pow(ratio, 7.0f) - 1.0f);

    float density_error = TARGET_DENSITY - density;
    float pressure = density_error * PRESSURE_MULTIPLIER;
    return pressure;
}

static glm::vec2 CalculatePressureForce(size_t index) {
    glm::vec2 pressure_force = glm::vec2(0.0f);

    const glm::vec2 point = g.predicted_positions[index];

    float density_i = g.densities[index];
    float pressure_i = ConvertDensityToPressure(density_i);

    ForEachNeighbor(point, g.predicted_positions, [index, density_i, pressure_i, &pressure_force](size_t j, glm::vec2 offset, float dst) {
        if (j == index) return;

        const glm::vec2 dir = dst > glm::epsilon<float>() ? (offset / dst) : RandomDir();

        const glm::vec2 slope = DensityKernelDerivative(dst, SMOOTHING_RADIUS) * dir;

        const float density_j = g.densities[j];
        const float pressure_j = ConvertDensityToPressure(density_j);

        const float shared_pressure = (pressure_i + pressure_j) * 0.5f;

        // pressure_force += MASS * (pressure_i / (density_i * density_i) + pressure_j / (density_j * density_j)) * slope;
        pressure_force -= MASS * shared_pressure * slope / density_j;

    });

    return pressure_force * DensityKernelDerivativeScale(SMOOTHING_RADIUS);
}

static glm::vec2 CalculateExternalForces(size_t index) {
    glm::vec2 gravity_force = glm::vec2(0.0f);

    if (g.gravity) {
        gravity_force.y = GRAVITY;
    }

    if (!approx_equals(g.interaction_strength, 0.0f)) {
        const glm::vec2 input_pos = Input::MouseScreenPosition() * PIXEL_TO_METER;
        const glm::vec2 offset = input_pos - g.positions[index];
        const float sqr_dst = glm::dot(offset, offset);

        if (sqr_dst < INTERACTION_RADIUS * INTERACTION_RADIUS) {
            const float dst = glm::sqrt(sqr_dst);
            const glm::vec2 dir = dst > glm::epsilon<float>() ? offset / dst : RandomDir();
            float scale = 1.0f - dst / INTERACTION_RADIUS;
            return (dir * g.interaction_strength - g.velocities[index]) * scale;
        }
    }

    return gravity_force * MASS;
}

static glm::vec2 CalculateViscosityForce(size_t index) {
    glm::vec2 viscosity_force = glm::vec2(0.0f);

    const glm::vec2 point = g.positions[index];

    ForEachNeighbor(point, g.positions, [index, &viscosity_force](size_t i, glm::vec2 offset, float dst) {
        if (index == i) return;

        float influence = ViscosityKernel(dst, SMOOTHING_RADIUS);
        float density = g.densities[i];
        viscosity_force += MASS * (g.velocities[i] - g.velocities[index]) * influence;
    });

    return viscosity_force * VISCOSITY_STRENGTH;
}

static void ResolveCollisions(glm::vec2& position, glm::vec2& velocity) {
    const float width = (g.camera.viewport().x) * PIXEL_TO_METER;
    const float height = (g.camera.viewport().y) * PIXEL_TO_METER;

    const float half_size = PARTICLE_SIZE / 2.0f;

    if (position.x + half_size > width) {
        position.x = width - half_size;
        velocity.x *= -1.0f * COLLISION_DAMPING;
    }

    if (position.y + half_size > height) {
        position.y = height - half_size;
        velocity.y *= -1.0f * COLLISION_DAMPING;
    }

    if (position.x - half_size < 0) {
        position.x = half_size;
        velocity.x *= -1.0f * COLLISION_DAMPING;
    }

    if (position.y - half_size < 0) {
        position.y = half_size;
        velocity.y *= -1.0f * COLLISION_DAMPING;
    }

    for (const Rect& b : g.obstacles) {
        const Rect a = Rect::from_center_size(position, glm::vec2(PARTICLE_SIZE));

        // check to see if the two rectangles are intersecting
        if (a.left() < b.right() && a.right() > b.left() && a.top() > b.bottom() && a.bottom() < b.top()) {

            // check to see if we hit on the left or right side
            if (a.left() < b.left() && a.right() > b.left() && a.right() < b.right()) {
                position.x = b.left() - half_size;
                velocity.x = (b.left() - a.right()) * COLLISION_DAMPING;
            } else if (a.left() > b.left() && a.left() < b.right() && a.right() > b.right()) {
                position.x = b.right() + half_size;
                velocity.x = (a.left() - b.right()) * COLLISION_DAMPING;
            }

            // check to see if we hit on the top or bottom side
            if (a.bottom() < b.bottom() && a.top() > b.bottom() && a.top() < b.top()) {
                position.y = b.bottom() - half_size;
                velocity.y = (b.bottom() - a.top()) * COLLISION_DAMPING;
            } else if (a.bottom() > b.bottom() && a.bottom() < b.top() && a.top() > b.top()) {
                position.y = b.top() + half_size;
                velocity.y = (a.bottom() - b.top()) * COLLISION_DAMPING;
            }
        }
    }
}

static void UpdateSpatialLookup(float radius) {
    for (size_t i = 0; i < PARTICLE_COUNT; ++i) {
        const glm::ivec2 coord = PositionToCellCoord(g.positions[i], radius);
        size_t cell_key = GetKeyFromHash(HashCell(coord));
        g.spatial_lookup[i] = Cell{i, cell_key};
        g.start_indices[i] = SIZE_MAX;
    };

    std::sort(
        g.spatial_lookup.begin(),
        g.spatial_lookup.end(),
        [](const Cell& a, const Cell& b) {
            return a.cell_key < b.cell_key;
        }
    );

    for (size_t i = 0; i < PARTICLE_COUNT; ++i) {
        size_t key = g.spatial_lookup[i].cell_key;
        size_t prev_key = i == 0 ? SIZE_MAX : g.spatial_lookup[i - 1].cell_key;
        if (key != prev_key) {
            g.start_indices[key] = i;
        }
    };
}

static void FixedUpdate() {
    const float dt = Time::FixedDeltaSeconds();

    UpdateSpatialLookup(LOOKUP_RADIUS);

    g.pool.submit_loop(0, PARTICLE_COUNT, [dt](size_t i) {
        g.forces[i] += CalculateViscosityForce(i);
    }).wait();

    g.pool.submit_loop(0, PARTICLE_COUNT, [dt](size_t i) {
        g.forces[i] += CalculateExternalForces(i);
    }).wait();

    g.pool.submit_loop(0, PARTICLE_COUNT, [dt](size_t i) {
        const glm::vec2 acceleration = g.forces[i] / MASS * dt;
        g.predicted_positions[i] = g.positions[i] + g.velocities[i] * dt + acceleration * dt;
        g.densities[i] = CalculateDensity(i);
    }).wait();

    g.pool.submit_loop(0, PARTICLE_COUNT, [dt](size_t i) {
        g.forces[i] -= CalculatePressureForce(i) / g.densities[i];
    }).wait();

    g.pool.submit_loop(0, PARTICLE_COUNT, [dt](size_t i) {
        glm::vec2 acceleration = g.forces[i] / MASS;
        g.velocities[i] += acceleration * dt;

        glm::vec2 next_position = g.positions[i] + g.velocities[i] * dt;
        ResolveCollisions(next_position, g.velocities[i]);

        g.positions[i] = next_position;
        g.forces[i] = glm::zero<glm::vec2>();
    }).wait();

    float velocity_max = 10.0f;
    for (size_t i = 0; i < PARTICLE_COUNT; ++i) {
        const glm::vec2& v = g.velocities[i];
        velocity_max = std::max(velocity_max, glm::dot(v, v));
    }

    g.pool.submit_loop(0, PARTICLE_COUNT, [dt, velocity_max](size_t i) {
        const glm::vec2 v = g.velocities[i];
        float t = glm::dot(v, v) / velocity_max;
        g.colors[i] = GradientEvaluate(GRADIENT, t);
    }).wait();
}

static void Update() {
    if (Input::JustPressed(Key::G)) {
        g.gravity = !g.gravity;
    }

    if (Input::JustPressed(Key::D)) {
        g.show_debug_info = !g.show_debug_info;
    }

    if (Input::JustPressed(Key::C)) {
        g.obstacles.clear();
    }

    if (Input::JustPressed(Key::R)) {
        InitParticles();
    }

    if (Input::Pressed(Key::LeftShift) && Input::Pressed(MouseButton::Left)) {
        if (!g.selection) {
            g.selection_rect = Rect::from_top_left(Input::MouseScreenPosition(), glm::vec2(0.0f));
        }
        g.selection_rect.max = Input::MouseScreenPosition();
        g.selection = true;
    } else if (Input::Pressed(Key::LeftShift) && g.selection) {
        if (g.selection_rect.width() > 0.0f && g.selection_rect.height() > 0.0f) {
            g.obstacles.push_back(g.selection_rect * PIXEL_TO_METER);
        }
        g.selection = false;
    } else {
        g.selection = false;
    }

    g.interaction_strength = 0.0f;

    if (!g.selection) {
        if (Input::Pressed(MouseButton::Left)) {
            g.interaction_strength = -PUSH_INTERACTION_STRENGTH;
        } else if (Input::Pressed(MouseButton::Right)) {
            g.interaction_strength = PULL_INTERACTION_STRENGTH;
        }
    }
}

static void Render() {
    Renderer& renderer = Engine::Renderer();

    renderer.Begin(g.camera);

    g.batch.BeginOrderMode();
    {
        if (g.selection) {
            g.batch.DrawRect(g.selection_rect.min, {
                .size = g.selection_rect.size(),
                .color = LinearRgba::transparent(),
                .border_thickness = 1.0f,
                .border_color = LinearRgba(73, 214, 153),
                .anchor = Anchor::TopLeft
            });
        }

        for (const Rect& rect : g.obstacles) {
            g.batch.DrawRect(rect.min * METER_TO_PIXEL, {
                .size = rect.size() * METER_TO_PIXEL,
                .color = LinearRgba::transparent(),
                .border_thickness = 1.0f,
                .border_color = LinearRgba(73, 214, 153),
                .anchor = Anchor::TopLeft
            });
        }

        if (g.show_debug_info) {
            g.batch.DrawCircle(Input::MouseScreenPosition(), {
                .radius = LOOKUP_RADIUS * 2.0f * METER_TO_PIXEL,
                .color = LinearRgba::transparent(),
                .border_thickness = 1.0f,
                .border_color = LinearRgba(73, 214, 153),
            });

            const glm::ivec2 center = PositionToCellCoord(Input::MouseScreenPosition() * PIXEL_TO_METER, LOOKUP_RADIUS);

            size_t keys[9] = {};
            for (size_t i = 0; i < 9; ++i) {
                keys[i] = GetKeyFromHash(HashCell(center + CELL_OFFSETS[i]));
            }

            for (size_t i = 0; i < g.spatial_lookup.size(); ++i) {
                const size_t index = g.spatial_lookup[i].index;

                LinearRgba color = g.colors[index];
                const glm::vec2 pos = g.positions[index] * METER_TO_PIXEL;

                for (size_t j = 0; j < 9; ++j) {
                    if (g.spatial_lookup[i].cell_key == keys[j]) {
                        color = LinearRgba::red();
                        break;
                    }
                }

                g.batch.DrawCircle(pos, {
                    .radius = PARTICLE_SIZE / 2.0f * METER_TO_PIXEL,
                    .color = color,
                    .anchor = Anchor::Center
                });
            }
        } else {
            for (size_t i = 0; i < PARTICLE_COUNT; ++i) {
                const LinearRgba& color = g.colors[i];
                const glm::vec2 pos = g.positions[i] * METER_TO_PIXEL;

                g.batch.DrawCircle(pos, {
                    .radius = PARTICLE_SIZE / 2.0f * METER_TO_PIXEL,
                    .color = color,
                    .anchor = Anchor::Center
                });
            }
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

static void PostRender() {
#if SGE_DEBUG
    if (Input::Pressed(Key::C)) {
        Engine::Renderer().PrintDebugInfo();
    }
#endif
}

static void WindowResized(uint32_t width, uint32_t height, uint32_t w , uint32_t h) {
    g.camera.set_viewport(glm::uvec2(width, height));
    g.camera.update();
    InitParticles();
    Render();
}

bool App::Init(RenderBackend backend, AppConfig config) {
    Engine::SetUpdateCallback(Update);
    Engine::SetFixedUpdateCallback(FixedUpdate);
    Engine::SetRenderCallback(Render);
    Engine::SetPostRenderCallback(PostRender);
    Engine::SetWindowResizeCallback(WindowResized);

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

    Time::SetFixedTimestepSeconds(1.0 / 360.0);

    g.camera.set_viewport({resolution.width, resolution.height});
    g.camera.set_zoom(1.0f);
    g.camera.update();

    g.batch.SetIsUi(true);

    Engine::ShowWindow();

    g.positions.resize(PARTICLE_COUNT);
    g.predicted_positions.resize(PARTICLE_COUNT);
    g.velocities.resize(PARTICLE_COUNT);
    g.densities.resize(PARTICLE_COUNT);
    g.colors.resize(PARTICLE_COUNT);
    g.forces.resize(PARTICLE_COUNT);

    g.spatial_lookup.resize(PARTICLE_COUNT);
    g.start_indices.resize(PARTICLE_COUNT);

    InitParticles();

    return true;
}

void App::Run() {
    Engine::Run();
}

void App::Destroy() {
    Engine::Destroy();
}

