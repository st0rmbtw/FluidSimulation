#include "app.hpp"
#include "spatial_lookup.hpp"

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
#include <SGE/profile.hpp>

#include <glm/trigonometric.hpp>
#include <glm/gtc/random.hpp>
#include <glm/ext/scalar_constants.hpp>

using namespace sge;

static constexpr float PI = glm::pi<float>();

static constexpr size_t PARTICLE_COUNT = 12000;
static constexpr float PIXELS_IN_METER = 3.0f;
static constexpr float PIXEL_TO_METER = 1.0f / PIXELS_IN_METER;
static constexpr float METER_TO_PIXEL = PIXELS_IN_METER;

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
static constexpr float PULL_INTERACTION_STRENGTH = PRESSURE_MULTIPLIER * 2.0f;
static constexpr float PUSH_INTERACTION_STRENGTH = PULL_INTERACTION_STRENGTH * 2.0f;

static constexpr float LOOKUP_RADIUS = SMOOTHING_RADIUS;

static constexpr GradientKey GRADIENT[] = {
    GradientKey{LinearRgba(13, 72, 209), 0.0f},
    GradientKey{LinearRgba(73, 214, 153), 0.45f},
    GradientKey{LinearRgba(235, 215, 66), 0.68f},
    GradientKey{LinearRgba(222, 33, 33), 1.0f},
};

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

float App::CalculateDensity(size_t index) {
    float density = DensityKernel(0.0f, SMOOTHING_RADIUS) * MASS;

    const glm::vec2 point = m_predicted_positions[index];

    m_lookup.ForEachNeighbor(SMOOTHING_RADIUS, point, m_predicted_positions, [index, &density](size_t i, glm::vec2 offset, float dst) {
        if (i != index) {
            density += MASS * DensityKernel(dst, SMOOTHING_RADIUS);
        }
    });

    return density * DensityKernelScale(SMOOTHING_RADIUS);
}

void App::InitParticles() {
    const float width = m_camera.viewport().width;
    const float height = m_camera.viewport().height;

    float x = (width / 2.0f) * PIXEL_TO_METER;
    float y = (height / 2.0f) * PIXEL_TO_METER;

    // Spawn in spiral order
    size_t s = 1;
    int direction = 0;
    size_t index = 0;
    while (index < PARTICLE_COUNT) {
        for (int i = 0; i < glm::min(PARTICLE_COUNT, index+s)-index; ++i) {
            glm::vec2 pos = glm::vec2(x, y);
            m_positions[index] = pos;
            m_predicted_positions[index] = pos;
            m_velocities[index] = glm::vec2(0.0);
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

    m_pool.submit_loop(0, PARTICLE_COUNT, [this](size_t i) {
        m_densities[i] = CalculateDensity(i);
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

glm::vec2 App::CalculatePressureForce(size_t index) {
    glm::vec2 pressure_force = glm::vec2(0.0f);

    const glm::vec2 point = m_predicted_positions[index];

    float density_i = m_densities[index];
    float pressure_i = ConvertDensityToPressure(density_i);

    m_lookup.ForEachNeighbor(SMOOTHING_RADIUS, point, m_predicted_positions, [this, index, density_i, pressure_i, &pressure_force](size_t j, glm::vec2 offset, float dst) {
        if (j == index) return;

        const glm::vec2 dir = dst > glm::epsilon<float>() ? (offset / dst) : RandomDir();

        const glm::vec2 slope = DensityKernelDerivative(dst, SMOOTHING_RADIUS) * dir;

        const float density_j = m_densities[j];
        const float pressure_j = ConvertDensityToPressure(density_j);

        const float shared_pressure = (pressure_i + pressure_j) * 0.5f;

        // pressure_force += MASS * (pressure_i / (density_i * density_i) + pressure_j / (density_j * density_j)) * slope;
        pressure_force -= MASS * shared_pressure * slope / density_j;

    });

    return pressure_force * DensityKernelDerivativeScale(SMOOTHING_RADIUS);
}

glm::vec2 App::CalculateExternalForces(size_t index) {
    glm::vec2 gravity_force = glm::vec2(0.0f);

    if (m_gravity) {
        gravity_force.y = GRAVITY;
    }

    if (!approx_equals(m_interaction_strength, 0.0f)) {
        const glm::vec2 input_pos = Input::CursorPosition() * PIXEL_TO_METER;
        const glm::vec2 offset = input_pos - m_positions[index];
        const float sqr_dst = glm::dot(offset, offset);

        if (sqr_dst < INTERACTION_RADIUS * INTERACTION_RADIUS) {
            const float dst = glm::sqrt(sqr_dst);
            const glm::vec2 dir = dst > glm::epsilon<float>() ? offset / dst : RandomDir();
            float scale = 1.0f - dst / INTERACTION_RADIUS;
            return (dir * m_interaction_strength - m_velocities[index]) * scale;
        }
    }

    return gravity_force * MASS;
}

glm::vec2 App::CalculateViscosityForce(size_t index) {
    glm::vec2 viscosity_force = glm::vec2(0.0f);

    const glm::vec2 point = m_positions[index];

    m_lookup.ForEachNeighbor(SMOOTHING_RADIUS, point, m_positions, [this, index, &viscosity_force](size_t i, glm::vec2 offset, float dst) {
        if (index == i) return;

        float influence = ViscosityKernel(dst, SMOOTHING_RADIUS);
        float density = m_densities[i];
        viscosity_force += MASS * (m_velocities[i] - m_velocities[index]) * influence;
    });

    return viscosity_force * VISCOSITY_STRENGTH;
}

void App::ResolveCollisions(glm::vec2& position, glm::vec2& velocity) {
    const float width = (m_camera.viewport().width) * PIXEL_TO_METER;
    const float height = (m_camera.viewport().height) * PIXEL_TO_METER;

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

    for (const Rect& b : m_obstacles) {
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

void App::UpdateSpatialLookup(float radius) {
    for (size_t i = 0; i < PARTICLE_COUNT; ++i) {
        const glm::ivec2 coord = SpatialLookup::PositionToCellCoord(m_positions[i], radius);
        size_t cell_key = m_lookup.GetKeyFromHash(SpatialLookup::HashCell(coord));
        m_lookup.SetCell(i, SpatialLookup::Cell{i, cell_key});
        m_lookup.SetStartIndex(i, SIZE_MAX);
    };

    m_lookup.Sort();

    for (size_t i = 0; i < PARTICLE_COUNT; ++i) {
        size_t key = m_lookup.GetCell(i).key;
        size_t prev_key = i == 0 ? SIZE_MAX : m_lookup.GetCell(i - 1).key;
        if (key != prev_key) {
            m_lookup.SetStartIndex(key, i);
        }
    };
}

void App::OnFixedUpdate() {
    const float dt = Time::FixedDeltaSeconds();

    UpdateSpatialLookup(LOOKUP_RADIUS);

    {
        ZoneScopedN("CalculateViscosityForce Block");
        m_pool.submit_loop(0, PARTICLE_COUNT, [this, dt](size_t i) {
            m_forces[i] += CalculateViscosityForce(i);
        }).wait();
    }

    {
        ZoneScopedN("CalculateExternalForces Block");
        m_pool.submit_loop(0, PARTICLE_COUNT, [this, dt](size_t i) {
            m_forces[i] += CalculateExternalForces(i);
        }).wait();
    }

    {
        ZoneScopedN("CalculateDensity Block");
        m_pool.submit_loop(0, PARTICLE_COUNT, [this, dt](size_t i) {
            const glm::vec2 acceleration = m_forces[i] / MASS * dt;
            m_predicted_positions[i] = m_positions[i] + m_velocities[i] * dt + acceleration * dt;
            m_densities[i] = CalculateDensity(i);
        }).wait();
    }

    {
        ZoneScopedN("CalculatePressureForce Block");
        m_pool.submit_loop(0, PARTICLE_COUNT, [this, dt](size_t i) {
            m_forces[i] -= CalculatePressureForce(i) / m_densities[i];
        }).wait();
    }

    {
        ZoneScopedN("Update position & velocity Block");
        m_pool.submit_loop(0, PARTICLE_COUNT, [this, dt](size_t i) {
            glm::vec2 acceleration = m_forces[i] / MASS;
            m_velocities[i] += acceleration * dt;

            glm::vec2 next_position = m_positions[i] + m_velocities[i] * dt;
            ResolveCollisions(next_position, m_velocities[i]);

            m_positions[i] = next_position;
            m_forces[i] = glm::zero<glm::vec2>();
        }).wait();
    }

    float velocity_max = 10.0f;
    for (size_t i = 0; i < PARTICLE_COUNT; ++i) {
        const glm::vec2& v = m_velocities[i];
        velocity_max = std::max(velocity_max, glm::dot(v, v));
    }

    {
        ZoneScopedN("GradientEvaluation Block");
        m_pool.submit_loop(0, PARTICLE_COUNT, [this, dt, velocity_max](size_t i) {
            const glm::vec2 v = m_velocities[i];
            float t = glm::dot(v, v) / velocity_max;
            m_colors[i] = GradientEvaluate(GRADIENT, t);
        }).wait();
    }
}

void App::OnUpdate() {
    if (Input::JustPressed(Key::G)) {
        m_gravity = !m_gravity;
    }

    if (Input::JustPressed(Key::D)) {
        m_show_debug_info = !m_show_debug_info;
    }

    if (Input::JustPressed(Key::C)) {
        m_obstacles.clear();
    }

    if (Input::JustPressed(Key::R)) {
        InitParticles();
    }

    if (Input::Pressed(Key::LeftShift) && Input::Pressed(MouseButton::Left)) {
        if (!m_selection) {
            m_selection_rect = Rect::from_top_left(Input::CursorPosition(), glm::vec2(0.0f));
        }
        m_selection_rect.max = Input::CursorPosition();
        m_selection = true;
    } else if (Input::Pressed(Key::LeftShift) && m_selection) {
        if (m_selection_rect.width() > 0.0f && m_selection_rect.height() > 0.0f) {
            m_obstacles.push_back(m_selection_rect * PIXEL_TO_METER);
        }
        m_selection = false;
    } else {
        m_selection = false;
    }

    m_interaction_strength = 0.0f;

    if (!m_selection) {
        if (Input::Pressed(MouseButton::Left)) {
            m_interaction_strength = -PUSH_INTERACTION_STRENGTH;
        } else if (Input::Pressed(MouseButton::Right)) {
            m_interaction_strength = PULL_INTERACTION_STRENGTH;
        }
    }
}

void App::OnRender(const std::shared_ptr<sge::GlfwWindow>& window) {
    m_renderer->Begin();

    m_batch->BeginOrderMode();
    {
        if (m_selection) {
            m_batch->DrawRect(m_selection_rect.min, {
                .size = m_selection_rect.size(),
                .color = LinearRgba::transparent(),
                .border_thickness = 1.0f,
                .border_color = LinearRgba(73, 214, 153),
                .anchor = Anchor::TopLeft
            });
        }

        for (const Rect& rect : m_obstacles) {
            m_batch->DrawRect(rect.min * METER_TO_PIXEL, {
                .size = rect.size() * METER_TO_PIXEL,
                .color = LinearRgba::transparent(),
                .border_thickness = 1.0f,
                .border_color = LinearRgba(73, 214, 153),
                .anchor = Anchor::TopLeft
            });
        }

        if (m_show_debug_info) {
            m_batch->DrawCircle(Input::CursorPosition(), {
                .radius = LOOKUP_RADIUS * 2.0f * METER_TO_PIXEL,
                .color = LinearRgba::transparent(),
                .border_thickness = 1.0f,
                .border_color = LinearRgba(73, 214, 153),
            });

            const glm::ivec2 center = SpatialLookup::PositionToCellCoord(Input::CursorPosition() * PIXEL_TO_METER, LOOKUP_RADIUS);

            size_t keys[9] = {};
            for (size_t i = 0; i < 9; ++i) {
                keys[i] = m_lookup.GetKeyFromHash(SpatialLookup::HashCell(center + SpatialLookup::CELL_OFFSETS[i]));
            }

            for (size_t i = 0; i < m_lookup.GetSize(); ++i) {
                const size_t index = m_lookup.GetCell(i).index;

                LinearRgba color = m_colors[index];
                const glm::vec2 pos = m_positions[index] * METER_TO_PIXEL;

                for (size_t j = 0; j < 9; ++j) {
                    if (m_lookup.GetCell(i).key == keys[j]) {
                        color = LinearRgba::red();
                        break;
                    }
                }

                m_batch->DrawCircle(pos, {
                    .radius = PARTICLE_SIZE / 2.0f * METER_TO_PIXEL,
                    .color = color,
                    .anchor = Anchor::Center
                });
            }
        } else {
            for (size_t i = 0; i < PARTICLE_COUNT; ++i) {
                const LinearRgba& color = m_colors[i];
                const glm::vec2 pos = m_positions[i] * METER_TO_PIXEL;

                m_batch->DrawCircle(pos, {
                    .radius = PARTICLE_SIZE / 2.0f * METER_TO_PIXEL,
                    .color = color,
                    .anchor = Anchor::Center
                });
            }
        }
    }
    m_batch->EndOrderMode();

    m_renderer->BeginPass(window, m_camera);
        m_renderer->Clear(LLGL::ClearValue(0.0f, 0.0f, 0.0f, 1.0f));

        m_renderer->PrepareBatch(*m_batch);
        m_renderer->UploadBatchData();
        m_renderer->RenderBatch(*m_batch);

        m_batch->Reset();
    m_renderer->EndPass();

    m_renderer->End();
    GetRenderContext()->Present(window);
}

bool App::OnInit() {
    if (!InitRenderContext(m_config.backend))
        return false;

    glm::uvec2 window_size = glm::uvec2(1280, 720);

    WindowSettings window_settings;
    window_settings.width = window_size.x;
    window_settings.height = window_size.y;
    window_settings.fullscreen = m_config.fullscreen;
    window_settings.vsync = m_config.vsync;
    window_settings.hidden = true;
    window_settings.samples = 8;

    Time::SetFixedTimestepSeconds(1.0 / 360.0);

    auto result = CreateWindow(window_settings);
    if (!result.has_value()) {
        SGE_LOG_ERROR("Couldn't create a window: {}", result.error());
        return false;
    }

    std::shared_ptr<sge::GlfwWindow> window = result.value();
    m_primary_window_id = window->GetID();

    m_renderer = std::make_unique<Renderer>(GetRenderContext());
    m_batch = m_renderer->CreateBatch();
    m_batch->SetIsUi(true);

    m_camera.set_viewport(window->GetContentSize());
    m_camera.set_zoom(1.0f);
    m_camera.update();

    m_positions.resize(PARTICLE_COUNT);
    m_predicted_positions.resize(PARTICLE_COUNT);
    m_velocities.resize(PARTICLE_COUNT);
    m_densities.resize(PARTICLE_COUNT);
    m_colors.resize(PARTICLE_COUNT);
    m_forces.resize(PARTICLE_COUNT);

    m_lookup.Resize(PARTICLE_COUNT);

    InitParticles();

    window->ShowWindow();

    return true;
}

