#include "app.hpp"
#include "spatial_lookup.hpp"
#include "kernels.hpp"

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

// static constexpr size_t PARTICLE_COUNT = 12000;
// static constexpr float PIXELS_IN_METER = 1280.0f / 8.0f;
// static constexpr float PIXEL_TO_METER = 1.0f / PIXELS_IN_METER;
// static constexpr float METER_TO_PIXEL = PIXELS_IN_METER;

// static constexpr float PARTICLE_SIZE = 0.026f; // m

// static constexpr float GRAVITY = 9.8f;
// static constexpr float TARGET_DENSITY = 1000.0f;
// static constexpr float MASS = TARGET_DENSITY * PARTICLE_SIZE * PARTICLE_SIZE;

// static constexpr float SMOOTHING_RADIUS = 3.0f * PARTICLE_SIZE; // m
// static constexpr float COLLISION_DAMPING = 0.9f;
// static constexpr float VISCOSITY_STRENGTH = 0.15f;

// static constexpr float SPEED_OF_SOUND = 15.0f; // m/s
// static constexpr float PRESSURE_MULTIPLIER = SPEED_OF_SOUND * SPEED_OF_SOUND * TARGET_DENSITY / 7.0f;

// static constexpr float INTERACTION_RADIUS = 200.0f * PIXEL_TO_METER;
// static constexpr float PUSH_INTERACTION_STRENGTH = PRESSURE_MULTIPLIER / 10000.0f;
// static constexpr float PULL_INTERACTION_STRENGTH = PUSH_INTERACTION_STRENGTH / 2.0f;

static constexpr size_t PARTICLE_COUNT = 12000;
static constexpr float PIXELS_IN_METER = 3.0f;
static constexpr float PIXEL_TO_METER = 1.0f / PIXELS_IN_METER;
static constexpr float METER_TO_PIXEL = PIXELS_IN_METER;

static constexpr float PARTICLE_SIZE = 5.0f * PIXEL_TO_METER; // m

static constexpr float GRAVITY = 9.8f * 100.0f * PIXEL_TO_METER;
static constexpr float TARGET_DENSITY = 1000.0f;
static constexpr float MASS = 10.0f;

static constexpr float SMOOTHING_RADIUS = 2.0f * PARTICLE_SIZE; // m
static constexpr float COLLISION_DAMPING = 0.9f;
static constexpr float VISCOSITY_STRENGTH = 0.1f;

static constexpr float SPEED_OF_SOUND = 15.0f; // m/s
static constexpr float PRESSURE_MULTIPLIER = 100.0f * PIXELS_IN_METER * MASS;

static constexpr float INTERACTION_RADIUS = 200.0f * PIXEL_TO_METER;
static constexpr float PULL_INTERACTION_STRENGTH = PRESSURE_MULTIPLIER * 2.0f;
static constexpr float PUSH_INTERACTION_STRENGTH = PULL_INTERACTION_STRENGTH * 2.0f;

static constexpr float LOOKUP_RADIUS = 1.5f * SMOOTHING_RADIUS;

static constexpr GradientKey GRADIENT[] = {
    GradientKey{LinearRgba(13, 72, 209), 0.0f},
    GradientKey{LinearRgba(73, 214, 153), 0.45f},
    GradientKey{LinearRgba(235, 215, 66), 0.68f},
    GradientKey{LinearRgba(222, 33, 33), 1.0f},
};


static inline float DensityKernel(float dst)
{
	return kernel::Poly6Kernel(dst, SMOOTHING_RADIUS);
}

static inline float DensityKernelScale()
{
	return kernel::Poly6KernelScale(SMOOTHING_RADIUS);
}

static inline float DensityKernelDerivative(float dst)
{
	return kernel::SpikyKernelDerivative(dst, SMOOTHING_RADIUS);
}

static inline float DensityKernelDerivativeScale()
{
	return kernel::SpikyKernelDerivativeScale(SMOOTHING_RADIUS);
}

static inline float ViscosityKernel(float dst)
{
	// return kernel::Poly6Kernel(dst, SMOOTHING_RADIUS);
	return kernel::LaplacianKernel(dst, SMOOTHING_RADIUS);
}

static inline float ViscosityKernelScale()
{
	// return kernel::Poly6KernelScale(SMOOTHING_RADIUS);
	return kernel::LaplacianKernelScale(SMOOTHING_RADIUS);
}

static inline glm::vec2 RandomDir() {
    return glm::linearRand(glm::vec2(-1.0f), glm::vec2(1.0f));
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

    UpdateSpatialLookup(LOOKUP_RADIUS);

    m_pool.submit_loop(0, PARTICLE_COUNT, [this](size_t i) {
        m_densities[i] = CalculateDensity(i);
    }).wait();
}

static float ConvertDensityToPressure(float density) {
    // float ratio = density / TARGET_DENSITY;
    // // Tait equation: k * ((ρ/ρ₀)^γ - 1)
    // return PRESSURE_MULTIPLIER * (glm::pow(ratio, 7.0f) - 1.0f);

    float density_error = density - TARGET_DENSITY;
    float pressure = density_error * PRESSURE_MULTIPLIER;
    return pressure;
}

float App::CalculateDensity(size_t index) {
    ZoneScoped;

    float density = DensityKernel(0.0f) * MASS;

    const glm::vec2 point = m_predicted_positions[index];

    m_lookup.ForEachNeighbor(LOOKUP_RADIUS, point, m_predicted_positions, [index, &density](size_t i, glm::vec2 offset, float dst) {
        if (i == index) return;
        density += MASS * DensityKernel(dst);
    });

    return density * DensityKernelScale();
}

glm::vec2 App::CalculatePressureForce(size_t index) {
    ZoneScoped;

    glm::vec2 pressure_force = glm::vec2(0.0f);

    const glm::vec2 point = m_predicted_positions[index];

    float density_i = glm::max(m_densities[index], 0.0001f);
    float pressure_i = ConvertDensityToPressure(density_i);

    m_lookup.ForEachNeighbor(LOOKUP_RADIUS, point, m_predicted_positions, [this, index, density_i, pressure_i, &pressure_force](size_t j, glm::vec2 offset, float dst) {
        if (j == index) return;

        const glm::vec2 dir = dst > glm::epsilon<float>() ? (offset / dst) : RandomDir();

        const glm::vec2 slope = DensityKernelDerivative(dst) * dir;

        const float density_j = glm::max(m_densities[j], 0.0001f);
        const float pressure_j = ConvertDensityToPressure(density_j);

        const float shared_pressure = (pressure_i + pressure_j) * 0.5f;

        // pressure_force += MASS * (pressure_i / (density_i * density_i) + pressure_j / (density_j * density_j)) * slope;
        pressure_force += MASS * shared_pressure * slope / density_j;
    });

    return pressure_force * DensityKernelDerivativeScale() / density_i;
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

    const glm::vec2 point = m_predicted_positions[index];

    m_lookup.ForEachNeighbor(LOOKUP_RADIUS, point, m_predicted_positions, [this, index, &viscosity_force](size_t j, glm::vec2 offset, float dst) {
        if (index == j) return;
        viscosity_force += MASS * (m_velocities[j] - m_velocities[index]) * ViscosityKernel(dst) / m_densities[j];
    });

    return viscosity_force * ViscosityKernelScale() * VISCOSITY_STRENGTH;
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
    ZoneScoped;

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
    ZoneScoped;

    const float dt = Time::FixedDeltaSeconds();

    UpdateSpatialLookup(LOOKUP_RADIUS);

    for (size_t i = 0; i < PARTICLE_COUNT; ++i) {
        m_forces[i] = glm::zero<glm::vec2>();
    }

    {
        ZoneScopedN("CalculateDensity Block");
        m_pool.submit_loop(0, PARTICLE_COUNT, [this, dt](size_t i) {
            m_predicted_positions[i] = m_positions[i] + m_velocities[i] * dt;
            m_densities[i] = CalculateDensity(i);
        }).wait();
    }

    {
        ZoneScopedN("CalculateViscosityForce Block");
        m_pool.submit_loop(0, PARTICLE_COUNT, [this, dt](size_t i) {
            m_forces[i] += CalculateViscosityForce(i);
        }).wait();
    }

    {
        ZoneScopedN("CalculatePressureForce Block");
        m_pool.submit_loop(0, PARTICLE_COUNT, [this, dt](size_t i) {
            m_forces[i] += CalculatePressureForce(i);
        }).wait();
    }

    {
        ZoneScopedN("CalculateExternalForces Block");
        m_pool.submit_loop(0, PARTICLE_COUNT, [this, dt](size_t i) {
            m_forces[i] += CalculateExternalForces(i);
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
    ZoneScoped;

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

