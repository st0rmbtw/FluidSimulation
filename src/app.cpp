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
#include <SGE/math/math.hpp>
#include <SGE/utils/string.hpp>

#include <glm/trigonometric.hpp>
#include <glm/gtc/random.hpp>
#include <glm/ext/scalar_constants.hpp>

#include <imgui.h>
#include <imgui_internal.h>

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

static constexpr GradientKey GRADIENT[] = {
    GradientKey{LinearRgba(13, 72, 209), 0.0f},
    GradientKey{LinearRgba(73, 214, 153), 0.45f},
    GradientKey{LinearRgba(235, 215, 66), 0.68f},
    GradientKey{LinearRgba(222, 33, 33), 1.0f},
};


static inline float DensityKernel(float dst, float h)
{
	return kernel::Poly6Kernel(dst, h);
}

static inline float DensityKernelScale(float h)
{
	return kernel::Poly6KernelScale(h);
}

static inline float DensityKernelDerivative(float dst, float h)
{
	return kernel::SpikyKernelDerivative(dst, h);
}

static inline float DensityKernelDerivativeScale(float h)
{
	return kernel::SpikyKernelDerivativeScale(h);
}

static inline float ViscosityKernel(float dst, float h)
{
	return kernel::Poly6Kernel(dst, h);
}

static inline float ViscosityKernelScale(float h)
{
	return kernel::Poly6KernelScale(h);
}

static inline glm::vec2 RandomDir() {
    return glm::linearRand(glm::vec2(-1.0f), glm::vec2(1.0f));
}

void App::InitParticles() {
    for (size_t i = 0; i < m_constants.ParticleCount; ++i) {
        m_velocities[i] = glm::vec2(0.0f);
        m_colors[i] = GRADIENT[0].color;
    }

    const float width = m_simulation_camera.viewport().width;
    const float height = m_simulation_camera.viewport().height;

    float x = (width / 2.0f) * m_constants.PixelToMeter();
    float y = (height / 2.0f) * m_constants.PixelToMeter();

    // Spawn in spiral order
    int s = 1;
    int direction = 0;
    int index = 0;
    while (index < m_constants.ParticleCount) {
        for (int i = 0; i < glm::min(m_constants.ParticleCount, index+s)-index; ++i) {
            glm::vec2 pos = glm::vec2(x, y);
            m_positions[index] = pos;
            m_predicted_positions[index] = pos;
            index += 1;

            switch (direction) {
                case 0: x += m_constants.ParticleSize;
                break;
                case 1: y += m_constants.ParticleSize;
                break;
                case 2: x -= m_constants.ParticleSize;
                break;
                case 3: y -= m_constants.ParticleSize;
                break;
            }
        }

        direction = (direction + 1) % 4;
        if (direction % 2 == 0)
            s++;
    }

    UpdateSpatialLookup(m_constants.LookupRadius);

    m_pool.submit_loop(0, m_constants.ParticleCount, [this](size_t i) {
        m_densities[i] = CalculateDensity(i);
    }).wait();

    m_initialized = true;
}

static float ConvertDensityToPressure(float density, float target_density, float k) {
    // float ratio = density / target_density;
    // // Tait equation: k * ((ρ/ρ₀)^γ - 1)
    // return PRESSURE_MULTIPLIER * (glm::pow(ratio, 7.0f) - 1.0f);

    float density_error = density - target_density;
    float pressure = density_error * k;
    return pressure;
}

float App::CalculateDensity(size_t index) {
    ZoneScoped;

    float density = DensityKernel(0.0f, m_constants.SmoothingRadius) * m_constants.Mass;

    const glm::vec2 point = m_predicted_positions[index];

    m_lookup.ForEachNeighbor(m_constants.LookupRadius, point, m_predicted_positions, [this, index, &density](size_t i, glm::vec2 offset, float dst) {
        if (i == index) return;
        density += m_constants.Mass * DensityKernel(dst, m_constants.SmoothingRadius);
    });

    return density * DensityKernelScale(m_constants.SmoothingRadius);
}

glm::vec2 App::CalculatePressureForce(size_t index) {
    ZoneScoped;

    glm::vec2 pressure_force = glm::vec2(0.0f);

    const glm::vec2 point = m_predicted_positions[index];

    float density_i = glm::max(m_densities[index], 0.0001f);
    float pressure_i = ConvertDensityToPressure(density_i, m_constants.TargetDensity, m_constants.PressureMultiplier);

    m_lookup.ForEachNeighbor(m_constants.LookupRadius, point, m_predicted_positions, [this, index, density_i, pressure_i, &pressure_force](size_t j, glm::vec2 offset, float dst) {
        if (j == index) return;

        const glm::vec2 dir = dst > glm::epsilon<float>() ? (offset / dst) : RandomDir();

        const glm::vec2 slope = DensityKernelDerivative(dst, m_constants.SmoothingRadius) * dir;

        const float density_j = glm::max(m_densities[j], 0.0001f);
        const float pressure_j = ConvertDensityToPressure(density_j, m_constants.TargetDensity, m_constants.PressureMultiplier);

        const float shared_pressure = (pressure_i + pressure_j) * 0.5f;

        // pressure_force += m_constants.Mass * (pressure_i / (density_i * density_i) + pressure_j / (density_j * density_j)) * slope;
        pressure_force += m_constants.Mass * shared_pressure * slope / density_j;
    });

    return pressure_force * DensityKernelDerivativeScale(m_constants.SmoothingRadius) / density_i;
}

glm::vec2 App::CalculateExternalForces(size_t index) {
    glm::vec2 gravity_force = glm::vec2(0.0f);

    if (m_gravity) {
        gravity_force.y = m_constants.Gravity;
    }

    if (!approx_equals(m_interaction_strength, 0.0f)) {
        const glm::vec2 input_pos = Input::CursorPosition() * m_constants.PixelToMeter();
        const glm::vec2 offset = input_pos - m_positions[index];
        const float sqr_dst = glm::dot(offset, offset);

        if (sqr_dst < m_constants.InteractionRadius * m_constants.InteractionRadius) {
            const float dst = glm::sqrt(sqr_dst);
            const glm::vec2 dir = dst > glm::epsilon<float>() ? offset / dst : RandomDir();
            float scale = 1.0f - dst / m_constants.InteractionRadius;
            return (dir * m_interaction_strength - m_velocities[index]) * scale;
        }
    }

    return gravity_force * m_constants.Mass;
}

glm::vec2 App::CalculateViscosityForce(size_t index) {
    glm::vec2 viscosity_force = glm::vec2(0.0f);

    const glm::vec2 point = m_predicted_positions[index];

    m_lookup.ForEachNeighbor(m_constants.LookupRadius, point, m_predicted_positions, [this, index, &viscosity_force](size_t j, glm::vec2 offset, float dst) {
        if (index == j) return;
        viscosity_force += m_constants.Mass * (m_velocities[j] - m_velocities[index]) * ViscosityKernel(dst, m_constants.SmoothingRadius) / m_densities[j];
    });

    return viscosity_force * ViscosityKernelScale(m_constants.SmoothingRadius) * m_constants.ViscosityStrength;
}

void App::ResolveObstacleCollisions(glm::vec2& position, glm::vec2& velocity) {
    const float width = (m_simulation_camera.viewport().width) * m_constants.PixelToMeter();
    const float height = (m_simulation_camera.viewport().height) * m_constants.PixelToMeter();

    const float half_size = m_constants.ParticleSize / 2.0f;

    if (position.x + half_size > width) {
        position.x = width - half_size;
        velocity.x *= -1.0f * m_constants.CollisionDamping;
    }

    if (position.y + half_size > height) {
        position.y = height - half_size;
        velocity.y *= -1.0f * m_constants.CollisionDamping;
    }

    if (position.x - half_size < 0) {
        position.x = half_size;
        velocity.x *= -1.0f * m_constants.CollisionDamping;
    }

    if (position.y - half_size < 0) {
        position.y = half_size;
        velocity.y *= -1.0f * m_constants.CollisionDamping;
    }

    for (const Rect& b : m_obstacles) {
        glm::vec2 closest = glm::clamp(position, b.min, b.max);
        glm::vec2 difference = position - closest;

        float distance_sqr = glm::dot(difference, difference);

        if (distance_sqr < m_constants.ParticleRadius() * m_constants.ParticleRadius()) {
            glm::vec2 pen = m_constants.ParticleRadius() - glm::abs(difference);
            
            float penetration;
            glm::vec2 normal;

            if (pen.x < pen.y) {
                penetration = pen.x;
                normal = glm::vec2(glm::sign(difference.x), 0.0f);
            } else {
                penetration = pen.y;
                normal = glm::vec2(0.0f, glm::sign(difference.y));
            }

            position += normal * penetration;

            float vn = glm::dot(velocity, normal);
            velocity -= m_constants.CollisionDamping * vn * normal;
        }   
    }
}

void App::UpdateSpatialLookup(float radius) {
    ZoneScoped;

    for (size_t i = 0; i < m_constants.ParticleCount; ++i) {
        const glm::ivec2 coord = SpatialLookup::PositionToCellCoord(m_positions[i], radius);
        size_t cell_key = m_lookup.GetKeyFromHash(SpatialLookup::HashCell(coord));
        m_lookup.SetCell(i, SpatialLookup::Cell{i, cell_key});
        m_lookup.SetStartIndex(i, SIZE_MAX);
    };

    m_lookup.Sort();

    for (size_t i = 0; i < m_constants.ParticleCount; ++i) {
        size_t key = m_lookup.GetCell(i).key;
        size_t prev_key = i == 0 ? SIZE_MAX : m_lookup.GetCell(i - 1).key;
        if (key != prev_key) {
            m_lookup.SetStartIndex(key, i);
        }
    };
}

void App::OnFixedUpdate() {
    ZoneScoped;

    if (m_paused)
        return;

    if (m_constants.ParticleCount <= 0)
        return;

    if (!m_initialized)
        return;

    const float dt = Time::FixedDeltaSeconds();

    UpdateSpatialLookup(m_constants.LookupRadius);

    for (size_t i = 0; i < m_constants.ParticleCount; ++i) {
        m_forces[i] = glm::zero<glm::vec2>();
    }

    {
        ZoneScopedN("CalculateDensity Block");
        m_pool.submit_loop(0, m_constants.ParticleCount, [this, dt](size_t i) {
            m_predicted_positions[i] = m_positions[i] + m_velocities[i] * dt;
            m_densities[i] = CalculateDensity(i);
        }).wait();
    }

    {
        ZoneScopedN("CalculateViscosityForce Block");
        m_pool.submit_loop(0, m_constants.ParticleCount, [this, dt](size_t i) {
            m_forces[i] += CalculateViscosityForce(i);
        }).wait();
    }

    {
        ZoneScopedN("CalculatePressureForce Block");
        m_pool.submit_loop(0, m_constants.ParticleCount, [this, dt](size_t i) {
            m_forces[i] += CalculatePressureForce(i);
        }).wait();
    }

    {
        ZoneScopedN("CalculateExternalForces Block");
        m_pool.submit_loop(0, m_constants.ParticleCount, [this, dt](size_t i) {
            m_forces[i] += CalculateExternalForces(i);
        }).wait();
    }

    for (size_t i = 0; i < m_constants.ParticleCount; ++i) {
        glm::vec2 acceleration = m_forces[i] / m_constants.Mass;
        m_velocities[i] += acceleration * dt;
    }

    {
        ZoneScopedN("Update position Block");
        m_pool.submit_loop(0, m_constants.ParticleCount, [this, dt](size_t i) {
            glm::vec2 next_position = m_positions[i] + m_velocities[i] * dt;
            ResolveObstacleCollisions(next_position, m_velocities[i]);
            m_positions[i] = next_position;
        }).wait();
    }

    float velocity_max = 10.0f;
    for (size_t i = 0; i < m_constants.ParticleCount; ++i) {
        const glm::vec2 v = m_velocities[i];
        velocity_max = std::max(velocity_max, glm::dot(v, v));
    }

    {
        ZoneScopedN("GradientEvaluation Block");
        m_pool.submit_loop(0, m_constants.ParticleCount, [this, dt, velocity_max](size_t i) {
            const glm::vec2 v = m_velocities[i];
            float t = glm::dot(v, v) / velocity_max;
            m_colors[i] = GradientEvaluate(GRADIENT, t);
        }).wait();
    }
}

void App::OnUpdate() {
    if (Input::JustPressed(Key::P)) {
        m_paused = !m_paused;
    }
    
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
            m_obstacles.push_back(m_selection_rect * m_constants.PixelToMeter());
        }
        m_selection = false;
    } else {
        m_selection = false;
    }

    m_interaction_strength = 0.0f;

    if (!m_selection) {
        if (Input::Pressed(MouseButton::Left)) {
            m_interaction_strength = -m_constants.PushInteractionStrength;
        } else if (Input::Pressed(MouseButton::Right)) {
            m_interaction_strength = m_constants.PullInteractionStrength;
        }
    }
}

void App::OnInputEvent(const InputEvent& event) {
#if SGE_IMGUI_ENABLED
    if (!m_simulation_view_focused)
        return;
    if (!m_simulation_view_hovered && event.Type == sge::InputEventType::MouseButton)
        return;
#endif
    Input::ProcessEvent(event);
}

void App::CreateSceneTarget(LLGL::Extent2D resolution) {
    sge::TextureConfig textureConfig;
    textureConfig.textureType = LLGL::TextureType::Texture2D;
    textureConfig.extent.width = resolution.width;
    textureConfig.extent.height = resolution.height;
    textureConfig.sampler = GetRenderContext()->GetLinearSampler();

    m_target_texture = GetRenderContext()->CreateTexture(textureConfig);

    sge::RenderTargetConfig targetConfig;
    targetConfig.resolution.width = resolution.width;
    targetConfig.resolution.height = resolution.height;
    targetConfig.format = LLGL::Format::RGBA8UNorm;

    targetConfig.colorAttachments[0] = sge::AttachmentConfig(m_target_texture.internal());

    m_render_target = GetRenderContext()->CreateRenderTarget(targetConfig);
    m_target_resolution = resolution;
}

void App::OnRender(const std::shared_ptr<sge::GlfwWindow>& window) {
    ZoneScoped;

    LLGL::RenderTarget& simulationTarget = GetRenderContext()->GetOrCreateRenderTarget(m_render_target, 4);
    if (m_target_resolution != simulationTarget.GetResolution()) {
        CreateSceneTarget(m_target_resolution);
        m_simulation_camera.set_viewport(m_target_resolution);
        m_simulation_camera.update();
    }

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
            m_batch->DrawRect(rect.min * m_constants.MeterToPixel(), {
                .size = rect.size() * m_constants.MeterToPixel(),
                .color = LinearRgba::transparent(),
                .border_thickness = 1.0f,
                .border_color = LinearRgba(73, 214, 153),
                .anchor = Anchor::TopLeft
            });
        }

        if (m_show_debug_info) {
            m_batch->DrawCircle(Input::CursorPosition(), {
                .radius = m_constants.LookupRadius * 2.0f * m_constants.MeterToPixel(),
                .color = LinearRgba::transparent(),
                .border_thickness = 1.0f,
                .border_color = LinearRgba(73, 214, 153),
            });

            const glm::ivec2 center = SpatialLookup::PositionToCellCoord(Input::CursorPosition() * m_constants.PixelToMeter(), m_constants.LookupRadius);

            size_t keys[9] = {};
            for (size_t i = 0; i < 9; ++i) {
                keys[i] = m_lookup.GetKeyFromHash(SpatialLookup::HashCell(center + SpatialLookup::CELL_OFFSETS[i]));
            }

            for (size_t i = 0; i < m_lookup.GetSize(); ++i) {
                const size_t index = m_lookup.GetCell(i).index;

                LinearRgba color = m_colors[index];
                const glm::vec2 pos = m_positions[index] * m_constants.MeterToPixel();

                for (size_t j = 0; j < 9; ++j) {
                    if (m_lookup.GetCell(i).key == keys[j]) {
                        color = LinearRgba::red();
                        break;
                    }
                }

                m_batch->DrawCircle(pos, {
                    .radius = m_constants.ParticleRadius() * m_constants.MeterToPixel(),
                    .color = color,
                    .anchor = Anchor::Center
                });
            }
        } else {
            for (size_t i = 0; i < m_constants.ParticleCount; ++i) {
                const LinearRgba& color = m_colors[i];
                const glm::vec2 pos = m_positions[i] * m_constants.MeterToPixel();

                m_batch->DrawCircle(pos, {
                    .radius = m_constants.ParticleRadius() * m_constants.MeterToPixel(),
                    .color = color,
                    .anchor = Anchor::Center
                });
            }
        }
    }
    m_batch->EndOrderMode();

    m_renderer->BeginPass(simulationTarget, m_simulation_camera);
        m_renderer->Clear(LLGL::ClearValue(0.0f, 0.0f, 0.0f, 1.0f));
        m_renderer->PrepareBatch(*m_batch);
        m_renderer->UploadBatchData();
        m_renderer->RenderBatch(*m_batch);
        m_batch->Reset();
    m_renderer->EndPass();

    m_renderer->BeginPass(window, m_camera);
        m_renderer->Clear(LLGL::ClearValue(0.0f, 0.0f, 0.0f, 1.0f));

        #if SGE_IMGUI_ENABLED
        GetRenderContext()->BeginImGuiFrame(*window);
        {
            ImGui::NewFrame();
            {
                ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
                               ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | 
                               ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | 
                               ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs;

                ImGuiViewport* viewport = ImGui::GetMainViewport();
                ImGui::SetNextWindowPos(viewport->WorkPos);
                ImGui::SetNextWindowSize(viewport->WorkSize);
                ImGui::SetNextWindowViewport(viewport->ID);

                ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
                ImGui::Begin("RootDockSpaceWindow", nullptr, window_flags);
                ImGui::PopStyleVar(3);
                {
                    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
                    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);

                    static bool layout_initialized = false;

                    if (!layout_initialized) {
                        layout_initialized = true;

                        ImGui::DockBuilderRemoveNode(dockspace_id);
                        ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
                        ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

                        ImGuiID dock_id_left;
                        ImGuiID dock_id_right;
                        dock_id_left = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.8f, nullptr, &dock_id_right);

                        ImGui::DockBuilderDockWindow("Simulation", dock_id_left);
                        ImGui::DockBuilderDockWindow("Settings", dock_id_right);

                        ImGuiDockNode* left_node = ImGui::DockBuilderGetNode(dock_id_left);
                        if (left_node != nullptr) {
                            left_node->LocalFlags |= ImGuiDockNodeFlags_NoTabBar;
                        }

                        ImGuiDockNode* right_node = ImGui::DockBuilderGetNode(dock_id_right);
                        if (right_node != nullptr) {
                            right_node->LocalFlags |= ImGuiDockNodeFlags_NoTabBar;
                        }

                        ImGui::DockBuilderFinish(dockspace_id);
                    }

                    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
                    ImGui::Begin("Simulation", nullptr, ImGuiWindowFlags_NoBackground);
                    ImGui::PopStyleVar(3);
                    {
                        m_simulation_view_focused = ImGui::IsWindowFocused();
                        m_simulation_view_hovered = ImGui::IsWindowHovered();

                        ImVec2 avail = ImGui::GetContentRegionAvail();
                        if (avail.x > 0.0f)
                            m_target_resolution.width = static_cast<uint32_t>(avail.x);
                        if (avail.y > 0.0f)
                            m_target_resolution.height = static_cast<uint32_t>(avail.y);

                        ImGui::Image((ImTextureID)&m_target_texture, avail);
                    }
                    ImGui::End();
                    
                    ImGui::Begin("Settings");
                    {
                        if (ImGui::CollapsingHeader("Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
                            if (ImGui::Button("Reset", ImVec2(-FLT_MIN, 0.0f))) {
                                InitParticles();
                            }
                            ImGui::Checkbox("Enable Gravity", &m_gravity);
                            ImGui::Checkbox("Paused", &m_paused);
                        }

                        if (ImGui::CollapsingHeader("Constants", ImGuiTreeNodeFlags_DefaultOpen)) {
                            if (ImGui::DragInt("Particle Count", &m_constants.ParticleCount, 50.0f, 0, INT_MAX)) {
                                if (m_constants.ParticleCount > 0) {
                                    m_positions.resize(m_constants.ParticleCount);
                                    m_predicted_positions.resize(m_constants.ParticleCount);
                                    m_velocities.resize(m_constants.ParticleCount);
                                    m_densities.resize(m_constants.ParticleCount);
                                    m_colors.resize(m_constants.ParticleCount);
                                    m_forces.resize(m_constants.ParticleCount);
                                    m_lookup.Resize(m_constants.ParticleCount);
                                }
                                InitParticles();
                            }
                            ImGui::DragFloat("Pixels In Meter", &m_constants.PixelsInMeter, 0.01f, 0.0f, FLT_MAX);
                            ImGui::DragFloat("Particle Size", &m_constants.ParticleSize, 0.001f, 0.0f, FLT_MAX);
                            ImGui::DragFloat("Gravity", &m_constants.Gravity, 5.0f);
                            ImGui::DragFloat("Mass", &m_constants.Mass, 1.0f, 0.0f, FLT_MAX);
                            ImGui::DragFloat("Target Density", &m_constants.TargetDensity, 0.05f, 0.0f, FLT_MAX);
                            ImGui::DragFloat("Viscosity Strength", &m_constants.ViscosityStrength, 0.01f, 0.0f, FLT_MAX);
                            ImGui::DragFloat("Smoothing Radius", &m_constants.SmoothingRadius, 0.01f, 0.0f, FLT_MAX);
                            ImGui::DragFloat("Pressure Multiplier", &m_constants.PressureMultiplier, 10.0f, 0.0f, FLT_MAX);
                            ImGui::DragFloat("Collision Damping", &m_constants.CollisionDamping, 0.01f, 0.0f, FLT_MAX);

                            ImGui::SeparatorText("Interaction");

                            ImGui::DragFloat("Interaction Radius", &m_constants.InteractionRadius, 0.1f);
                            ImGui::DragFloat("Pull Strength", &m_constants.PullInteractionStrength, 10.0f);
                            ImGui::DragFloat("Push Strength", &m_constants.PushInteractionStrength, 10.0f);

                            if (ImGui::Button("Reset To Default", ImVec2(-FLT_MIN, 0.0f))) {
                                m_constants = SimulationConstants::GetDefault();
                                InitParticles();
                            }
                        }

                        if (ImGui::CollapsingHeader("Statistics")) {
                            ImGui::Text("Frame Time: %f (%.0f FPS)", sge::Time::DeltaSeconds(), 1.0f / sge::Time::DeltaSeconds());
                        }
                    }
                    ImGui::End();
                }
                ImGui::End();
            }
            ImGui::Render();
        }
        GetRenderContext()->EndImGuiFrame();
        #endif
    m_renderer->EndPass();

    m_renderer->End();
}

bool App::OnInit() {
    ImGuiConfig imguiConfig;
    imguiConfig.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    imguiConfig.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    if (!InitRenderContext(m_config.backend, imguiConfig))
        return false;

    SetAutoPresent(true);

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

    m_positions.resize(m_constants.ParticleCount);
    m_predicted_positions.resize(m_constants.ParticleCount);
    m_velocities.resize(m_constants.ParticleCount);
    m_densities.resize(m_constants.ParticleCount);
    m_colors.resize(m_constants.ParticleCount);
    m_forces.resize(m_constants.ParticleCount);

    m_lookup.Resize(m_constants.ParticleCount);

    CreateSceneTarget(LLGL::Extent2D(window_settings.width, window_settings.height));

    m_simulation_camera.set_viewport(m_target_resolution);
    m_simulation_camera.update();

    window->ShowWindow();

    return true;
}

