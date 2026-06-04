#ifndef APP_HPP
#define APP_HPP

#include "i18n.hpp"
#pragma once

#include <SGE/types/backend.hpp>
#include <SGE/renderer/camera.hpp>
#include <SGE/renderer/batch.hpp>
#include <SGE/renderer/renderer.hpp>
#include <SGE/engine.hpp>

#include "thread_pool.hpp"
#include "spatial_lookup.hpp"

struct AppConfig {
    sge::RenderBackend backend = sge::RenderBackend::OpenGL;
    bool vsync = false;
    bool fullscreen = false;
};

struct SimulationConstants {
    int ParticleCount = 10000;
    float PixelsInMeter = 3.0f;

    float ParticleSize = 7.0f * PixelToMeter(); // m

    float Gravity = 9.8f * 100.0f * PixelToMeter();
    float TargetDensity = 1000.0f;
    float Mass = 12.0f;

    float SmoothingRadius = 1.5f * ParticleSize; // m
    float CollisionDamping = 0.9f;
    float ViscosityStrength = 0.4f;

    float SpeedOfSound = 15.0f; // m/s
    float PressureMultiplier = 1500.0f * PixelsInMeter;

    float InteractionRadius = 200.0f * PixelToMeter();
    float PullInteractionStrength = PressureMultiplier * 4.0f;
    float PushInteractionStrength = PullInteractionStrength * 2.0f;

    float MaxAcceleration = 200.0f;

    float ParticleRadius() const {
        return ParticleSize * 0.5f;
    }

    float PixelToMeter() const {
        return 1.0f / PixelsInMeter;
    }

    float MeterToPixel() const {
        return PixelsInMeter;
    }

    float LookupRadius() const {
        return 1.5f * SmoothingRadius;
    }

    static SimulationConstants GetDefault() {
        SimulationConstants constants;
        constants.ParticleCount = 10000;
        constants.PixelsInMeter = 3.0f;
        constants.ParticleSize = 7.0f * constants.PixelToMeter();
        constants.Gravity = 9.8f * 100.0f * constants.PixelToMeter();
        constants.TargetDensity = 1000.0f;
        constants.Mass = 12.0f;
        constants.SmoothingRadius = 1.5f * constants.ParticleSize;
        constants.CollisionDamping = 0.9f;
        constants.ViscosityStrength = 0.4f;
        constants.SpeedOfSound = 15.0f;
        constants.PressureMultiplier = 1500.0f * constants.PixelsInMeter;
        constants.InteractionRadius = 200.0f * constants.PixelToMeter();
        constants.PullInteractionStrength = constants.PressureMultiplier * 4.0f;
        constants.PushInteractionStrength = constants.PullInteractionStrength * 2.0f;
        return constants;
    }
};

class App : public sge::IEngine {
public:
    App(AppConfig config) : m_config(config) {};

protected:
    bool OnInit() override;
    void OnFixedUpdate() override;
    void OnUpdate() override;
    void OnRender(const std::shared_ptr<sge::GlfwWindow>& window) override;
    void OnInputEvent(const sge::InputEvent &event) override;

#if SGE_DEBUG
    void OnPostRender(const std::shared_ptr<sge::GlfwWindow>& window) override {
        if (sge::Input::Pressed(sge::Key::C)) {
            LLGL::FrameProfile profile;
            GetRenderContext()->GetDebugInfo(&profile);
            SGE_LOG_INFO("Draw count: {}", profile.commandBufferRecord.drawCommands);
        }
    }
#endif

    void OnWindowResized(const std::shared_ptr<sge::GlfwWindow>& window, int width, int height) override {
        m_camera.set_viewport(sge::Size(width, height));
        m_camera.update();
    }

    void OnWindowDestroy(sge::GlfwWindow &window) override {
        if (window.GetID() == m_primary_window_id) {
            Stop();
        }
    }

private:
    void InitParticles();

    glm::vec2 CalculatePressureForce(size_t index);
    glm::vec2 CalculateExternalForces(size_t index);
    glm::vec2 CalculateViscosityForce(size_t index);
    float CalculateDensity(size_t index);

    void ResolveObstacleCollisions(glm::vec2& position, glm::vec2& velocity);

    void UpdateSpatialLookup(float radius);

    void CreateSceneTarget(LLGL::Extent2D resolution);

private:
    i18n::Language m_language = i18n::GetDefault();
    sge::Camera m_camera = sge::Camera(sge::CameraConfig { .origin = sge::CameraOrigin::TopLeft });
    sge::Camera m_simulation_camera = sge::Camera(sge::CameraConfig { .origin = sge::CameraOrigin::TopLeft });
    BS::thread_pool<> m_pool;
    SimulationConstants m_constants = SimulationConstants::GetDefault();
    SpatialLookup m_lookup;
    std::vector<glm::vec2> m_positions;
    std::vector<glm::vec2> m_predicted_positions;
    std::vector<glm::vec2> m_velocities;
    std::vector<glm::vec2> m_forces;
    std::vector<float> m_densities;
    std::vector<sge::LinearRgba> m_colors;
    std::vector<sge::Rect> m_obstacles;
    sge::Rect m_selection_rect;
    std::unique_ptr<sge::Batch> m_batch;
    std::unique_ptr<sge::Renderer> m_renderer;

    sge::Handle<LLGL::RenderTarget> m_render_target;
    sge::Texture m_target_texture;
    LLGL::Extent2D m_target_resolution;

    float m_interaction_strength = 0.0f;
    float m_collision_restitution = 0.0f;
    AppConfig m_config;

    bool m_initialized = false;
    bool m_paused = false;
    bool m_gravity = false;
    bool m_selection = false;
    bool m_show_debug_info = false;
    bool m_show_grid = false;
    bool m_collision = true;
#if SGE_IMGUI_ENABLED
    bool m_simulation_view_focused = false;
    bool m_simulation_view_hovered = false;
#endif
    uint32_t m_primary_window_id = 0;
};

#endif
