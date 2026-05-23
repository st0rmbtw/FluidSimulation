#ifndef APP_HPP
#define APP_HPP

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

class App : public sge::IEngine {
public:
    App(AppConfig config) : m_config(config) {};

protected:
    bool OnInit() override;
    void OnFixedUpdate() override;
    void OnUpdate() override;
    void OnRender(const std::shared_ptr<sge::GlfwWindow>& window) override;

#if SGE_DEBUG
    void OnPostRender(const std::shared_ptr<sge::GlfwWindow>& window) override {
        if (sge::Input::Pressed(sge::Key::C)) {
            LLGL::FrameProfile profile = GetRenderContext()->GetDebugInfo();
            SGE_LOG_INFO("Draw count");
        }
    }
#endif

    void OnWindowResized(const std::shared_ptr<sge::GlfwWindow>& window, int width, int height) override {
        m_camera.set_viewport(sge::Size(width, height));
        m_camera.update();
        InitParticles();
        OnRender(window);
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

    void ResolveCollisions(glm::vec2& position, glm::vec2& velocity);

    void UpdateSpatialLookup(float radius);

private:
    sge::Camera m_camera = sge::Camera(sge::CameraConfig { .origin = sge::CameraOrigin::TopLeft });
    BS::thread_pool<> m_pool;
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
    float m_interaction_strength = 0.0f;
    AppConfig m_config;
    bool m_paused = false;
    bool m_gravity = false;
    bool m_selection = false;
    bool m_show_debug_info = false;
    uint32_t m_primary_window_id = 0;
};

#endif
