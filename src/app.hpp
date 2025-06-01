#ifndef APP_HPP
#define APP_HPP

#pragma once

#include <SGE/types/backend.hpp>
#include <SGE/types/config.hpp>

namespace App {
    bool Init(sge::RenderBackend backend, sge::AppConfig config);
    void Run();
    void Destroy();
};

#endif
