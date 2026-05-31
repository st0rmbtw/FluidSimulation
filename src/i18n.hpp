#ifndef I18N_HPP_
#define I18N_HPP_

#include <filesystem>

namespace i18n {

constexpr const char* MISSING_VALUE_STRING = "???";

struct Language {
    std::string CollisionDamping = MISSING_VALUE_STRING;
    std::string Constants = MISSING_VALUE_STRING;
    std::string Controls = MISSING_VALUE_STRING;
    std::string FrameTime = MISSING_VALUE_STRING;
    std::string Gravity = MISSING_VALUE_STRING;
    std::string GravityStrength = MISSING_VALUE_STRING;
    std::string Interaction = MISSING_VALUE_STRING;
    std::string InteractionPullStrength = MISSING_VALUE_STRING;
    std::string InteractionPushStrength = MISSING_VALUE_STRING;
    std::string InteractionRadius = MISSING_VALUE_STRING;
    std::string Language = MISSING_VALUE_STRING;
    std::string Mass = MISSING_VALUE_STRING;
    std::string MsPerFrame = MISSING_VALUE_STRING;
    std::string ParticleCount = MISSING_VALUE_STRING;
    std::string ParticleSize = MISSING_VALUE_STRING;
    std::string Paused = MISSING_VALUE_STRING;
    std::string PixelsInMeter = MISSING_VALUE_STRING;
    std::string PressureMultiplier = MISSING_VALUE_STRING;
    std::string Reset = MISSING_VALUE_STRING;
    std::string ResetToDefault = MISSING_VALUE_STRING;
    std::string SmoothingRadius = MISSING_VALUE_STRING;
    std::string Statistics = MISSING_VALUE_STRING;
    std::string TargetDensity = MISSING_VALUE_STRING;
    std::string ViscosityStrength = MISSING_VALUE_STRING;
};

const Language& GetDefault();
Language LoadLanguage(const std::filesystem::path& path);

}

#endif