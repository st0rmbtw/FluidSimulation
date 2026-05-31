#include "i18n.hpp"
#include "json.hpp"

#include <SGE/log.hpp>

#include <fstream>

using json = nlohmann::json;

i18n::Language i18n::LoadLanguage(const std::filesystem::path &path) {
    i18n::Language language = {};

    std::ifstream f(path);
    if (!f.good())
        return language;

    json data = json::parse(f, nullptr, false, true, true);
    if (data.is_discarded()) {
        SGE_LOG_ERROR("Failed to parse language '{}'", path.string());
        return language;
    }

    language.CollisionDamping = data.value("CollisionDamping", MISSING_VALUE_STRING);
    language.Constants = data.value("Constants", MISSING_VALUE_STRING);
    language.Controls = data.value("Controls", MISSING_VALUE_STRING);
    language.FrameTime = data.value("FrameTime", MISSING_VALUE_STRING);
    language.Gravity = data.value("Gravity", MISSING_VALUE_STRING);
    language.GravityStrength = data.value("GravityStrength", MISSING_VALUE_STRING);
    language.Interaction = data.value("Interaction", MISSING_VALUE_STRING);
    language.InteractionPullStrength = data.value("InteractionPullStrength", MISSING_VALUE_STRING);
    language.InteractionPushStrength = data.value("InteractionPushStrength", MISSING_VALUE_STRING);
    language.InteractionRadius = data.value("InteractionRadius", MISSING_VALUE_STRING);
    language.Language = data.value("Language", MISSING_VALUE_STRING);
    language.Mass = data.value("Mass", MISSING_VALUE_STRING);
    language.MsPerFrame = data.value("MsPerFrame", MISSING_VALUE_STRING);
    language.ParticleCount = data.value("ParticleCount", MISSING_VALUE_STRING);
    language.ParticleSize = data.value("ParticleSize", MISSING_VALUE_STRING);
    language.Paused = data.value("Paused", MISSING_VALUE_STRING);
    language.PixelsInMeter = data.value("PixelsInMeter", MISSING_VALUE_STRING);
    language.PressureMultiplier = data.value("PressureMultiplier", MISSING_VALUE_STRING);
    language.Reset = data.value("Reset", MISSING_VALUE_STRING);
    language.ResetToDefault = data.value("ResetToDefault", MISSING_VALUE_STRING);
    language.SmoothingRadius = data.value("SmoothingRadius", MISSING_VALUE_STRING);
    language.Statistics = data.value("Statistics", MISSING_VALUE_STRING);
    language.TargetDensity = data.value("TargetDensity", MISSING_VALUE_STRING);
    language.ViscosityStrength = data.value("ViscosityStrength", MISSING_VALUE_STRING);

    return language;
}

const i18n::Language& i18n::GetDefault() {
    static i18n::Language language = {
        .CollisionDamping = "Collision Damping",
        .Constants = "Constants",
        .Controls = "Controls",
        .FrameTime = "Frame time",
        .Gravity = "Enable Gravity",
        .GravityStrength = "Gravity",
        .Interaction = "Interaction",
        .InteractionPullStrength = "Pull Interaction Strength",
        .InteractionPushStrength = "Push Interaction Strength",
        .InteractionRadius = "Interaction Radius",
        .Language = "Language",
        .Mass = "Mass",
        .MsPerFrame = "ms/frame",
        .ParticleCount = "Particle Count",
        .ParticleSize = "Particle Size",
        .Paused = "Paused",
        .PixelsInMeter = "Pixels In Meter",
        .PressureMultiplier = "Pressure Multiplier",
        .Reset = "Reset",
        .ResetToDefault = "Reset To Default",
        .SmoothingRadius = "Smoothing Radius",
        .Statistics = "Statistics",
        .TargetDensity = "Target Density",
        .ViscosityStrength = "Viscosity Strength",
    };
    return language;
}