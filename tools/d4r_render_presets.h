#pragma once

#include <cstdlib>

// Platform-specific environment reads stay in the caller. SR and RR deliberately
// use separate settings, even when both are configured for the same feature call.
inline constexpr const char* d4r_render_preset_variable(bool rayReconstruction)
{
    return rayReconstruction ? "D4R_RR_PRESET" : "D4R_DLSS_PRESET";
}

// An absent/empty override leaves every game-selected quality preset unchanged.
// An explicit zero is an override too: it requests NGX's default preset.
struct D4rRenderPresetOverride
{
    bool enabled;
    unsigned int value;

    void apply(unsigned int (&presets)[6]) const
    {
        if (enabled)
            for (unsigned int& preset : presets)
                preset = value;
    }
};

inline D4rRenderPresetOverride d4r_parse_render_preset_override(const char* overrideValue)
{
    if (overrideValue == nullptr || *overrideValue == '\0')
        return {false, 0};
    return {true, static_cast<unsigned int>(std::strtoul(overrideValue, nullptr, 0))};
}

// OptiScaler's own preset override (its menu's "Render Presets Override" and "Apply Changes", or
// RenderPresetOverride=true in OptiScaler.ini) recreates the feature with the presets the player picked. The
// first creation carries the game's presets; a later one that carries different presets was changed in
// OptiScaler, and those win over [DLSS] Model / [RayReconstruction] Model until they change back to the game's.
// Super Sampling and Ray Reconstruction keep a follower each.
inline constexpr unsigned int kD4rPresetLatest = 0x00FFFFFF; // OptiScaler's "Latest": the DLL's choice

struct D4rPresetFollower
{
    bool haveBaseline = false;
    unsigned int baseline[6] = {};

    // true when `presets` were chosen in OptiScaler. `iniOverride`: OptiScaler.ini enables the override, so
    // even the first creation carries the player's choice.
    bool follow(const unsigned int (&presets)[6], bool iniOverride)
    {
        if (!haveBaseline)
        {
            for (int index = 0; index < 6; ++index)
                baseline[index] = presets[index];
            haveBaseline = true;
            return iniOverride;
        }
        if (iniOverride)
            return true;
        for (int index = 0; index < 6; ++index)
            if (presets[index] != baseline[index])
                return true;
        return false;
    }
};

// A slot the follower may honour: Super Sampling E, F (CNN), J, K (DLSS 4), L, M (DLSS 4.5); Ray Reconstruction
// D, E (DLSS 4), F (DLSS 4.5); or Default/Latest. Everything else reads as Default (0): flagged values such as
// Alan Wake 2's 0x80000005/0x80000006 (Streamline's "default", E and F with the high bit set), and presets DLSS 310
// no longer ships (A-D, G-I) or that do not exist.
inline unsigned int d4r_followable_preset(unsigned int preset, bool rayReconstruction)
{
    if (preset == kD4rPresetLatest)
        return preset;
    if (rayReconstruction)
        return preset >= 4 && preset <= 6 ? preset : 0;
    switch (preset)
    {
    case 5: case 6: case 10: case 11: case 12: case 13:
        return preset;
    default:
        return 0;
    }
}

inline void d4r_sanitize_presets(unsigned int (&presets)[6], bool rayReconstruction)
{
    for (unsigned int& preset : presets)
        preset = d4r_followable_preset(preset, rayReconstruction);
}

// Followed presets keep every quality mode that names a model; Default (0) and "Latest" take d4r's override.
inline void d4r_apply_followed_presets(unsigned int (&presets)[6], const D4rRenderPresetOverride& overrideValue)
{
    if (!overrideValue.enabled)
        return;
    for (unsigned int& preset : presets)
        if (preset == 0 || preset == kD4rPresetLatest)
            preset = overrideValue.value;
}

// The preset slot ([DLAA, Quality, Balanced, Performance, UltraPerformance, UltraQuality]) NGX uses for
// NVSDK_NGX_PerfQuality_Value `quality` (MaxPerf 0, Balanced 1, MaxQuality 2, UltraPerformance 3,
// UltraQuality 4, DLAA 5).
inline int d4r_preset_slot(int quality)
{
    static const int slots[6] = {3, 2, 1, 4, 5, 0};
    return quality >= 0 && quality < 6 ? slots[quality] : 1;
}
