#ifndef KERNELS_HPP_
#define KERNELS_HPP_

#include <glm/glm.hpp>
#include <glm/ext/scalar_constants.hpp>

namespace kernel {

inline constexpr float PI = glm::pi<float>();

inline float Poly6KernelScale(float h) {
    return 4.0f / (PI * glm::pow(h, 8.0f));
}

inline float Poly6Kernel(float dst, float h) {
    if (dst >= h) return 0.0f;
    float x = (h * h - dst * dst);
    return x * x * x;
}

inline float SpikyKernelDerivativeScale(float h) {
    // return 30.0f / (PI * glm::pow(h, 5.0f));
    return 6.0f / (PI * glm::pow(h, 4.0f));
}

inline float SpikyKernelDerivative(float dst, float h) {
    if (dst >= h) return 0.0f;
    float x = h - dst;
    return x * x;
}

inline float LaplacianKernelScale(float h) {
    return 45.0f / (PI * glm::pow(h, 6.0f));
}

inline float LaplacianKernel(float dst, float h) {
    return h - dst;
}

inline float SpikySmoothingKernel(float dst, float radius) {
    if (dst >= radius) {
        return 0.0f;
    }
    const float volume = PI * glm::pow(radius, 6.0f) / 15.0f;
    const float v = (radius - dst);
    return v * v * v / volume;
}

inline float SbSmoothingKernel(float dst, float radius) {
    if (dst >= radius) {
        return 0.0f;
    }
    const float v = radius - dst;

    return v * v * 6.0f / (PI * glm::pow(radius, 4.0f));
}

inline float sb_smoothing_kernel_derivative(float dst, float radius) {
    if (dst >= radius) {
        return 0.0f;
    }
    return (dst - radius) * 12.0 / (PI * glm::pow(radius, 4.0f));
}

inline float CubicSplineKernel(float dst, float h) {
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

inline glm::vec2 CubicSplineKernelDerivative(glm::vec2 r, float dst, float h) {
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

inline float SpikyPow3ScalingFactor(float radius) {
    return 10.0f / (PI * glm::pow(radius, 5.0f));
}
inline float SpikyPow2ScalingFactor(float radius) {
    return 6.0f / (PI * glm::pow(radius, 4.0f));
}
inline float SpikyPow3DerivativeScalingFactor(float radius) {
    return 30.0f / (glm::pow(radius, 5.0f) * PI);
}

inline float SpikyPow2DerivativeScalingFactor(float radius) {
    return 12.0f / (glm::pow(radius, 4.0f) * PI);
}

inline float SpikyKernelPow3(float dst, float radius) {
    if (dst >= radius) return 0.0f;

    float v = radius - dst;
    return v * v * v * SpikyPow3ScalingFactor(radius);
}

inline float SpikyKernelPow2(float dst, float radius) {
	if (dst >= radius) return 0.0f;

    float v = radius - dst;
    return v * v * SpikyPow2ScalingFactor(radius);
}

inline float DerivativeSpikyPow3(float dst, float radius) {
	if (dst >= radius) return 0.0f;

    float v = radius - dst;
    return -v * v * SpikyPow3DerivativeScalingFactor(radius);
}

inline float DerivativeSpikyPow2(float dst, float radius) {
	if (dst >= radius) return 0.0f;

    float v = radius - dst;
    return -v * SpikyPow2DerivativeScalingFactor(radius);
}

}

#endif // KERNELS_HPP_