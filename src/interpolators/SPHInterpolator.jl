"""
    interpolate(distance::Float32, h_inv::Float32, container::SPH)

    Evaluates the SPH kernel for this particle.

...
# Arguments
- `distance::Float32`: The distance for the kernel.
- `h_inv::Float32`: The inverse smoothing length for this particle.
- `container::SPH`: The prepared SPH container.
...
"""
function interpolate(distance::Float32, h_inv::Float32, container::SPH)
    @fastmath Kernels.kernel_evaluate(distance, h_inv, container.kernel)
end