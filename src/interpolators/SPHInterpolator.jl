"""
    interpolate(container::Container)

    Interpolates particle data onto a grid using the SPH gather method.

...
# Arguments
- `container::SPH`: The prepared SPH container.
...
"""
function interpolate(p::Int64, distance::Float64, container::SPH)
    h_inv = 1.0 / container.particle_data["smoothing_lengths"][p]
    Kernels.kernel_evaluate(distance, h_inv, container.kernel)
end