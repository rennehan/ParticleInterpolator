module Interpolators
using NearestNeighbors
import ..Kernels

abstract type Container end

mutable struct MovingLeastSquares <: Container
    particle_data::Dict
    grid::Array{Float64, 3}
    grid_weight::Array{Float64, 3}
    kernel::Kernels.Kernel
end

mutable struct Remeshed <: Container
    particle_data::Dict
    grid::Array{Float64, 3}
    grid_weight::Array{Float64, 3}
    kernel::Kernels.Kernel
end

mutable struct SPH <: Container
    particle_data::Dict
    grid::Array{Float64, 3}
    grid_weight::Array{Float64, 3}
    kernel::Kernels.Kernel
end

include("MovingLeastSquaresInterpolator.jl")
include("RemeshedInterpolator.jl")
include("SPHInterpolator.jl")

end