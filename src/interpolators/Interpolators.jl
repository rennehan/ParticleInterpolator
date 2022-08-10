module Interpolators
using NearestNeighbors
import ..Kernels

abstract type Container end

struct MovingLeastSquares <: Container
    N_grid_x::UInt32
    N_grid_y::UInt32
    N_grid_z::UInt32
    grid_dimensions::Array{Float32}
    kernel::Kernels.Kernel
end

struct Remeshed <: Container
    N_grid_x::UInt32
    N_grid_y::UInt32
    N_grid_z::UInt32
    grid_dimensions::Array{Float32}
    kernel::Kernels.Kernel
end

struct SPH <: Container
    N_grid_x::UInt32
    N_grid_y::UInt32
    N_grid_z::UInt32
    grid_dimensions::Array{Float32}
    kernel::Kernels.Kernel
end

include("MovingLeastSquaresInterpolator.jl")
include("RemeshedInterpolator.jl")
include("SPHInterpolator.jl")

end