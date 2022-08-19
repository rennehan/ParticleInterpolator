module Interpolators
using NearestNeighbors
import ..Kernels

abstract type ParticleCentricContainer end
abstract type GridCentricContainer end

struct MovingLeastSquares <: GridCentricContainer
    N_grid_x::UInt32
    N_grid_y::UInt32
    N_grid_z::UInt32
    grid_dimensions::Array{Float32}
    kernel::Kernels.Kernel
end

struct SPH <: ParticleCentricContainer
    N_grid_x::UInt32
    N_grid_y::UInt32
    N_grid_z::UInt32
    grid_dimensions::Array{Float32}
    kernel::Kernels.Kernel
end

end
