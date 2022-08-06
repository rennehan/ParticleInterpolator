module Kernels

abstract type Kernel end

include("CubicKernel.jl")
include("NearestKernel.jl")

end