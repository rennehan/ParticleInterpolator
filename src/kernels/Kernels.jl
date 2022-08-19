module Kernels

abstract type Kernel end

include("CubicKernel.jl")
include("QuinticKernel.jl")

end
