struct QuinticKernel <: Kernel
    dimension::Integer
    normalization::Float32
end

function QuinticKernel(dimension::Integer)
    if dimension == 1
        normalization = 243.0 / 40.0
    elseif dimension == 2
        normalization = 15309.0 / (478.0 * π)
    elseif dimension == 3
        normalization = 2187.0 / (40.0 * π)
    else
        error("The quintic spline kernel is not defined for n=$dimension")
    end

    QuinticKernel(dimension, normalization)
end

function kernel_evaluate(u::Float32, h_inv::Float32, kernel::QuinticKernel)
    u *= h_inv
    if u < 1.0 / 3.0
        ((1.0 - u)^5 - 6.0 * ((2.0 / 3.0) - u)^5.0 + 15.0 * ((1.0 / 3.0) - u)^5.0) *
                kernel.normalization * h_inv^kernel.dimension 
    elseif u >= 1.0 / 3.0 && u < 2.0 / 3.0
        ((1.0 - u)^5.0 - 6.0 * ((2.0 / 3.0) - u)^5.0) *
                kernel.normalization * h_inv^kernel.dimension
    elseif u >= 2.0 / 3.0 && u < 1.0
        (1.0 - u)^5.0 * kernel.normalization * h_inv^kernel.dimension
    else
        0.0
    end
end

kappa(kernel::QuinticKernel) = 2.1
