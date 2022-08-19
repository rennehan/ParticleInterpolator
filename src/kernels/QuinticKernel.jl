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
    weight_factor = kernel.normalization * h_inv^kernel.dimension
    u *= h_inv

    one_minus_u = 1.0 - u
    one_third_minus_u = (1.0 / 3.0) - u
    two_thirds_minus_u = (2.0 / 3.0) - u

    if u < 1.0 / 3.0
        segment = one_minus_u^5 - 6.0 * two_thirds_minus_u^5 + 15.0 * one_third_minus_u^5
    elseif u >= 1.0 / 3.0 && u < 2.0 / 3.0
        segment = one_minus_u^5 - 6.0 * two_thirds_minus_u^5
    elseif u >= 2.0 / 3.0 && u < 1.0
        segment = one_minus_u^5
    else
        segment = 0.0
    end

    segment * weight_factor
end

kappa(kernel::QuinticKernel) = 2.0