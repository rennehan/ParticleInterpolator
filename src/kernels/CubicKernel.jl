struct CubicKernel <: Kernel
    dimension::Integer
    normalization::Float32
end

function CubicKernel(dimension::Integer)
    if dimension == 1
        normalization = 4.0 / 3.0
    elseif dimension == 2
        normalization = 40.0 / (7 * π)
    elseif dimension == 3
        normalization = 8.0 / π
    else
        error("The cubic spline kernel is not defined for n=$dimension")
    end

    CubicKernel(dimension, normalization)
end

function kernel_evaluate(u::Float32, h_inv::Float32, kernel::CubicKernel)
    weight_factor = kernel.normalization * h_inv^kernel.dimension
    u *= h_inv

    if u < 0.5
        segment = (1.0 + 6.0 * (u - 1.0) * u^2)
    elseif u >= 0.5 && u < 1.0
        segment = 2.0 * (1.0 - u)^3.0
    else
        segment = 0.0
    end

    segment * weight_factor
end

kappa(kernel::CubicKernel) = 2.0
