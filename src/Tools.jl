module Tools

# TODO: Maybe use floor?
get_max_idx(coord, radius, delta) = trunc(Int32, (coord + radius) / delta) + convert(Int32, 1)
get_min_idx(coord, radius, delta) = trunc(Int32, (coord - radius) / delta) + convert(Int32, 1)

# Assume everything is periodic. If the index is outside of the box,
# then just move it to the other side.
function real_idx(idx::Int32, grid_info::Dict)
    new_idx = idx
    if idx > grid_info["grid_size_1d"]
        new_idx = idx - grid_info["grid_size_1d"]
    elseif idx < 1
        new_idx = idx + grid_info["grid_size_1d"]
    end

    convert(Int32, new_idx)
end

function adjusted_difference(difference::Float64)
    if difference > 0.5
        difference - 1.0
    elseif difference < -0.5
        difference + 1.0
    else
        difference
    end
end

function get_indices_differences(min_idx::Int32, max_idx::Int32, 
                                 indices::Array{Int32}, 
                                 diffs::Array{Float64}, 
                                 coordinate::Float64, grid_info::Dict)
    running_idx::Int32 = 1
    for di=min_idx:max_idx
        indices[running_idx] = real_idx(di, grid_info)
        diffs[running_idx] = adjusted_difference(
            coordinate - grid_info["ruler"][indices[running_idx]]
        )

        running_idx += 1
    end
end

end