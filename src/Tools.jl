module Tools

"""
    centered_cube_bitmask(coords::Matrix{Float32}, radius::Float32)

    Returns a boolean array where True marks coords within the radius.
    The coordinates must be centered on a point of interest, because
    they are assumed here to be differences not the true coordinates.
        
...
# Arguments
- `coords::Matrix{Float32}`: The matrix of coordinates to check.
- `radius::Float32`: The radius to search.
...
"""
function centered_cube_bitmask(coords::Matrix{Float32}, radius::Float32)
    x = ifelse.(abs.(@view(coords[1, :])) .< radius, true, false)
    y = ifelse.(abs.(@view(coords[2, :])) .< radius, true, false)
    z = ifelse.(abs.(@view(coords[3, :])) .< radius, true, false)
    x .&& y .&& z
end

"""
    get_max_idx(coord::Float32, radius::Float32, delta::Float32)

    Get the maximum index in the grid for the extent of this particle.
        
...
# Arguments
- `coord::Float32`: The coordinate.
- `radius::Float32`: The particle smoothing length.
- `delta::Float32`: The cell physical size.
...
"""
@fastmath function get_max_idx(coord::Float32, radius::Float32, 
                               delta::Float32)
    if delta == 0.0
        1
    else
        trunc(Int64, (coord + radius) / delta) + 1
    end
end

"""
    get_min_idx(coord::Float32, radius::Float32, delta::Float32)

    Get the minimum index in the grid for the extent of this particle.
        
...
# Arguments
- `coord::Float32`: The coordinate.
- `radius::Float32`: The particle smoothing length.
- `delta::Float32`: The cell physical size.
...
"""
@fastmath function get_min_idx(coord::Float32, radius::Float32, 
                               delta::Float32)
    if delta == 0.0
        1
    else
        trunc(Int64, (coord - radius) / delta) + 1
    end
end

"""
    real_idx(idx::Int64, grid_size::Int64)

    Get the true grid index taking into account periodic
    boundary conditions. 
        
...
# Arguments
- `idx::Int64`: The measured, unshifted index.
- `grid_size::Int64`: The grid cell length in this direction.
...
"""
function real_idx(idx::Int64, grid_size::Int64)
    if idx > grid_size
        idx - grid_size
    elseif idx < 1
        idx + grid_size
    else
        idx
    end
end

"""
    adjusted_difference(difference::Float32)

    Get the true distance to the cell taking into account periodic
    boundary conditions. 
        
...
# Arguments
- `difference::Float32`: The measured, unshifted distance.
...
"""
@fastmath function adjusted_difference(difference::Float32)
    if difference > 0.5
        difference - 1.0
    elseif difference < -0.5
        difference + 1.0
    else
        difference
    end
end

"""
    interpolate_get_indices_differences(min_idx::Int64, max_idx::Int64, 
                                        indices::Array{Int64}, 
                                        diffs::Array{Float32}, 
                                        coordinate::Float32, 
                                        grid_size::Int64,
                                        ruler::Array{Float32})

    Fill up the pre-allocated arrays that contain the particle indices
    in the grid as well as the distances between the particle and those
    grid cells. 
        
...
# Arguments
- `min_idx::Int64`: The true minimum index in the grid.
- `max_idx::Int64`: The true maximum index in the grid.
- `indices::Array{Int64}`: The array to be filled with the shifted cell locations.
- `diffs::Array{Float32}`: The distances array to be filled.
- `coordinate::Float32`: The coordinate to measure the distance.
- `grid_size::Int64`: The grid size (in cells) in this direction.
- `ruler::Array{Float32}`: The measuring ruler in real coordinates.
...
"""
function get_indices_differences(min_idx::Int64, max_idx::Int64, 
                                 indices::Array{Int64}, 
                                 diffs::Array{Float32}, 
                                 coordinate::Float32, 
                                 grid_size::Int64,
                                 ruler::Array{Float32})
    running_idx::Int64 = 1
    @fastmath @inbounds for di=min_idx:max_idx
        indices[running_idx] = real_idx(di, grid_size)
        diffs[running_idx] = adjusted_difference(
            coordinate - ruler[indices[running_idx]]
        )

        running_idx += 1
    end
end

end
