module ParticleInterpolator

using NearestNeighbors, Distances, ProgressMeter

include("Tools.jl")
include("kernels/Kernels.jl")
include("Interpolators.jl")

"""
    function get_particle_tree(N_dimensions::Integer, grid_dimensions::Array{Float32}, 
                               coordinates::Array{Float32})

    Construct the BallTree of the input coordinates for fast distance computations. 
    Assumes a PeriodicEuclidean metric. Returns the tree and the metric in a tuple.

...
# Arguments
- `N_dimensions::Integer`: The true number of dimensions (taken from the kernel).
- `grid_dimensions::Array{Float32}`: A vector of the lengths of the grid in each dimension.
- `coordinates::Array{Float32}`: Columns of N_particles, rows of 3 coordinates.
...
"""
function get_particle_tree(N_dimensions::Integer, grid_dimensions::Array{Float32}, 
                           coordinates::Array{Float32})
    metric = PeriodicEuclidean([grid_dimensions[i] for i=1:N_dimensions])
    NearestNeighbors.BallTree(
        @views(coordinates[1:N_dimensions, :]),
        metric
    ), metric
end

"""
    interpolate_particles(container::Nearest, coordinates::Array{Float32},
                          smoothing_lengths::Array{Float32},
                          deposits::Array{Float32})

    Construct the grid to interpolate the particle data, and then interpolate. 
    The type of interpolation depends on the type of ParticleCentricContainer passed to the
    function.
...
# Arguments
- `container::Nearest`: A Nearest struct that contains the particle data, tree, and grid.
- `coordinates::Array{Float32}`: Columns of N particles, rows of 3 coordinates.
- `smoothing_lengths::Array{Float32}`: The smoothing lengths of all of the particles.
- `deposits::Array{Float32}`: The quantities to deposit onto the grid.
...
"""
function interpolate_particles(container::Interpolators.Nearest,
                               coordinates::Array{Float32},
                               smoothing_lengths::Array{Float32},
                               deposits::Array{Float32})
    N_particles = size(coordinates)[2]

    println("N_particles=$N_particles")
    grid_dimensions = container.grid_dimensions
    grid = zeros(
        Float32, 
        (container.N_grid_x, container.N_grid_y, container.N_grid_z)
    )
    grid_sizes = [container.N_grid_x, container.N_grid_y, container.N_grid_z]
    total_cells = container.N_grid_x * container.N_grid_y * container.N_grid_z

    for i=1:3
        println("grid_sizes[$i]=", grid_sizes[i])
        println("grid_dimensions[$i]=", grid_dimensions[i])
    end

    # Using a BallTree leads to over a factor of ~10 times
    # speed-up (maybe even more!)
    particle_tree, _ = get_particle_tree(
        container.kernel.dimension,
        container.grid_dimensions,
        coordinates
    )

    # True coordinates of the grid cell in that direction.
    ruler_x = zeros(Float32, grid_sizes[1])
    ruler_y = zeros(Float32, grid_sizes[2])
    ruler_z = zeros(Float32, grid_sizes[3])

    ruler_x = Float32.(
        ((1:(grid_sizes[1] + 1) .- 1) .+ 0.5) .* (grid_dimensions[1] / (grid_sizes[1] + 1))
    )

    if container.kernel.dimension == 1
        ruler_y = zeros(Float32, 1)
        ruler_z = zeros(Float32, 1)
    elseif container.kernel.dimension == 2
        ruler_y = Float32.(
            ((1:(grid_sizes[2] + 1) .- 1) .+ 0.5) .* (grid_dimensions[2] / (grid_sizes[2] + 1))
        )
        ruler_z = zeros(Float32, 1)
    else
        ruler_y = Float32.(
            ((1:(grid_sizes[2] + 1) .- 1) .+ 0.5) .* (grid_dimensions[2] / (grid_sizes[2] + 1))
        )
        ruler_z = Float32.(
            ((1:(grid_sizes[3] + 1) .- 1) .+ 0.5) .* (grid_dimensions[3] / (grid_sizes[3] + 1))
        )
    end

    # For each grid cell, search within some cutoff
    # radius for particles and then sum them into the bin.
    progress = ProgressMeter.Progress(
        total_cells, 
        0.5, 
        "Interpolating $N_particles particles over $total_cells cells..."
    )
    @inbounds Threads.@threads for I in CartesianIndices(grid) #   ck=1:container.N_grid_z, cj=1:container.N_grid_y, ci=1:container.N_grid_x
        ci, cj, ck = Tuple(I)
        idx, _ = NearestNeighbors.knn(
            particle_tree, 
            [ruler_x[ci], ruler_y[cj], ruler_z[ck]],
            1
        )

        grid[I] = deposits[idx[1]]

        ProgressMeter.next!(progress)
    end

    grid
end

"""
    interpolate_particles(container::MovingLeastSquares, coordinates::Array{Float32},
                          smoothing_lengths::Array{Float32},
                          deposits::Array{Float32})

    Construct the grid to interpolate the particle data, and then interpolate. 
    The type of interpolation depends on the type of ParticleCentricContainer passed to the
    function.
...
# Arguments
- `container::MovingLeastSquares`: A MovingLeastSquares struct that contains the particle data, tree, and grid.
- `coordinates::Array{Float32}`: Columns of N particles, rows of 3 coordinates.
- `smoothing_lengths::Array{Float32}`: The smoothing lengths of all of the particles.
- `deposits::Array{Float32}`: The quantities to deposit onto the grid.
...
"""
function interpolate_particles(container::Interpolators.MovingLeastSquares,
                               coordinates::Array{Float32},
                               smoothing_lengths::Array{Float32},
                               deposits::Array{Float32})
    N_particles = size(coordinates)[2]

    println("N_particles=$N_particles")
    grid_dimensions = container.grid_dimensions
    grid = zeros(
        Float32, 
        (container.N_grid_x, container.N_grid_y, container.N_grid_z)
    )
    grid_sizes = [container.N_grid_x, container.N_grid_y, container.N_grid_z]
    total_cells = container.N_grid_x * container.N_grid_y * container.N_grid_z

    for i=1:3
        println("grid_sizes[$i]=", grid_sizes[i])
        println("grid_dimensions[$i]=", grid_dimensions[i])
    end

    # Using a BallTree leads to over a factor of ~10 times
    # speed-up (maybe even more!)
    particle_tree, _ = get_particle_tree(
        container.kernel.dimension,
        container.grid_dimensions,
        coordinates
    )

    # True coordinates of the grid cell in that direction.
    ruler_x = zeros(Float32, grid_sizes[1])
    ruler_y = zeros(Float32, grid_sizes[2])
    ruler_z = zeros(Float32, grid_sizes[3])

    ruler_x = Float32.(
        ((1:(grid_sizes[1] + 1) .- 1) .+ 0.5) .* (grid_dimensions[1] / (grid_sizes[1] + 1))
    )

    if container.kernel.dimension == 1
        ruler_y = zeros(Float32, 1)
        ruler_z = zeros(Float32, 1)
    elseif container.kernel.dimension == 2
        ruler_y = Float32.(
            ((1:(grid_sizes[2] + 1) .- 1) .+ 0.5) .* (grid_dimensions[2] / (grid_sizes[2] + 1))
        )
        ruler_z = zeros(Float32, 1)
    else
        ruler_y = Float32.(
            ((1:(grid_sizes[2] + 1) .- 1) .+ 0.5) .* (grid_dimensions[2] / (grid_sizes[2] + 1))
        )
        ruler_z = Float32.(
            ((1:(grid_sizes[3] + 1) .- 1) .+ 0.5) .* (grid_dimensions[3] / (grid_sizes[3] + 1))
        )
    end

    # The cutoff radius for the method. We must have
    # cubes as our grid cells, so we can just look at
    # the first dimension to get the grid cell size.
    radius = Float32(Kernels.kappa(container.kernel) / N_particles^(1/3))
    h_inv = Float32(1.0) / radius

    println("radius=$radius")
    println("h_inv=$h_inv")

    # 500 maximum neighbors
    max_neighbors = 500

    # TODO: 10 is for 2nd order, 4 is for 1st order
    Lambda_I_ij = zeros(Float32, (10, max_neighbors))
    T_I_i = zeros(Float32, 10)
    R_IJ_i = zeros(Float32, (10, 10))
    F_J_i = zeros(Float32, 10)
    kernel_weights_ij = zeros(Float32, max_neighbors)
    L_ij = zeros(Float32, max_neighbors)
    local_coords = zeros(Float32, (3, max_neighbors))

    # To make sure that we conserve the deposit, we
    # need to find the sum of kernel weights for all
    # of the particles before we do the interpolation.
    # We will loop over all particles and all the grid
    # cells that they might contribute to, based on the
    # cutoff radius.
    delta_x = grid_dimensions[1] / grid_sizes[1]
    delta_y = grid_dimensions[2] / grid_sizes[2]
    delta_z = grid_dimensions[3] / grid_sizes[3]
    # Add a buffer of 2 cells on each side in case of overlap.
    max_first_cells = trunc(Int64, 2.0 * radius / delta_x + 2.0)
    max_second_cells = trunc(Int64, 2.0 * radius / delta_y + 2.0)
    max_third_cells = trunc(Int64, 2.0 * radius / delta_z + 2.0)

    println("max_first_cells=$max_first_cells")
    println("max_second_cells=$max_second_cells")
    println("max_third_cells=$max_third_cells")

    first_indices = zeros(Int64, max_first_cells)
    second_indices = zeros(Int64, max_second_cells)
    third_indices = zeros(Int64, max_third_cells)
    first_diffs = zeros(Float32, max_first_cells)
    second_diffs = zeros(Float32, max_second_cells)
    third_diffs = zeros(Float32, max_third_cells)
    all_particle_weights::Float32 = 0.0

    # All particles have the same extent here, since we are
    # grid centric and looking over a certain radius.
    # Just look at p=1 and be done.
    min_first_idx = Tools.get_min_idx(
        coordinates[1, 1],
        radius,
        delta_x
    )
    max_first_idx = Tools.get_max_idx(
        coordinates[1, 1],
        radius,
        delta_x
    )

    min_second_idx = Tools.get_min_idx(
        coordinates[2, 1],
        radius,
        delta_y
    )
    max_second_idx = Tools.get_max_idx(
        coordinates[2, 1],
        radius,
        delta_y
    )

    min_third_idx = Tools.get_min_idx(
        coordinates[3, 1],
        radius,
        delta_z
    )
    max_third_idx = Tools.get_max_idx(
        coordinates[3, 1],
        radius,
        delta_z
    )

    # What is the total extent in cells of the particle?
    num_first_cells = max_first_idx - min_first_idx + 1
    num_second_cells = max_second_idx - min_second_idx + 1
    num_third_cells = max_third_idx - min_third_idx + 1

    Tools.get_indices_differences(
        min_first_idx, max_first_idx, 
        first_indices, 
        first_diffs, 
        coordinates[1, 1], 
        grid_sizes[1],
        ruler_x
    )

    Tools.get_indices_differences(
        min_second_idx, max_second_idx, 
        second_indices, 
        second_diffs, 
        coordinates[2, 1], 
        grid_sizes[2],
        ruler_y
    )

    Tools.get_indices_differences(
        min_third_idx, max_third_idx, 
        third_indices, 
        third_diffs, 
        coordinates[3, 1], 
        grid_sizes[3],
        ruler_z
    )

    @inbounds for ck=1:num_third_cells, cj=1:num_second_cells, ci=1:num_first_cells
        all_particle_weights += Kernels.kernel_evaluate(
            sqrt(
                first_diffs[ci]^2 + 
                second_diffs[cj]^2 + 
                third_diffs[ck]^2
            ),
            h_inv,
            container.kernel
        )
    end

    # For each grid cell, search within some cutoff
    # radius for particles and then sum them into the bin.
    progress = ProgressMeter.Progress(
        total_cells, 
        0.5, 
        "Interpolating $N_particles particles over $total_cells cells..."
    )
    @inbounds for ck=1:container.N_grid_z, cj=1:container.N_grid_y, ci=1:container.N_grid_x
        idx = NearestNeighbors.inrange(
            particle_tree, 
            [ruler_x[ci], ruler_y[cj], ruler_z[ck]],
            radius,
            true
        )

        local_deposits = @views(deposits[idx])
        N_nearby = length(local_deposits)

        # local_coords hold the centered coordinates of the particle compared
        # to the grid cell.
        local_coords[1, 1:N_nearby] = @views(coordinates[1, idx]) .- ruler_x[ci]
        local_coords[2, 1:N_nearby] = @views(coordinates[2, idx]) .- ruler_y[cj]
        local_coords[3, 1:N_nearby] = @views(coordinates[3, idx]) .- ruler_z[ck]

        @inbounds for p=1:N_nearby, dim=1:3
            if local_coords[dim, p] < -0.5
                local_coords[dim, p] += 1.0
            end
            if local_coords[dim, p] > 0.5
                local_coords[dim, p] -= 1.0
            end
        end


        if N_nearby < 11 || N_nearby > max_neighbors
            error("Cell ($ci, $cj, $ck) has a bad neighbor count (N=$N_nearby)!")
            continue
        end

        @inbounds for p=1:N_nearby
            kernel_weights_ij[p] = Kernels.kernel_evaluate(
                sqrt(
                    local_coords[1, p]^2 +
                    local_coords[2, p]^2 +
                    local_coords[3, p]^2
                ),
                h_inv,
                container.kernel
            )
        end

        @inbounds for p=1:N_nearby
            L_ij[p] = kernel_weights_ij[p] / all_particle_weights
            Lambda_I_ij[1, p] = Float32(1.0)
            Lambda_I_ij[2, p] = local_coords[1, p]
            Lambda_I_ij[3, p] = local_coords[2, p]
            Lambda_I_ij[4, p] = local_coords[3, p]
            Lambda_I_ij[5, p] = 0.5 * local_coords[1, p]^2.0
            Lambda_I_ij[6, p] = 0.5 * local_coords[2, p]^2.0
            Lambda_I_ij[7, p] = 0.5 * local_coords[3, p]^2.0
            Lambda_I_ij[8, p] = 0.5 * local_coords[1, p] * local_coords[2, p]
            Lambda_I_ij[9, p] = 0.5 * local_coords[1, p] * local_coords[3, p]
            Lambda_I_ij[10, p] = 0.5 * local_coords[2, p] * local_coords[3, p]
        end

        @inbounds for dim2=1:10, dim1=1:10
            T_I_i[dim2] = 0.0
            R_IJ_i[dim1, dim2] = 0.0
        end

        @inbounds for p=1:N_nearby, dim=1:10
            T_I_i[dim] += L_ij[p] * local_deposits[p] * Lambda_I_ij[dim, p] 
        end

        @inbounds for p=1:N_nearby, dim2=1:10, dim1=1:10
            # The second Lambda_I_ij is really Theta_J_ij
            R_IJ_i[dim1, dim2] += L_ij[p] * Lambda_I_ij[dim1, p] * Lambda_I_ij[dim2, p]
        end

        try
            # Solve for the values and gradients
            F_J_i = R_IJ_i \ T_I_i
        catch err
            error("Error: $err, N_nearby=$N_nearby")
        end

        # Store the value of the deposit at this location
        grid[ci, cj, ck] = F_J_i[1]

        ProgressMeter.next!(progress)
    end

    grid
end

"""
    accumulate_in_grid(p::Int64, container::Interpolators.ParticleCentricContainer,
                       grid::Array{Float32}, min_first_idx::Int64, 
                       max_first_idx::Int64, N_dimensions::Integer, 
                       coordinate::Array{Float32}, smoothing_length::Float32,
                       inv_smoothing_length::Float32, deposit::Float32,
                       idx_list::Array{Integer}, deltas::Array{Float32},
                       rulers::Dict{Integer, Array{Float32}}, 
                       grid_sizes::Array{UInt32},
                       kernel_weights::Array{Float32}, 
                       first_indices::Array{Int64},
                       second_indices::Array{Int64}, 
                       third_indices::Array{Int64},
                       first_diffs::Array{Float32},
                       second_diffs::Array{Float32},
                       third_diffs::Array{Float32})

    Find the contributions to the grid cells by each particle,
    then accumulate the values in the grid.
...
# Arguments
- `p::Int64`: The particle index.
- `container::ParticleCentricContainer`: A ParticleCentricContainer struct holding the input data.
- `grid::Array{Float32}`: The grid to fill up.
- `min_first_idx::Int64`: The minimum grid index in the x direction of the particle.
- `max_first_idx::Int64`: The maximum grid index in the x direction of the particle.
- `N_dimensions::Integer`: What is the dimension of the simulation?
- `coordinate::Array{Float32}`: The particle coordinates.
- `smoothing_length::Float32`: The particle smoothing length.
- `inv_smoothing_length`::Float32`: The inverse smoothing length.
- `deposit::Float32`: The deposit field for this particle.
- `idx_list::Array{Integer}`: The mapping between index and coordinate.
- `deltas::Array{Float32}`: The grid cell size in real coordinates.
- `rulers::Dict{Integer, Array{Float32}}`: The ruler for each coordinate direction.
- `grid_sizes::Array{UInt32}`: The number of grid cells in each direction.
- `kernel_weights::Array{Float32}`: The kernel weights array to fill up for each particle.
- `first_indices::Array{Int64}`: The indices in the grid for the first direction.
- `second_indices::Array{Int64}`: The indices in the grid for the second direction.
- `third_indices::Array{Int64}`: The indices in the grid for the third direction.
- `first_diffs::Array{Float32}`: Distances in 1st direction between p and cell.
- `second_diffs::Array{Float32}`: Distances in 2nd direction between p and cell.
- `third_diffs::Array{Float32}`: Distances in 3rd direction between p and cell.
...
"""
function accumulate_in_grid(p::Int64, 
                            container::Interpolators.ParticleCentricContainer,
                            grid::Array{Float32},
                            min_first_idx::Int64, 
                            max_first_idx::Int64,
                            N_dimensions::Integer, 
                            coordinate::Array{Float32},
                            smoothing_length::Float32,
                            inv_smoothing_length::Float32,
                            deposit::Float32,
                            idx_list::Array{Integer},
                            deltas::Array{Float32},
                            rulers::Dict{Integer, Array{Float32}},
                            grid_sizes::Array{UInt32},
                            kernel_weights::Array{Float32},
                            first_indices::Array{Int64},
                            second_indices::Array{Int64},
                            third_indices::Array{Int64},
                            first_diffs::Array{Float32},
                            second_diffs::Array{Float32},
                            third_diffs::Array{Float32})

    primary_idx = idx_list[1]
    secondary_idx = idx_list[2]
    tertiary_idx = idx_list[3]

    # All of the interpolation methods rely on some 
    # kernel with compact support. That means that we
    # can safely ignore everything outside of some 
    # extent, h.
    max_second_idx = Tools.get_max_idx(
        coordinate[secondary_idx],
        smoothing_length,
        deltas[secondary_idx]
    )
    min_second_idx = Tools.get_min_idx(
        coordinate[secondary_idx],
        smoothing_length,
        deltas[secondary_idx]
    )

    max_third_idx = Tools.get_max_idx(
        coordinate[tertiary_idx],
        smoothing_length,
        deltas[tertiary_idx]
    )
    min_third_idx = Tools.get_min_idx(
        coordinate[tertiary_idx],
        smoothing_length,
        deltas[tertiary_idx]
    )

    # What is the total extent in cells of the particle?
    num_first_cells = max_first_idx - min_first_idx + 1
    num_second_cells = max_second_idx - min_second_idx + 1
    num_third_cells = max_third_idx - min_third_idx + 1

    Tools.get_indices_differences(
        min_first_idx, max_first_idx, 
        first_indices, 
        first_diffs, 
        coordinate[primary_idx], 
        grid_sizes[primary_idx],
        rulers[primary_idx]
    )

    Tools.get_indices_differences(
        min_second_idx, max_second_idx, 
        second_indices, 
        second_diffs, 
        coordinate[secondary_idx], 
        grid_sizes[secondary_idx],
        rulers[secondary_idx]
    )

    Tools.get_indices_differences(
        min_third_idx, max_third_idx, 
        third_indices, 
        third_diffs, 
        coordinate[tertiary_idx], 
        grid_sizes[tertiary_idx],
        rulers[tertiary_idx]
    )

    # Particle is only going into one cell?
    check_size_cond(h, d, gs) = ifelse(gs > 1, h < d, false)

    smoothing_length_first_cond = check_size_cond(
        smoothing_length, 
        deltas[primary_idx],
        grid_sizes[primary_idx]
    )
    smoothing_length_second_cond = check_size_cond(
        smoothing_length, 
        deltas[secondary_idx],
        grid_sizes[secondary_idx]
    )
    smoothing_length_third_cond = check_size_cond(
        smoothing_length, 
        deltas[tertiary_idx],
        grid_sizes[tertiary_idx]
    )

    cell_size_cond = num_first_cells == 1 && num_second_cells == 1 && num_third_cells == 1

    if cell_size_cond || smoothing_length_first_cond || 
        (N_dimensions == 2 && smoothing_length_second_cond) ||
        (N_dimensions == 3 && smoothing_length_third_cond)
        grid[first_indices[1], second_indices[1], third_indices[1]] += deposit
        return
    end

    # The sum of all of the weights that the
    # particle distributes across the small
    # domain.
    sum_of_kernel_weights::Float32 = 0

    # Loop over the entire chunk
    @fastmath @inbounds for ck=1:num_third_cells, cj=1:num_second_cells, ci=1:num_first_cells
        kernel_weights[ci, cj, ck] = Kernels.kernel_evaluate(
            Float32(sqrt(
                first_diffs[ci]^2 + second_diffs[cj]^2 + third_diffs[ck]^2
            )),
            inv_smoothing_length,
            container.kernel
        )

        sum_of_kernel_weights += kernel_weights[ci, cj, ck]
    end

    if sum_of_kernel_weights == 0.0
        error("sum_of_kernel_weights=0 for 
              p=$p
              min_first_idx=$min_first_idx
              max_first_idx=$max_first_idx
              min_second_idx=$min_second_idx
              max_second_idx=$max_second_idx
              min_third_idx=$min_third_idx
              max_third_idx=$max_third_idx
              num_first_cells=$num_first_cells 
              num_second_cells=$num_second_cells 
              num_third_cells=$num_third_cells")
    end

    # Now that we have the sum of kernel weights we can
    # actually properly deposit the information into the
    # full grid, at the correct location. Remember that
    # we are only looping over the maximum extent of the
    # kernel right now, and need to map back to the full
    # grid.
    # 
    # It is safe to directly add to the grid and grid_weight
    # fields since we have ensured that the parallelization
    # completely avoids any race conditions.
    @fastmath @inbounds for ck=1:num_third_cells, cj=1:num_second_cells, ci=1:num_first_cells
        grid[first_indices[ci], second_indices[cj], third_indices[ck]] += 
                deposit * kernel_weights[ci, cj, ck] / sum_of_kernel_weights
    end
end

"""
    interpolate_particles(container::ParticleCentricContainer, coordinates::Array{Float32},
                          smoothing_lengths::Array{Float32},
                          deposits::Array{Float32})

    Construct the grid to interpolate the particle data, and then interpolate. 
    The type of interpolation depends on the type of ParticleCentricContainer passed to the
    function.

...
# Arguments
- `container::ParticleCentricContainer`: A ParticleCentricContainer struct that contains the particle data, tree, and grid.
- `coordinates::Array{Float32}`: Columns of N particles, rows of 3 coordinates.
- `smoothing_lengths::Array{Float32}`: The smoothing lengths of all of the particles.
- `deposits::Array{Float32}`: The quantities to deposit onto the grid.
...
"""
function interpolate_particles(container::Interpolators.ParticleCentricContainer,
                               coordinates::Array{Float32},
                               smoothing_lengths::Array{Float32},
                               deposits::Array{Float32})

    N_particles = size(coordinates)[2]
    if length(smoothing_lengths) != N_particles
        error("If there are no particle smoothing lengths, please pass in an empty array of length the number of particles.")
    end

    if Threads.nthreads() > N_particles
        error("You have too many threads! Why are you even using that many threads!?")    
    end

    # @TODO N_neighbor should be a parameter
    N_neighbor = 32

    # In some parallelization schemes, we have to redo particles
    # that may exist on the boundaries between the domain
    # decomposition. We will mark these particles with a 1
    # for now, and redo them unparallelized at the end.
    boundary_flag = zeros(Integer, N_particles)

    grid_dimensions = container.grid_dimensions
    grid = zeros(
        Float32, 
        (container.N_grid_x, container.N_grid_y, container.N_grid_z)
    )
    grid_sizes = [container.N_grid_x, container.N_grid_y, container.N_grid_z]
    grid_dimensions = container.grid_dimensions
    N_dimensions = container.kernel.dimension

    println("N_dimensions: $N_dimensions")
    # Create a ruler for each gridded direction. It will
    # store the true coordinates of the grid cell in that
    # direction.
    rulers = Dict{Integer, Array{Float32}}()
    rulers[1] = zeros(Float32, grid_sizes[1])
    rulers[2] = zeros(Float32, grid_sizes[2])
    rulers[3] = zeros(Float32, grid_sizes[3])

    # What is the largest dimension of the grid? We will
    # normalize the internal coordinates to that
    # maximum.
    primary_idx = argmax(grid_dimensions)
    max_coord = maximum(grid_dimensions)

    # Replace the input grid dimensions with the new
    # normalization to the maximum grid length in
    # real (whatever those are) units.
    grid_dimensions ./= max_coord
    coordinates ./= max_coord

    if all(y->y==smoothing_lengths[1], smoothing_lengths)
        @warn "This will be very VERY slow, NearestNeighbors.jl is not threaded!"

        particle_tree, _ = get_particle_tree(
            N_dimensions,
            grid_dimensions,
            coordinates
        )

        # knn returns a multidimensional array of indices and distances
        # from each of the N_neighbor nearest neighbors. We want to
        # take the maximum distance as the smoothing length.
        _, dists = NearestNeighbors.knn(
            particle_tree, 
            @views(coordinates[1:N_dimensions, :]), 
            N_neighbor
        )

        smoothing_lengths = zeros(Float32, N_particles)

        @inbounds for i=1:N_particles
            smoothing_lengths[i] = maximum(dists[i])
        end
    end

    # Assume that it is x (idx=1) at first as the 
    # maximum dimension. Also sets for the 2D
    # case to always have the 3rd index be the
    # flat direction.
    secondary_idx::Integer = 2
    tertiary_idx::Integer = 3

    # Permute the indices since the directions may
    # no longer correspond to x,y,z because the 
    # direction of the maximum extent is treated as
    # the "main" coordinate direction.
    # 
    # Respect the right hand rule for coordinates
    # by assuming the maximum extent direction is
    # facing the observer.
    if N_dimensions == 3
        if primary_idx == 1
            secondary_idx = 3
            tertiary_idx = 2
        elseif primary_idx == 2
            secondary_idx = 1
            tertiary_idx = 3
        end
    end

    println("Dimensions: ")
    println("primary_idx=$primary_idx")
    println("secondary_idx=$secondary_idx")
    println("tertiary_idx=$tertiary_idx")

    idx_list = zeros(Integer, 3)
    idx_list[1] = primary_idx
    idx_list[2] = secondary_idx
    idx_list[3] = tertiary_idx
    
    # Always use 3 dimensions since we just set the
    # other dimensions to be empty in the lower
    # dimensional case.
    deltas = zeros(Float32, 3)
    @inbounds for i=1:3
        deltas[i] = grid_dimensions[i] / grid_sizes[i]
        @inbounds for j=1:grid_sizes[i]
            rulers[i][j] = ((j - 1) + 0.5) * deltas[i]
        end
    end

    println("Rulers: ")
    println("rulers[1]=", rulers[1])
    println("rulers[2]=", rulers[2])
    println("rulers[3]=", rulers[3])

    println("Deltas: ")
    println("deltas[1]=", deltas[1])
    println("deltas[2]=", deltas[2])
    println("deltas[3]=", deltas[3])

    # Sort all of the particles by their coordinates in 
    # the direction of the maximum grid extent, since
    # we will chunk in that direction.
    indices = sortperm(coordinates[primary_idx, :])
    coordinates = coordinates[:, indices]
    smoothing_lengths = smoothing_lengths[indices]
    deposits = deposits[indices]
    inv_smoothing_lengths = Float32(1.0) ./ smoothing_lengths

    # Loop over chunks of the particles. We will parallelize by
    # taking Threads.nthreads() chunks of the particles, and looping
    # over these chunks. Each thread does 1 chunk. We share the
    # particle information over all threads, in order to determine
    # which particles contribute to each chunk.
    chunk_size = convert(Int64, floor(length(indices) / Threads.nthreads()))
    remainder = convert(Int64, length(indices) - chunk_size * Threads.nthreads())
    first_chunk_size = convert(Int64, floor(grid_sizes[primary_idx] / Threads.nthreads()))
    first_remainder = grid_sizes[primary_idx] - first_chunk_size * Threads.nthreads()

    # If the maximum smoothing length is larger than a thread
    # chunk we crash.
    max_h = maximum(smoothing_lengths)
    if max_h > 1.0 / Threads.nthreads()
        error("At least one of the particle smoothing lengths is larger than a thread chunk. ",
              "Either reduce the number of threads, or remove spatially extensive particles.")
    end

    max_first_cells = trunc(Int64, 3.0 * max_h / deltas[primary_idx])

    if deltas[secondary_idx] == 0.0
        max_second_cells = 1
    else
        max_second_cells = trunc(Int64, 3.0 * max_h / deltas[secondary_idx]) + 1
    end

    if deltas[tertiary_idx] == 0.0
        max_third_cells = 1
    else
        max_third_cells = trunc(Int64, 3.0 * max_h / deltas[tertiary_idx]) + 1
    end

    println("max_h=$max_h")
    println("max_first_cells=$max_first_cells")
    println("max_second_cells=$max_second_cells")
    println("max_third_cells=$max_third_cells")

    println("Begin parallel loop with nthreads=", Threads.nthreads())
    println("chunk_size=$chunk_size")
    println("remainder=$remainder")
    println("first_chunk_size=$first_chunk_size")
    println("first_remainder=$first_remainder")

    Threads.@threads for N=1:Threads.nthreads()
        # Chunk along the direction of the maximum grid 
        # extent.
        #
        # What is the starting index of this grid chunk?
        # This is true for all chunks, even with the 
        # remainder.
        start_idx = (N - 1) * chunk_size + 1

        # The ending index is a bit different because of the 
        # remainder. If we are on the "last" thread, we need
        # to include the remainder or we will (could) have a 
        # small slice missing in the projection.
        end_idx = N * chunk_size
        if N == Threads.nthreads()
            end_idx += remainder
        end

        # We also need to know the particle chunk we are dealing
        # with so that we can check for boundary particles.
        first_start_idx = (N - 1) * first_chunk_size + 1
        first_end_idx = N * first_chunk_size
        if N == Threads.nthreads()
            first_end_idx += first_remainder
        end

        println()
        println("Begin particle loop on threadid=", Threads.threadid())
        println("first_start_idx=$first_start_idx")
        println("first_end_idx=$first_end_idx")
        println()

        # Preallocate some arrays
        kernel_weights = zeros(
            Float32, 
            (max_first_cells, max_second_cells, max_third_cells)
        )
        first_indices = zeros(Int64, max_first_cells)
        second_indices = zeros(Int64, max_second_cells)
        third_indices = zeros(Int64, max_third_cells)
        first_diffs = zeros(Float32, max_first_cells)
        second_diffs = zeros(Float32, max_second_cells)
        third_diffs = zeros(Float32, max_third_cells)

        # By this point we already know the subset of particles
        # that we have to deal with. Loop over all indices.
        @inbounds for p in start_idx:end_idx
            # We do this check here because we may have to redo
            # boundary particles in the parallelization scheme.
            # Therefore, we cannot have the check to see if they
            # are out-of-bounds buried inside of 
            # accumulate_in_grid() or else the boundary
            # particles would be skipped again!
            max_first_idx = Tools.get_max_idx(
                coordinates[primary_idx, p],
                smoothing_lengths[p],
                deltas[primary_idx]
            )
            min_first_idx = Tools.get_min_idx(
                coordinates[primary_idx, p],
                smoothing_lengths[p],
                deltas[primary_idx]
            )

            # We parallelize along the direction of maximum extent so 
            # we can first check to see if the total extent of this 
            # particle intersects a boundary.
            if min_first_idx < first_start_idx || max_first_idx > first_end_idx
                # Identify the boundary to the thread
                boundary_flag[p] = Threads.threadid()
                continue
            end

            accumulate_in_grid(
                p, 
                container, 
                grid,
                min_first_idx, 
                max_first_idx, 
                N_dimensions,
                coordinates[:, p],
                smoothing_lengths[p],
                inv_smoothing_lengths[p],
                deposits[p],
                idx_list,
                deltas,
                rulers,
                grid_sizes,
                kernel_weights,
                first_indices,
                second_indices,
                third_indices,
                first_diffs,
                second_diffs,
                third_diffs
            )
        end
    end  # End parallel loop

    println("End parallel loop on all threads")
    println()
    println("Begin parallel processing boundary particles.")
    println()

    # Now that we have done the parallel calculation,
    # we have to deal with any particles that were 
    # on the boundaries of our domain decomposition.
    # For now, let's just do a naive calculation 
    # of their deposition across the grid.
    #
    # (1, 2, 3, ..., N) then we can just parallelize across
    # those particles that have a certain boundary index,
    # given by the original threadid that they were 
    # found on.
    #
    # It is VERY important to use the same number of threads
    # here as the previous loop since the identifiers are for
    # number of threads that we had before (i.e. the boundary
    # field).
    Threads.@threads for N=1:Threads.nthreads()
        boundary_count = 0

        # Preallocate some arrays
        kernel_weights = zeros(
            Float32, 
            (max_first_cells, max_second_cells, max_third_cells)
        )
        first_indices = zeros(Int64, max_first_cells)
        second_indices = zeros(Int64, max_second_cells)
        third_indices = zeros(Int64, max_third_cells)
        first_diffs = zeros(Float32, max_first_cells)
        second_diffs = zeros(Float32, max_second_cells)
        third_diffs = zeros(Float32, max_third_cells)

        @inbounds for p=1:N_particles
            # Only do particles that were on the same thread boundary
            if boundary_flag[p] != Threads.threadid()
                continue
            end

            boundary_count += 1

            max_first_idx = Tools.get_max_idx(
                coordinates[primary_idx, p],
                smoothing_lengths[p],
                deltas[primary_idx]
            )
            min_first_idx = Tools.get_min_idx(
                coordinates[primary_idx, p],
                smoothing_lengths[p],
                deltas[primary_idx]
            )

            accumulate_in_grid(
                p, 
                container, 
                grid,
                min_first_idx, 
                max_first_idx, 
                N_dimensions,
                coordinates[:, p],
                smoothing_lengths[p],
                inv_smoothing_lengths[p],
                deposits[p],
                idx_list,
                deltas,
                rulers,
                grid_sizes,
                kernel_weights,
                first_indices,
                second_indices,
                third_indices,
                first_diffs,
                second_diffs,
                third_diffs
            )
        end

        println("boundary_count=$boundary_count on threadid=", Threads.threadid())
        println()
    end

    grid
end

end # module
