module ParticleInterpolator

using NearestNeighbors

include("Tools.jl")
include("kernels/Kernels.jl")
include("interpolators/Interpolators.jl")


"""
    accumulate_in_grid(p::Int64, container::Interpolators.Container,
                       min_x_idx::Int32, max_x_idx::Int32,
                       N_dimensions::Integer, 
                       grid_info::Dict)

    Find the contributions to the grid cells by each particle,
    then accumulate the values in the grid.
...
# Arguments
- `p::Int64`: The particle index.
- `container::Container`: A Container struct that contains the particle data, 
                          tree, and grid.
- `min_x_idx::Int32`: The minimum grid index in the x direction of the particle.
- `max_x_idx::Int32`: The maximum grid index in the x direction of the particle.
- `N_dimensions::Integer`: What is the dimension of the simulation?
- `grid_info::Dict`: All of the information about the grid.
...
"""
function accumulate_in_grid(p::Int64, container::Interpolators.Container,
                            min_x_idx::Int32, max_x_idx::Int32,
                            N_dimensions::Integer, 
                            grid_info::Dict)

    # All of the interpolation methods rely on some 
    # kernel with compact support. That means that we
    # can safely ignore everythign outside of some 
    # extent, h.
    max_y_idx = Tools.get_max_idx(
        container.particle_data["coordinates"][2, p],
        container.particle_data["smoothing_lengths"][p],
        grid_info["delta"]
    )
    min_y_idx = Tools.get_min_idx(
        container.particle_data["coordinates"][2, p],
        container.particle_data["smoothing_lengths"][p],
        grid_info["delta"]
    )

    # What is the total extent in cells of the particle?
    num_x_cells = max_x_idx - min_x_idx + 1
    num_y_cells = max_y_idx - min_y_idx + 1
    total_cell_chunk = num_x_cells * num_y_cells

    if N_dimensions == 3
        max_z_idx = Tools.get_max_idx(
            container.particle_data["coordinates"][3, p],
            container.particle_data["smoothing_lengths"][p],
            grid_info["delta"]
        )
        min_z_idx = Tools.get_min_idx(
            container.particle_data["coordinates"][3, p],
            container.particle_data["smoothing_lengths"][p],
            grid_info["delta"]
        )

        num_z_cells = max_z_idx - min_z_idx + 1
        total_cell_chunk *= num_z_cells
    else
        # This will allow a triple loop over the chunk
        # spanned by the particle.
        num_z_cells = 1
        min_z_idx = 1
        max_z_idx = 1
    end

    x_indices = ones(Int32, num_x_cells)
    y_indices = ones(Int32, num_y_cells)  
    x_diffs = zeros(Float64, num_x_cells)
    y_diffs = zeros(Float64, num_y_cells)
    # We keep this set to zero so we don't have to
    # check inside of the triple loop if we are
    # in 3D.
    z_diffs = zeros(Float64, num_z_cells)

    Tools.get_indices_differences(
        min_x_idx,
        max_x_idx,
        x_indices, # This is filled up
        x_diffs, # This is filled up
        container.particle_data["coordinates"][1, p],
        grid_info
    )
    Tools.get_indices_differences(
        min_y_idx,
        max_y_idx,
        y_indices, # This is filled up
        y_diffs, # This is filled up
        container.particle_data["coordinates"][2, p],
        grid_info,
    )

    if N_dimensions == 3
        z_indices = zeros(Int32, num_z_cells)
        Tools.get_indices_differences(
            min_z_idx,
            max_z_idx,
            z_indices, # This is filled up
            z_diffs, # This is filled up
            container.particle_data["coordinates"][3, p],
            grid_info
        )
    else
        z_indices = ones(Int32, 1)
    end

    # This is the value of the kernel evaluated
    # at the distance to each cell.
    kernel_weight = zeros(Float64, (num_x_cells, num_y_cells, num_z_cells))

    # The sum of all of the weights that the
    # particle distributes across the small
    # domain.
    sum_of_kernel_weights::Float64 = 0

    # Particle is only going into one cell!
    smoothing_length_cond = 
            container.particle_data["smoothing_lengths"][p] < grid_info["delta"]
    cell_size_cond = num_x_cells == 1 && num_y_cells == 1 && num_z_cells == 1
    if cell_size_cond || smoothing_length_cond
        container.grid[x_indices[1], y_indices[1], z_indices[1]] += 
                container.particle_data["deposit"][p]
        return
    end

    # Loop over the entire chunk
    @inbounds for ci=1:num_x_cells, cj=1:num_y_cells, ck=1:num_z_cells
        distance = sqrt(
            x_diffs[ci] * x_diffs[ci] + 
            y_diffs[cj] * y_diffs[cj] + 
            z_diffs[ck] * z_diffs[ck]
        )

        kernel_weight[ci, cj, ck] = Interpolators.interpolate(
            p,
            distance,
            container
        )

        sum_of_kernel_weights += kernel_weight[ci, cj, ck]
    end

    if sum_of_kernel_weights == 0
        error("sum_of_kernel_weights=0 for 
              p=$p
              min_x_idx=$min_x_idx
              max_x_idx=$max_x_idx
              min_y_idx=$min_y_idx
              max_y_idx=$max_y_idx
              min_z_idx=$min_z_idx
              max_z_idx=$max_z_idx
              num_x_cells=$num_x_cells 
              num_y_cells=$num_y_cells 
              num_z_cells=$num_z_cells")
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
    @inbounds for ci=1:num_x_cells, cj=1:num_y_cells, ck=1:num_z_cells  
        kernel_weight_cijk = kernel_weight[ci, cj, ck] / sum_of_kernel_weights      
        container.grid[x_indices[ci], y_indices[cj], z_indices[ck]] += 
                container.particle_data["deposit"][p] *
                kernel_weight_cijk
    end
end

"""
    interpolate_particles(container::Container)

    Construct the grid to interpolate the particle data, and then interpolate. 
    The type of interpolation depends on the type of Container passed to the
    function.
...
# Arguments
- `container::Container`: A Container struct that contains the particle data, tree, and grid.
...
"""
function interpolate_particles(container::Interpolators.Container)
    # @TODO N_neighbor should be a parameter
    N_neighbor = 32
    N_particles = length(container.particle_data["coordinates"][1, :])

    if Threads.nthreads() > N_particles
        error("You have too many threads! Why are you even using threads!?")    
    end

    # In some parallelization schemes, we have to redo particles
    # that may exist on the boundaries between the domain
    # decomposition. We will mark these particles with a 1
    # for now, and redo them unparallelized at the end.
    container.particle_data["boundary"] = zeros(Int8, N_particles)

    particle_tree = NearestNeighbors.KDTree(
        container.particle_data["coordinates"],
        leafsize = 10
    )

    if !("smoothing_lengths" in keys(container.particle_data))
        # knn returns a multidimensional array of indices and distances
        # from each of the N_neighbor nearest neighbors. We want to
        # take the maximum distance as the smoothing length.
        idxs, dists = NearestNeighbors.knn(
            particle_tree, 
            container.particle_data["coordinates"], 
            N_neighbor
        )

        container.particle_data["smoothing_lengths"] = zeros(Float64, N_particles)

        @inbounds for i in eachIndex(dists)
            container.particle_data["smoothing_lengths"][i] = maximum(dists[i, :])
        end
    end

    grid_size = size(container.grid)
    grid_size_1d = grid_size[1]
    N_dimensions = container.kernel.dimension
    ruler = zeros(Float64, grid_size_1d)
    delta = 1.0 / grid_size_1d

    @inbounds for i=1:grid_size_1d
        ruler[i] = ((i - 1) + 0.5) * delta;
    end

    grid_info = Dict(
        "delta" => delta,
        "grid_size_1d" => grid_size_1d,
        "ruler" => ruler
    )

    # Sort all of the particles by their x-coordinate since
    # we will chunk in that direction.
    indices = sortperm(container.particle_data["coordinates"][1, :])

    # Loop over chunks of the particles. We will parallelize by
    # taking Threads.nthreads() chunks of the particles, and looping
    # over these chunks. Each thread does 1 chunk. We share the
    # particle information over all threads, in order to determine
    # which particles contribute to each chunk.
    chunk_size = convert(UInt64, floor(length(indices) / Threads.nthreads()))
    remainder = convert(UInt64, length(indices) - chunk_size * Threads.nthreads())
    x_chunk_size = floor(grid_size_1d / Threads.nthreads())
    x_remainder = grid_size_1d - x_chunk_size * Threads.nthreads()
    Threads.@threads for N = 1:Threads.nthreads()
        # We will chunk along the x-axis.
        #
        # What is the starting index of this PARTICLE
        # chunk?
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

        # We also need to know the x-chunk we are dealing with
        # so that we can check for boundary particles.
        x_start_idx = (N - 1) * x_chunk_size + 1
        x_end_idx = N * x_chunk_size
        if N == Threads.nthreads()
            x_end_idx += x_remainder
        end

        # By this point we already know the subset of particles
        # that we have to deal with. Loop over all indices.
        @inbounds for p in indices[start_idx:end_idx]
            # We do this check here because we may have to redo
            # boundary particles in the parallelization scheme.
            # Therefore, we cannot have the check to see if they
            # are out-of-bounds buried inside of 
            # accumulate_in_grid() or else the boundary
            # particles would be skipped again!
            max_x_idx = Tools.get_max_idx(
                container.particle_data["coordinates"][1, p],
                container.particle_data["smoothing_lengths"][p],
                delta
            )
            min_x_idx = Tools.get_min_idx(
                container.particle_data["coordinates"][1, p],
                container.particle_data["smoothing_lengths"][p],
                delta
            )

            # We parallelize along the x-axis so we can first check
            # to see if the total extent of this particle intersects 
            # a boundary.
            if min_x_idx < x_start_idx || max_x_idx > x_end_idx
                container.particle_data["boundary"][p] = 1
                continue
            end

            @fastmath accumulate_in_grid(
                p, 
                container, 
                min_x_idx, 
                max_x_idx, 
                N_dimensions,
                grid_info
            )
        end
    end  # End parallel loop

    # Now that we have done the parallel calculation,
    # we have to deal with any particles that were 
    # on the boundaries of our domain decomposition.
    # For now, let's just do a naive calculation 
    # of their deposition across the grid.
    #
    # TODO: Parallelize? I think if we index the boundaries
    # (1, 2, 3, ..., N) then we can just parallelize across
    # those particles that have a certain boundary index.
    @inbounds for p=1:N_particles
        if container.particle_data["boundary"][p] < 1
            continue
        end

        max_x_idx = Tools.get_max_idx(
                container.particle_data["coordinates"][1, p],
                container.particle_data["smoothing_lengths"][p],
                delta
        )
        min_x_idx = Tools.get_min_idx(
            container.particle_data["coordinates"][1, p],
            container.particle_data["smoothing_lengths"][p],
            delta
        )

        @fastmath accumulate_in_grid(
            p, 
            container, 
            min_x_idx, 
            max_x_idx,
            N_dimensions,
            grid_info
        )
    end
end

end # module
