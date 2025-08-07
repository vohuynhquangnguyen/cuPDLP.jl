#!/usr/bin/env julia
################################################################################
# convert_to_canonical.jl
#
# Usage (from a shell or VS Code integrated terminal):
#   julia --project=@. convert_to_canonical.jl INPUT.mps OUTPUT.npz
################################################################################

# 1) Activate & instantiate your environment (one-time)
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# 2) Imports
using cuPDLP
using LinearAlgebra
using SparseArrays
using NPZ

# 3) Helper to print usage
function usage()
    println("Usage: julia --project=@. convert_to_canonical.jl INPUT.mps OUTPUT.npz")
end

# 4) Main conversion
function convert_main(mps_path::String, out_path::String)
    println("▶️  Reading MPS from: ", mps_path)
    prob = cuPDLP.qps_reader_to_standard_form(mps_path)

    # ──────────────── THE PROBLEM BLOCK ─────────────────────
    # pull out everything from the returned problem object:
    A           = prob.constraint_matrix
    b           = prob.right_hand_side
    lb          = prob.variable_lower_bound
    ub          = prob.variable_upper_bound
    c           = prob.objective_vector
    const_term  = prob.objective_constant
    n_eq        = prob.num_equalities

    # these two may or may not exist depending on your cuPDLP version;
    # if they aren’t fields on `prob`, just default to no-scaling:
    row_scale = hasfield(typeof(prob), :row_scaling_vec)   ? prob.row_scaling_vec   : ones(size(A,1))
    col_scale = hasfield(typeof(prob), :col_scaling_vec)   ? prob.col_scaling_vec   : ones(size(A,2))
    # ───────────────────────────────────────────────────────

    # now export the sparse-matrix triplet plus all the vectors
    iu, ju, vu = findnz(A)    # 1-based row/col indices
    println("▶️  Writing canonical data to: ", out_path)
    npzwrite(out_path, Dict(
        "A_row"         => iu .- 1,        # zero-base for Python
        "A_col"         => ju .- 1,
        "A_data"        => vu,
        "A_shape"       => collect(size(A)),
        "b"             => b,
        "c"             => c,
        "lb"            => lb,
        "ub"            => ub,
        "row_scale_vec" => row_scale,
        "col_scale_vec" => col_scale,
        "n_eq"          => Int32[n_eq],
        "const_term"    => [const_term]
    ))
    println("✅ Done.")
end

# 5) Only perform argument-checking & conversion when run as a script:
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 2
        usage()
        exit(1)
    end
    convert_main(ARGS[1], ARGS[2])
end
