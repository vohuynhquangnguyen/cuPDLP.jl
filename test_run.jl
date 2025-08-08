# test_run.jl

################################################################################
# 0. (Optional) If you run without `--project=.` you can activate here:
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

################################################################################
# 1. Import the solver
using cuPDLP
using LinearAlgebra   # for dot()

################################################################################
# 2. Read your already-decompressed MPS file into the internal QP type
# qp = cuPDLP.qps_reader_to_standard_form("qap15.mps")
qp = cuPDLP.qps_reader_to_standard_form("problems/relaxed/relaxed_gen-ip054.mps")


################################################################################
# 3. Build default termination criteria
termination = cuPDLP.construct_termination_criteria()

################################################################################
# 4. Build default restart scheme (adaptive KKT-based restarts)
restart = cuPDLP.construct_restart_parameters(
    cuPDLP.ADAPTIVE_KKT,     # restart scheme
    cuPDLP.KKT_GREEDY,       # restart metric
    1000,                    # freq (unused for ADAPTIVE_KKT)
    0.36,                    # artificial_restart_threshold
    0.20,                    # sufficient_reduction_for_restart
    0.80,                    # necessary_reduction_for_restart
    0.50,                    # primal_weight_update_smoothing
)

################################################################################
# 5. Build default stepsize policy (adaptive)
stepsize = cuPDLP.AdaptiveStepsizeParams(0.3, 0.6)

################################################################################
# 6. Pack everything into the PdhgParameters struct
params = cuPDLP.PdhgParameters(
    10,            # l_inf_ruiz_iterations
    false,         # l2_norm_rescaling
    1.0,           # pock_chambolle_alpha
    1.0,           # primal_importance
    true,          # scale_invariant_initial_primal_weight
    2,             # verbosity (1–3)
    true,          # record_iteration_stats
    Int32(64),     # termination_evaluation_frequency
    termination,   # TerminationCriteria
    restart,       # RestartParameters
    stepsize       # Stepsize policy
)

################################################################################
# 7. Solve via the core API
output = cuPDLP.optimize(params, qp)

################################################################################
# 8. Print solver summary
println("▶︎ Status:      ", output.termination_string)
println("▶︎ Iterations:  ", output.iteration_count)

# 9. Compute and print the LP objective: cᵀx + constant
c          = qp.objective_vector
x          = output.primal_solution
const_term = qp.objective_constant
obj        = dot(c, x) + const_term
println("▶︎ Objective:   ", obj)
