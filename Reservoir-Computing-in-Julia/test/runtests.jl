using Test
using Random

const SOURCE_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(SOURCE_DIR, "common.jl"))
using .RCCommon

@testset "RCCommon numerical semantics" begin
    train, t_train, test, t_test = generate_lorenz_split(
        t_train = 0.10, t_test = 0.05, t_transient = 0.10, dt = 0.01)
    @test size(train, 1) == length(t_train)
    @test size(test, 1) == length(t_test)
    @test t_train[1] == 0.0
    @test t_test[1] ≈ 0.01
    @test t_test[end] ≈ 0.05

    scaler = Standardizer(ones(4, 2))
    @test transform(scaler, ones(4, 2)) == zeros(4, 2)
    @test all(isfinite, scaler.sigma)

    truth = [0.0 1.0; 1.0 0.0]
    times = [0.1, 0.2]
    @test valid_prediction_time(truth, truth, times)[1] == 0.2
    bad = copy(truth); bad[1, :] .= 100.0
    @test valid_prediction_time(truth, bad, times)[1] == 0.1
    @test_throws DimensionMismatch valid_prediction_time(truth, truth[1:1, :], times)
    @test_throws ArgumentError valid_prediction_time(ones(2, 2), ones(2, 2), times)

    lerp = make_lerp([0.0, 1.0], [0.0 2.0; 2.0 4.0])
    @test lerp(0.5) ≈ [1.0, 3.0]
    @test_throws ArgumentError make_lerp([0.0], zeros(2, 1))
    @test_throws DimensionMismatch make_lerp([0.0, 1.0], zeros(2, 3))

    A = zeros(1, 1); W = ones(1, 1); data = reshape([1.0, 2.0], :, 1)
    @test drive_reservoir(A, W, data; f = identity) == reshape([1.0, 2.0], 1, :)
end

@testset "Regression guards for experiment alignment" begin
    lpct = read(joinpath(SOURCE_DIR, "RC_LPCTESN.jl"), String)
    lpct_ablate = read(joinpath(SOURCE_DIR, "RC_LPCTESN_ablate.jl"), String)
    lpct_deriv = read(joinpath(SOURCE_DIR, "RC_LPCTESN_deriv.jl"), String)
    @test occursin("(0.0, tgrid[end])", lpct)
    @test occursin("(0.0, tgrid[end])", lpct_ablate)
    @test occursin("(0.0, t_te[end])", lpct_deriv)

    narma = read(joinpath(SOURCE_DIR, "RC_NARMA10.jl"), String)
    @test occursin("y[tr .+ 1]", narma)
    @test occursin("y[te .+ 1]", narma)

    for name in ("RC_FHN_NN.jl", "RC_FHN_NN_expt.jl",
                 "RC_FHN_NN_lobe.jl", "RC_FHN_NN_nextstep.jl")
        source = read(joinpath(SOURCE_DIR, name), String)
        @test occursin("coupling_arg === nothing ? 0.3", source)
        @test occursin("ODEProblem(fhn_test!, r_end, (0.0, t_test[end]))", source)
    end

    shd = read(joinpath(SOURCE_DIR, "RC_SHD_sweep.jl"), String)
    @test occursin("stratified_fit_validation", shd)
    @test occursin("one-shot test accuracy", shd)
    coupling = read(joinpath(SOURCE_DIR, "RC_FHN_coupling_sweep.jl"), String)
    @test occursin("validation_data", coupling)
    @test occursin("Final held-out test", coupling)
    rollout = read(joinpath(SOURCE_DIR, "RC_FHN_NN_rollout.jl"), String)
    @test occursin("eval_lo = n_tr - eval_tail_steps", rollout)
    drybean = read(joinpath(SOURCE_DIR, "RC_drybean.jl"), String)
    @test occursin("stratified_split", drybean)
    @test occursin("reservoir_features", drybean)
end

@testset "All Julia entry points parse" begin
    for path in filter(p -> endswith(p, ".jl"), readdir(SOURCE_DIR; join = true))
        @test Meta.parseall(read(path, String)) isa Expr
    end
end
