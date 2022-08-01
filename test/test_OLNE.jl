using Test
using StaticArrays
using LinearAlgebra
using BenchmarkTools

using iLQGames:
    LinearSystem,
    AffineStrategy,
    QuadraticPlayerCost,
    LTVSystem,
    LQGame,
    SystemTrajectory,
    solve_lq_game!,
    n_states,
    n_controls,
    n_players,
    uindex,
    horizon,
    samplingtime,
    dynamics,
    strategytype,
    player_costs,
    trajectory!


function generate_1D_pointmass_game()
    # Testing the solver at a simple example: A two-player point mass 1D system.
    # The state composes of position and oritation. Therefore, the system dynamics
    # are a pure integrator.
    ΔT = 0.1
    H = 10.0
    N_STEPS = Int(H / ΔT)

    # dynamical system
    A = SMatrix{2, 2}([1. ΔT; 0. 1.])
    B = SMatrix{2, 2}([0.5*ΔT^2 ΔT; 0.32*ΔT^2 0.11*ΔT]')
    dyn = LinearSystem{ΔT}(A, B)
    # costs for each player
    c1 = QuadraticPlayerCost(@SVector([0., 0.]),       # l
                             @SMatrix([1. 0.; 0. 1.]), # Q
                             @SVector([0., 0.]),       # r
                             @SMatrix([1. 0.; 0. 0.])) # R
    c2 = QuadraticPlayerCost(-c1.l,                    # l
                             -c1.Q,                    # Q
                             @SVector([0., 0.]),       # r
                             @SMatrix([0. 0.; 0. 1.])) # R

    costs = @SVector [c1, c2]
    # the lq game (player one has control input 1 and 2; player 2 has control input 3
    ltv_dyn = LTVSystem(SizedVector{N_STEPS}(repeat([dyn], N_STEPS)))
    qtv_costs = SizedVector{N_STEPS}(repeat([costs], N_STEPS))
    uids = (SVector(1), SVector(2))
    lqGame = LQGame(uids, ltv_dyn, qtv_costs)

    # test all the function calls:
    @test n_players(lqGame) == length(costs)
    @test horizon(lqGame) == N_STEPS
    uindex(lqGame)

    return lqGame
end


