# copied from linear_system_example.jl
using iLQGames
import iLQGames: dx
import BenchmarkTools
using Plots
using ForwardDiff
using iLQGames:
    SystemTrajectory
using Infiltrator
using Optim
using LinearAlgebra
include("diff_solver.jl")
include("inverse_game_solver.jl")

nx, nu, ΔT, game_horizon = 4, 4, 0.1, 5

struct LinearSystem <: ControlSystem{ΔT,nx,nu} end
dx(cs::LinearSystem, x, u, t) = SVector(u[1],u[2],u[3],u[4])
dynamics = LinearSystem()

costs = (FunctionPlayerCost((g, x, u, t) -> ( 2*(x[3])^2 + 2*(x[4])^2 + u[1]^2 + u[2]^2)),
         FunctionPlayerCost((g, x, u, t) -> ( 2*(x[1]-x[3])^2 + 2*(x[2]-x[4])^2 + u[3]^2 + u[4]^2)))

# indices of inputs that each player controls
player_inputs = (SVector(1,2), SVector(3,4))
# the horizon of the game
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
x0 = SVector(0, 1, 1,1)


solver = iLQSolver(g, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type="Stackelberg_KKT")
@time c, x, π = solve(g, solver, x0)


solver1 = iLQSolver(g, max_scale_backtrack=10, max_elwise_diff_step=Inf, equilibrium_type="Stackelberg_KKT_dynamic_factorization")
@time c1, x1, π1 = solve(g, solver1, x0)


