
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

A = I(2);
B1 = [1;0]
B2 = [0;1]
Q1 = [4 0; 0 4]
Q2 = [1 0; 0 1]
R1 = 1
R2 = 4
T = 6
x0 = [1;2]

include("src/inverse_LQ_loss.jl")


nx, nu, ΔT, game_horizon = 2, 2, 0.1, 10
struct LinearSystem <: ControlSystem{ΔT,nx,nu} end
dx(cs::LinearSystem, x, u, t) = SVector(2*u[1] + u[2], u[1] + 2*u[2])
dynamics = LinearSystem()

costs_expert = (FunctionPlayerCost((g, x, u, t) -> ( 4*x[1]^2 + 4*x[2]^2 + u[1]^2 )),
         FunctionPlayerCost((g, x, u, t) -> ( x[1]^2 + x[2]^2 + 4*u[2]^2)))

# indices of inputs that each player controls
player_inputs = (SVector(1), SVector(2))
# the horizon of the game
g_expert = GeneralGame(game_horizon, player_inputs, dynamics, costs_expert)
x0 = SVector(1,2)


solver = iLQSolver(g_expert, max_scale_backtrack=10, max_elwise_diff_step=Inf,max_n_iter = 10, equilibrium_type="FBNE")
@time c_expert, x_expert, π_expert = solve(g_expert, solver, x0)



# the blows are for testing the inverse_LQ_loss function
sol = [10,10]

costs = (FunctionPlayerCost((g, x, u, t) -> ( sol[1]*x[1]^2 + sol[1]*x[2]^2 + u[1]^2 )),
            FunctionPlayerCost((g, x, u, t) -> ( sol[2]*x[1]^2 + sol[2]*x[2]^2 + 4*u[2]^2)))

g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
solver = iLQSolver(g, max_scale_backtrack=10, max_elwise_diff_step=Inf,max_n_iter = 10, equilibrium_type="FBNE")
c, x, π = solve(g, solver, x0)
loss = norm(x_expert.x - x.x,2 )

inverse_LQ_loss(x_expert, sol, game_horizon, player_inputs, dynamics, x0)

ForwardDiff.gradient(sol -> inverse_LQ_loss(x_expert, sol, game_horizon, player_inputs, dynamics, x0), sol)


