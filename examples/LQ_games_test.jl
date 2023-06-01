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


nx, nu, ΔT, game_horizon = 4, 4, 0.1, 10

struct LinearSystem <: ControlSystem{ΔT,nx,nu} end
dx(cs::LinearSystem, x, u, t) = SVector(u[1],u[2],u[3],u[4])
dynamics = LinearSystem()

costs = (FunctionPlayerCost((g, x, u, t) -> ( 2*(x[3])^2 + 2*(x[4])^2 + u[1]^2 + u[2]^2)),
         FunctionPlayerCost((g, x, u, t) -> ( 2*(x[1]-x[3])^2 + 2*(x[2]-x[4])^2 + u[3]^2 + u[4]^2)))

# indices of inputs that each player controls
player_inputs = (SVector(1,2), SVector(3,4))
# the horizon of the game
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
x0 = SVector(0, 1, 1,1/3)

solver0 = iLQSolver(g)
convergence, x_traj, strategies = solve(g, solver0, x0)
