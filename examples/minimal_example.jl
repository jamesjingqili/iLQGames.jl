using iLQGames
import iLQGames: dx
import BenchmarkTools
using Plots

# parametes: number of states, number of inputs, sampling time, horizon
nx, nu, ΔT, game_horizon = 8, 4, 0.1, 100

# setup the dynamics
struct DoubleUnicycle <: ControlSystem{ΔT,nx,nu} end
# state: (px, py, phi, v)
dx(cs::DoubleUnicycle, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), u[3], u[4])
dynamics = DoubleUnicycle()

# costs = (FunctionPlayerCost((g, x, u, t) -> (10*(x[1]-1)^2 + 0.1*(x[3]-pi/2)^2 + (x[4]-1)^2 + u[1]^2 + u[2]^2 - 0.1*((x[1]-x[5])^2 + (x[2]-x[6])^2))),
         # FunctionPlayerCost((g, x, u, t) -> ((x[5]-1)^2 + 0.1*(x[7]-pi/2)^2 + (x[8]-1)^2 + u[3]^2 + u[4]^2- 0.1*((x[1]-x[5])^2 + (x[2]-x[6])^2))))
costs = (FunctionPlayerCost((g, x, u, t) -> (10*(x[1]-1)^2 + 0*(x[3]-pi/2)^2 + (x[4]-1)^2 + u[1]^2 + u[2]^2 - 0.1*((x[1]-x[5])^2 + (x[2]-x[6])^2))),
         FunctionPlayerCost((g, x, u, t) -> ((x[5]-1)^2 + 0*(x[7]-pi/2)^2 + (x[8]-1)^2 + u[3]^2 + u[4]^2- 0.1*((x[1]-x[5])^2 + (x[2]-x[6])^2))))

# indices of inputs that each player controls
player_inputs = (SVector(1,2), SVector(3,4))
# the horizon of the game
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)

# get a solver, choose initial conditions and solve (in about 9 ms with AD)
solver1 = iLQSolver(g, equilibrium_type="OLNE_KKT")
x0 = SVector(0, 0, pi/2, 1,       1, 0, pi/2, 1)
converged1, trajectory1, strategies1 = solve(g, solver1, x0)

solver2 = iLQSolver(g, equilibrium_type="FBNE_KKT")
x0 = SVector(0, 0, pi/2, 1,       1, 0, pi/2, 1)
converged2, trajectory2, strategies2 = solve(g, solver2, x0)

x1_OL, y1_OL = [trajectory1.x[i][1] for i in 1:game_horizon], [trajectory1.x[i][2] for i in 1:game_horizon];
x2_OL, y2_OL = [trajectory1.x[i][5] for i in 1:game_horizon], [trajectory1.x[i][6] for i in 1:game_horizon];

x1_FB, y1_FB = [trajectory2.x[i][1] for i in 1:game_horizon], [trajectory2.x[i][2] for i in 1:game_horizon];
x2_FB, y2_FB = [trajectory2.x[i][5] for i in 1:game_horizon], [trajectory2.x[i][6] for i in 1:game_horizon];


anim1 = @animate for i in 1:game_horizon
    plot([x1_OL[i], x1_OL[i]], [y1_OL[i], y1_OL[i]], markershape = :square, label = "player 1, OL", xlims = (0, 1.5), ylims = (0, 6))
    plot!([x2_OL[i], x2_OL[i]], [y2_OL[i], y2_OL[i]], markershape = :square, label = "player 2, OL", xlims = (0, 1.5), ylims = (0, 6))
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "") 
end

gif(anim1, "car_OLNE_KKT.gif", fps = 10)


anim2 = @animate for i in 1:game_horizon
    plot([x1_FB[i], x1_FB[i]], [y1_FB[i], y1_FB[i]], markershape = :square, label = "player 1, FB", xlims = (0, 1.5), ylims = (0, 6))
    plot!([x2_FB[i], x2_FB[i]], [y2_FB[i], y2_FB[i]], markershape = :square, label = "player 2, FB", xlims = (0, 1.5), ylims = (0, 6))    
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "")
end

gif(anim2, "car_FBNE_KKT.gif", fps = 10)

