using iLQGames
import iLQGames: dx
import BenchmarkTools
using Plots
using ForwardDiff

# parametes: number of states, number of inputs, sampling time, horizon
nx, nu, ΔT, game_horizon = 8, 4, 0.1, 50

# setup the dynamics
struct DoubleUnicycle <: ControlSystem{ΔT,nx,nu} end
# state: (px, py, phi, v)
dx(cs::DoubleUnicycle, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), u[3], u[4])
dynamics = DoubleUnicycle()

# costs = (FunctionPlayerCost((g, x, u, t) -> (10*(x[1]-1)^2 + 0.1*(x[3]-pi/2)^2 + (x[4]-1)^2 + u[1]^2 + u[2]^2 - 0.1*((x[1]-x[5])^2 + (x[2]-x[6])^2))),
         # FunctionPlayerCost((g, x, u, t) -> ((x[5]-1)^2 + 0.1*(x[7]-pi/2)^2 + (x[8]-1)^2 + u[3]^2 + u[4]^2- 0.1*((x[1]-x[5])^2 + (x[2]-x[6])^2))))
costs = (FunctionPlayerCost((g, x, u, t) -> ( 2*(x[5]-1)^2  + u[1]^2 + u[2]^2 )),
         FunctionPlayerCost((g, x, u, t) -> (  4*(x[5] - x[1])^2 + 2*(x[8]-1)^2 + u[3]^2 + u[4]^2 )))

# indices of inputs that each player controls
player_inputs = (SVector(1,2), SVector(3,4))
# the horizon of the game
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
x0 = SVector(0, 0.5, pi/2, 1,       1, 0, pi/2, 1)

# get a solver, choose initial conditions and solve (in about 9 ms with AD)



solver = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="Stackelberg_KKT_dynamic_factorization")
@time c, expert_traj, strategies = solve(g, solver, x0)

solver1 = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="Stackelberg_KKT")
@time c1, expert_traj1, strategies1 = solve(g, solver1, x0)

solver2 = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="FBNE")
@time c2, expert_traj2, strategies2 = solve(g, solver2, x0)

solver3 = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="FBNE_KKT")
@time c3, expert_traj3, strategies3 = solve(g, solver3, x0)



# belows are for plotting
x1_FB, y1_FB = [expert_traj.x[i][1] for i in 1:game_horizon], [expert_traj.x[i][2] for i in 1:game_horizon];
x2_FB, y2_FB = [expert_traj.x[i][5] for i in 1:game_horizon], [expert_traj.x[i][6] for i in 1:game_horizon];
anim = @animate for i in 1:game_horizon
    plot([x1_FB[i], x1_FB[i]], [y1_FB[i], y1_FB[i]], markershape = :square, label = "player 1, Stackelberg KKT", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([x2_FB[i], x2_FB[i]], [y2_FB[i], y2_FB[i]], markershape = :square, label = "player 2, Stackelberg KKT", xlims = (-0.5, 1.5), ylims = (0, 6))    
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "")
end

gif(anim, "lane_guiding_Stackelberg_dynamic.gif", fps = 10)



# lq_approx()
# solve_lq_FBNE!()
# trajectory!()

x1_FB, y1_FB = [expert_traj1.x[i][1] for i in 1:game_horizon], [expert_traj1.x[i][2] for i in 1:game_horizon];
x2_FB, y2_FB = [expert_traj1.x[i][5] for i in 1:game_horizon], [expert_traj1.x[i][6] for i in 1:game_horizon];
anim = @animate for i in 1:game_horizon
    plot([x1_FB[i], x1_FB[i]], [y1_FB[i], y1_FB[i]], markershape = :square, label = "player 1, Stackelberg KKT", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([x2_FB[i], x2_FB[i]], [y2_FB[i], y2_FB[i]], markershape = :square, label = "player 2, Stackelberg KKT", xlims = (-0.5, 1.5), ylims = (0, 6))    
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "")
end

gif(anim, "lane_guiding_Stackelberg_dynamic.gif", fps = 10)



