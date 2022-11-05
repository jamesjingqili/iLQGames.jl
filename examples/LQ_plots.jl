


using iLQGames
import iLQGames: dx
using Plots
using ForwardDiff
using iLQGames:
    SystemTrajectory
using iLQGames:
    LinearSystem
using Infiltrator
using Optim
using LinearAlgebra
using Distributed
using Dates
using Statistics
include("../src/diff_solver.jl")
include("../src/inverse_game_solver.jl")
include("../src/experiment_utils.jl") # NOTICE!! Many functions are defined there.

num_players=3
# nx=18
nx, nu, ΔT, game_horizon = 4*num_players+1+5, 2*num_players, 0.1, 30
struct ThreeCar <: ControlSystem{ΔT,nx,nu} end
dx(cs::ThreeCar, x, u, t) = SVector(x[1],x[2],x[3],x[4]
                                        )
dynamics = LinearSystem()
# platonning
# x0 = SVector(0.0, 3, pi/2, 2,       0.3, 0, pi/2, 2,      0.7, 2,pi/2,1,                   0.2,     0, 10, 0, 10  )
# x0 = SVector(0.4, 3, pi/2, 2,       0.4, 0, pi/2, 2,      0.7, 2,pi/2,1,                   0.0,     0, 10, 0, 10  )
# costs = (FunctionPlayerCost((g,x,u,t) -> ( x[14]*(x[1])^2  + x[15]*(x[5]-x[13])^2  + 4*(x[3]-pi/2)^2   +8*(x[4]-2)^2       +2*(u[1]^2 + u[2]^2)    )),
#          FunctionPlayerCost((g,x,u,t) -> ( x[16]*(x[5])^2  + x[17]*(x[5]-x[1])^2     +  8*(x[8]-2)^2  +4*(x[7]-pi/2)^2     -log((x[5]-x[9])^2+(x[6]-x[10])^2)  +2*(u[3]^2+u[4]^2)    )),
#          FunctionPlayerCost((g,x,u,t) -> ( 2*(x[9]-x0[9])^2   +2*(u[5]^2+u[6]^2)  ))
#     )
# x0 = SVector(0.0, 1, pi/2, 2,       1, 0, pi/2, 2,   0.5, 0.5,pi/2,2,                   0.2, 0, 8, 8, 0)
# x0 = SVector(0.0, 1, pi/2, 2,       0.3, 0, pi/2, 2,   0.5, 0.5,pi/2,2,                   0.2, 0, 8, 8, 0, 2) # final draft
x0 = SVector(1,2,3,4) # test

costs = (FunctionPlayerCost((g,x,u,t) -> ( (u[1]^2 + u[2]^2)    )),
         FunctionPlayerCost((g,x,u,t) -> ( (u[3]^2+u[4]^2)    )),
         )
player_inputs = (SVector(1,2), SVector(3,4))
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
# get a solver, choose initial conditions and solve (in about 9 ms with AD)

# x0 = SVector(0.0, 3, pi/2, 2,       0.3, 0, pi/2, 1.5,      0.5, 2,pi/2,1,                   1,     0, 10, 0, 10  )


solver1 = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="OLNE_costate")
c1, expert_traj1, strategies1 = solve(g, solver1, x0)
solver2 = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="FBNE_costate")
c2, expert_traj2, strategies2 = solve(g, solver2, x0)

# function parameterized_cost(θ::Vector)
# costs = (FunctionPlayerCost((g,x,u,t) -> ( x[14]*(x[1])^2  + x[15]*(x[5]-x[13])^2  + 4*(x[3]-pi/2)^2   +8*(x[4]-2)^2       +2*(u[1]^2 + u[2]^2)    )),
#          FunctionPlayerCost((g,x,u,t) -> ( x[16]*(x[5])^2  + x[17]*(x[5]-x[1])^2     +  8*(x[8]-2)^2  +4*(x[7]-pi/2)^2     -log((x[5]-x[9])^2+(x[6]-x[10])^2)  +2*(u[3]^2+u[4]^2)    )),
#          FunctionPlayerCost((g,x,u,t) -> ( 2*(x[9]-x0[9])^2   +2*(u[5]^2+u[6]^2)  ))
#     )
#     return costs
# end
function parameterized_cost(θ::Vector)
costs = (FunctionPlayerCost((g,x,u,t) -> ( x[14]*(x[1])^2  + x[15]*(x[5]-x[13])^2  + 4*(x[3]-pi/2)^2   +8*(x[4]-2)^2       +2*(u[1]^2 + u[2]^2)    )),
         FunctionPlayerCost((g,x,u,t) -> ( x[16]*(x[5])^2  + x[17]*(x[5]-x[1])^2     +  8*(x[8]-2)^2  +4*(x[7]-pi/2)^2     -log((x[5]-x[9])^2+(x[6]-x[10])^2)  +2*(u[3]^2+u[4]^2)    )),
         FunctionPlayerCost((g,x,u,t) -> ( x[18]*(x[9]-x0[9])^2   +2*(u[5]^2+u[6]^2)  ))
    )
    return costs
end
θ_true = [0, 8, 8, 0,2]




x1_FB, y1_FB = [expert_traj2.x[i][1] for i in 1:game_horizon], [expert_traj2.x[i][2] for i in 1:game_horizon];
x2_FB, y2_FB = [expert_traj2.x[i][5] for i in 1:game_horizon], [expert_traj2.x[i][6] for i in 1:game_horizon];
x3_FB, y3_FB = [expert_traj2.x[i][9] for i in 1:game_horizon], [expert_traj2.x[i][10] for i in 1:game_horizon];
anim2 = @animate for i in 1:game_horizon
    plot([x1_FB[i], x1_FB[i]], [y1_FB[i], y1_FB[i]], markershape = :square, label = "player 1, FB", xlims = (-0.5, 1.5), ylims = (0, 8))
    plot!([x2_FB[i], x2_FB[i]], [y2_FB[i], y2_FB[i]], markershape = :square, label = "player 2, FB", xlims = (-0.5, 1.5), ylims = (0, 8))    
    plot!([x3_FB[i], x3_FB[i]], [y3_FB[i], y3_FB[i]], markershape = :square, label = "player 3, FB", xlims = (-0.5, 1.5), ylims = (0, 8))    
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "",size=(300,600),xlabel="p_x",ylabel="p_y", title="FBNE")
    plot!([x0[13]], seriestype="vline",linestyle=:dash, color="black", label="target lane")
end
gif(anim2, "lane_guiding_3cars_FB_moving.gif", fps = 8)
x1_OL, y1_OL = [expert_traj1.x[i][1] for i in 1:game_horizon], [expert_traj1.x[i][2] for i in 1:game_horizon];
x2_OL, y2_OL = [expert_traj1.x[i][5] for i in 1:game_horizon], [expert_traj1.x[i][6] for i in 1:game_horizon];
x3_OL, y3_OL = [expert_traj1.x[i][9] for i in 1:game_horizon], [expert_traj1.x[i][10] for i in 1:game_horizon];
anim1 = @animate for i in 1:game_horizon
    plot([x1_OL[i], x1_OL[i]], [y1_OL[i], y1_OL[i]], markershape = :square, label = "player 1, OL", xlims = (-0.5, 1.5), ylims = (0, 8))
    plot!([x2_OL[i], x2_OL[i]], [y2_OL[i], y2_OL[i]], markershape = :square, label = "player 2, OL", xlims = (-0.5, 1.5), ylims = (0, 8))
    plot!([x3_OL[i], x3_OL[i]], [y3_OL[i], y3_OL[i]], markershape = :square, label = "player 3, OL", xlims = (-0.5, 1.5), ylims = (0, 8))    
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "",size=(300,600),xlabel="p_x",ylabel="p_y", title="OLNE")
    plot!([x0[13]], seriestype="vline",linestyle=:dash, color="black", label="target lane")
end
gif(anim1, "lane_guiding_3cars_OL_moving.gif", fps = 8)

