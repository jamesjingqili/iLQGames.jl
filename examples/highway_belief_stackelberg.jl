using iLQGames
import iLQGames: dx
import BenchmarkTools
using Plots
using ForwardDiff

nx, nu, ΔT, game_horizon = 8+1, 4, 0.1, 40


# dynamics in the mind of the first player: iLQR, substituting player 2's control with player 2's varying belief
struct player1_dynamics <: ControlSystem{ΔT, 8+1+2, 4 } end
dx(cs::player1_dynamics, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), 0, 0, 
                                    0
                                    )
dynamics1 = player1_dynamics()


# dynamics in the mind of the second player
struct player2_dynamics <: ControlSystem{ΔT, 8+2, 4 } end
dx(cs::player2_dynamics, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), u[3], u[4], 
                                    0
                                    )
dynamics2 = player2_dynamics()


struct total_dynamics <: ControlSystem{ΔT, 8+1+2, 4 } end
dx(cs::total_dynamics, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), u[3], u[4], 
                                    0
                                    )
dynamics2 = total_dynamics()






costs = (FunctionPlayerCost((g, x, u, t) -> ( 2*(x[5]-1)^2  + u[1]^2 + u[2]^2 )),
         FunctionPlayerCost((g, x, u, t) -> (  x[9]*(x[5] - x[1])^2 + 2*(x[8]-1)^2 + u[3]^2 + u[4]^2 )))

player_inputs = (SVector(1,2), SVector(3,4))

g1 = GeneralGame(game_horizon, (SVector(1,2), SVector(3,4)), (dynamics1, dynamics2), costs)



g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
x0 = SVector(0, 0.5, pi/2, 1,       1, 0, pi/2, 1)
solver = iLQSolver(g, max_scale_backtrack=5, max_elwise_diff_step=Inf, equilibrium_type="Stackelberg_KKT")
c, expert_traj, strategies = solve(g, solver, x0)


x1_FB, y1_FB = [expert_traj.x[i][1] for i in 1:game_horizon], [expert_traj.x[i][2] for i in 1:game_horizon];
x2_FB, y2_FB = [expert_traj.x[i][5] for i in 1:game_horizon], [expert_traj.x[i][6] for i in 1:game_horizon];
anim = @animate for i in 1:game_horizon
    plot([x1_FB[i], x1_FB[i]], [y1_FB[i], y1_FB[i]], markershape = :square, label = "player 1, Stackelberg KKT", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([x2_FB[i], x2_FB[i]], [y2_FB[i], y2_FB[i]], markershape = :square, label = "player 2, Stackelberg KKT", xlims = (-0.5, 1.5), ylims = (0, 6))    
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "")
end

gif(anim, "lane_guiding_Stackelberg.gif", fps = 10)






