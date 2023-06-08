using iLQGames
import iLQGames: dx
import BenchmarkTools
using Plots
using ForwardDiff

nx, nu, ΔT, game_horizon = 8+1, 4, 0.1, 10


# dynamics in the mind of the first player: iLQR, substituting player 2's control with player 2's varying belief
struct player1_dynamics <: ControlSystem{ΔT, 8+1+2, 4 } end
dx(cs::player1_dynamics, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), u[3], u[4], 
                                    0, 
                                    0, 
                                    0
                                    )
dynamics1 = player1_dynamics()


# dynamics in the mind of the second player: iLQGames, substituting player 1's control for player 2's belief update!
struct player2_dynamics <: ControlSystem{ΔT, 8+1+2, 4 } end
dx(cs::player2_dynamics, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), u[3], u[4],
                                    0, 
                                      1/ΔT * x[11]*π1[t]*inv(1+π1[t]*x[11]*π1[t]')* ([u[1]; u[2]] - π1[t]*x ),  # mean
                                    - 1/ΔT * x[11]*π1[t]*inv(1+π1[t]*x[11]*π1[t]')* π1[t]*x[11]                 # variance
                                    )
dynamics2 = player2_dynamics()


# dynamics of the 2-player game
struct total_dynamics <: ControlSystem{ΔT, 8+1+2, 4 } end
dx(cs::total_dynamics, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), u[3], u[4], 
                                    0, 0, 0
                                    )
dynamics3 = total_dynamics()




# note that we can define a dynamics model which is a function of the other player's control policy!!
π1_P_list = [ π1[t].P[1:2,:] for t in 1:game_horizon ]
π1_α_list = [ π1[t].α[1:2,:] for t in 1:game_horizon ]

π2_P_list = [ π2[t].P[3:4,:] for t in 1:game_horizon ]
π2_α_list = [ π2[t].α[3:4,:] for t in 1:game_horizon ]


costs = (FunctionPlayerCost((g, x, u, t) -> ( 2*(x[5]- x[9])^2  + u[1]^2 + u[2]^2 )),
         FunctionPlayerCost((g, x, u, t) -> (  (x[5] - x[1])^2 + 2*(x[8]-1)^2 + u[3]^2 + u[4]^2 )))
player_inputs = (SVector(1,2), SVector(3,4))


g1 = GeneralGame(game_horizon, player_inputs, dynamics1, costs)
g2 = GeneralGame(game_horizon, player_inputs, dynamics2, costs)
g3 = GeneralGame(game_horizon, player_inputs, dynamics3, costs)


x01 = SVector(0, 0.5, pi/2, 1,       1, 0, pi/2, 1,      0, 0, 1) # the last three states are: target lane, player 2's belief mean, player 2's belief variance
x02 = SVector(0, 0.5, pi/2, 1,       1, 0, pi/2, 1,      0, 1) # the last two states are: player 2's belief mean, player 2's belief variance
x03 = SVector(0, 0.5, pi/2, 1,       1, 0, pi/2, 1,      1, 0, 1) # the last three states are: target lane, player 2's belief mean, player 2's belief variance






solver1 = iLQSolver(g1, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="Stackelberg_KKT_dynamic_factorization")
c1, x1, π1 = solve(g1, solver1, x01)

solver2 = iLQSolver(g2, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="Stackelberg_KKT_dynamic_factorization")
c2, x2, π2 = solve(g2, solver2, x02)

solver3 = iLQSolver(g3, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="Stackelberg_KKT_dynamic_factorization")
c3, x3, π3 = solve(g3, solver3, x03)



# TODO: check Gaussian prior + Gaussian likelihood or Bernoulli + Beta


x1_FB, y1_FB = [x1.x[i][1] for i in 1:game_horizon], [x1.x[i][2] for i in 1:game_horizon];
x2_FB, y2_FB = [x1.x[i][5] for i in 1:game_horizon], [x1.x[i][6] for i in 1:game_horizon];
anim = @animate for i in 1:game_horizon
    plot([x1_FB[i], x1_FB[i]], [y1_FB[i], y1_FB[i]], markershape = :square, label = "player 1, Stackelberg KKT", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([x2_FB[i], x2_FB[i]], [y2_FB[i], y2_FB[i]], markershape = :square, label = "player 2, Stackelberg KKT", xlims = (-0.5, 1.5), ylims = (0, 6))    
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "")
end

gif(anim, "lane_guiding_Stackelberg.gif", fps = 10)






