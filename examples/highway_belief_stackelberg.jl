using iLQGames
import iLQGames: dx
import BenchmarkTools
using Plots
using ForwardDiff
using LinearAlgebra

nx, nu, ΔT, game_horizon = 8+1+2, 4, 0.1, 40
# weird behavior: when horizon = 40, fine, 50 or 100 blows up


θ = 0.2;

# STEP 1:
struct player21_dynamics <: ControlSystem{ΔT, 8+1+2, 4 } end
dx(cs::player21_dynamics, x, u, t) = SVector(x[4]cos(x[3]), 
                                             x[4]sin(x[3]), 
                                             u[1], 
                                             u[2], 
                                             x[8]cos(x[7]), 
                                             x[8]sin(x[7]), 
                                             u[3], 
                                             u[4],
                                             0,  # parameter of player 1, invisible to player 2
                                             0,  # mean is not updated in the first iteration
                                             0   # variance is not updated in the first iteration
                                            )
dynamics21 = player21_dynamics()
costs2 = (FunctionPlayerCost((g, x, u, t) -> (10*(x[5]-x[10])^2  + (x[3]-pi/2)^2 + u[1]^2 + u[2]^2 )), # target lane is x[10], mean of player 2 belief
         FunctionPlayerCost((g, x, u, t) -> (  4*(x[5] - x[1])^2   +(x[7]-pi/2)^2 + u[3]^2 + u[4]^2 ))) 

player_inputs = (SVector(1,2), SVector(3,4))
g21 = GeneralGame(game_horizon, player_inputs, dynamics21, costs2)
x021 = SVector(0, 0.5, pi/2, 1,       
              1, 0, pi/2, 1,  
              0, θ, 1) # the last two states are: player 2's belief mean, player 2's belief variance
solver21 = iLQSolver(g21, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")#Stackelberg_KKT_dynamic_factorization
c21, x21, π21 = solve(g21, solver21, x021)

x211_FB, y211_FB = [x21.x[i][1] for i in 1:game_horizon], [x21.x[i][2] for i in 1:game_horizon];
x212_FB, y212_FB = [x21.x[i][5] for i in 1:game_horizon], [x21.x[i][6] for i in 1:game_horizon];



plot(x211_FB,y211_FB)
plot!(x212_FB,y212_FB)




π21_P_list = [ π21[t].P[1:2,:] for t in 1:game_horizon ]
push!(π21_P_list, π21_P_list[end])
π21_α_list = [ π21[t].α[1:2,:] for t in 1:game_horizon ]
push!(π21_α_list, π21_α_list[end])

u_list = [ x21.u[t][1:2] for t in 1:game_horizon ]
push!(u_list, u_list[end])
x_list = [ x21.x[t] for t in 1:game_horizon ]
push!(x_list, x_list[end])














# step 1.5: test belief update in player 2's mind
struct player2_dynamics <: ControlSystem{ΔT, 8+1+2, 4 } end
# dx(cs::player2_dynamics, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
#                                     x[8]cos(x[7]), x[8]sin(x[7]), u[3], u[4],
#                                     0, 
#                                       1/ΔT * x[11]*π21_P_list[Int(floor(t/ΔT))+1][:,10]'*inv(I(2)+π21_P_list[Int(floor(t/ΔT))+1][:,10]*x[11]*π21_P_list[Int(floor(t/ΔT))+1][:,10]') * (π21_P_list[Int(floor(t/ΔT))+1][:,10] * (θ - x[10]) ),  # mean
#                                     - 1/ΔT * x[11]*π21_P_list[Int(floor(t/ΔT))+1][:,10]'*inv(I(2)+π21_P_list[Int(floor(t/ΔT))+1][:,10]*x[11]*π21_P_list[Int(floor(t/ΔT))+1][:,10]') * π21_P_list[Int(floor(t/ΔT))+1][:,10]*x[11]                 # variance
#                                     )

dx(cs::player2_dynamics, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), u[3], u[4],
                                    0, 
                                     1* x[11]*π21_P_list[Int(floor(t/ΔT))+1][:,10]'*inv(I(2)+π21_P_list[Int(floor(t/ΔT))+1][:,10]*x[11]*π21_P_list[Int(floor(t/ΔT))+1][:,10]') * ( 
                                      [u[1];u[2]] - u_list[Int(floor(t/ΔT))+1]+ 
                                      π21_P_list[Int(floor(t/ΔT))+1]*(x-x_list[Int(floor(t/ΔT))+1]) + π21_α_list[Int(floor(t/ΔT))+1][:] ),  # mean 
                                    - 1* x[11]*π21_P_list[Int(floor(t/ΔT))+1][:,10]'*inv(I(2)+π21_P_list[Int(floor(t/ΔT))+1][:,10]*x[11]*π21_P_list[Int(floor(t/ΔT))+1][:,10]') * π21_P_list[Int(floor(t/ΔT))+1][:,10]*x[11]                 # variance
                                    )
dynamics2 = player2_dynamics()
costs = (FunctionPlayerCost((g, x, u, t) -> (10*(x[5]-θ)^2  + (x[3]-pi/2)^2 + u[1]^2 + u[2]^2 )), # target lane is x[10], mean of player 2 belief
         FunctionPlayerCost((g, x, u, t) -> (  4*(x[5] - x[1])^2   +(x[7]-pi/2)^2 + u[3]^2 + u[4]^2 ))) 
g2 = GeneralGame(game_horizon, player_inputs, dynamics2, costs)
solver2 = iLQSolver(g2, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")
x02 = SVector(0, 0.5, pi/2, 1,       
              1, 0, pi/2, 1,  
              0.0, 0.1, 1) # the last two states are: player 2's belief mean, player 2's belief variance
c2, x2, π2 = solve(g2, solver2, x02)

x2_list = [ x2.x[t][10] for t in 1:game_horizon ]




# Problems: 1. \hat{u}_t + K_t(x - \hat{x}_t) + α_t = u_t, deep RL policy?
#           2. b(0) = 0.1, not working
#           3. b(0) = 0.2, not working, if 1/ΔT










# TODO: substituting the above controller to player 1's mind for computing iLQR
# STEP 2:
# dynamics in the mind of the first player: iLQR, substituting player 2's control with player 2's varying belief
x01 = SVector(0, 0.5, pi/2, 1,       1, 0, pi/2, 1,      0, 0, 1) # the last three states are: target lane, player 2's belief mean, player 2's belief variance

struct player1_dynamics <: ControlSystem{ΔT, 8+1+2, 4 } end
dx(cs::player1_dynamics, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), 0, 0, 
                                    0, 
                                    0, 
                                    0
                                    ) + SVector(0,0,0,0,0,0, -π21_P_list[Int(floor(t/ΔT))+1][1,:]'*x-π21_α_list[Int(floor(t/ΔT))+1][1], 
                                    -π21_P_list[Int(floor(t/ΔT))+1][2,:]'*x-π21_α_list[Int(floor(t/ΔT))+1][2],0,0,0 )
dynamics1 = player1_dynamics()
costs1 = (FunctionPlayerCost((g, x, u, t) -> (10*(x[5]-0.5)^2 + 2*(x[4]-1)^2  + u[1]^2 + u[2]^2 )),
         FunctionPlayerCost((g, x, u, t) -> (  4*(x[5] - x[1])^2  +2*(x[8]-1)^2  + u[3]^2 + u[4]^2 )))
g1 = GeneralGame(game_horizon, player_inputs, dynamics1, costs1)
solver1 = iLQSolver(g1, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")
c1, x1, π1 = solve(g1, solver1, x01)



























# STEP 3:
# dynamics in the mind of the second player: iLQGames, substituting player 1's control for player 2's belief update!
struct player2_dynamics <: ControlSystem{ΔT, 8+1+2, 4 } end
dx(cs::player2_dynamics, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), u[3], u[4],
                                    0, 
                                      1/ΔT * x[11]*π1[t].P[1:2,:]*inv(1+π1[t]*x[11]*π1[t]')* ([u[1]; u[2]] - π1[t]*x ),  # mean
                                    - 1/ΔT * x[11]*π1[t]*inv(1+π1[t]*x[11]*π1[t]')* π1[t]*x[11]                 # variance
                                    )
dynamics2 = player2_dynamics()
π1_P_list = [ π1[t].P[1:2,:] for t in 1:game_horizon ]
π1_α_list = [ π1[t].α[1:2,:] for t in 1:game_horizon ]


costs = (FunctionPlayerCost((g, x, u, t) -> (10*(x[5]-x[10])^2 + 2*(x[4]-1)^2  + u[1]^2 + u[2]^2 )),
         FunctionPlayerCost((g, x, u, t) -> (  4*(x[5] - x[1])^2  +2*(x[8]-1)^2  + u[3]^2 + u[4]^2 )))
g2 = GeneralGame(game_horizon, player_inputs, dynamics2, costs)
solver2 = iLQSolver(g2, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")
c2, x2, π2 = solve(g2, solver2, x02)



















# dynamics of the two-player game
struct total_dynamics <: ControlSystem{ΔT, 8+1+2, 4 } end
dx(cs::total_dynamics, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), u[3], u[4], 
                                    0, 0, 0
                                    )
dynamics3 = total_dynamics()




# note that we can define a dynamics model which is a function of the other player's control policy!!







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




x1_FB, y1_FB = [x1.x[i][1] for i in 1:game_horizon], [x1.x[i][2] for i in 1:game_horizon];
x2_FB, y2_FB = [x1.x[i][5] for i in 1:game_horizon], [x1.x[i][6] for i in 1:game_horizon];


anim = @animate for i in 1:game_horizon
    plot([x1_FB[i], x1_FB[i]], [y1_FB[i], y1_FB[i]], markershape = :square, label = "player 1, Stackelberg KKT", xlims = (-0.5, 1.5), ylims = (0, 6))
    plot!([x2_FB[i], x2_FB[i]], [y2_FB[i], y2_FB[i]], markershape = :square, label = "player 2, Stackelberg KKT", xlims = (-0.5, 1.5), ylims = (0, 6))    
    plot!([0], seriestype = "vline", color = "black", label = "")
    plot!([1], seriestype = "vline", color = "black", label = "")
end

gif(anim, "lane_guiding_Stackelberg.gif", fps = 10)



