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
              0, 1.0, 1) # the last two states are: player 2's belief mean, player 2's belief variance
solver21 = iLQSolver(g21, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")#Stackelberg_KKT_dynamic_factorization
c21, x21, π21 = solve(g21, solver21, x021)

x211_FB, y211_FB = [x21.x[i][1] for i in 1:game_horizon], [x21.x[i][2] for i in 1:game_horizon];
x212_FB, y212_FB = [x21.x[i][5] for i in 1:game_horizon], [x21.x[i][6] for i in 1:game_horizon];


π21_P_list = [ π21[t].P for t in 1:game_horizon ]
push!(π21_P_list, π21_P_list[end])
π21_α_list = [ π21[t].α for t in 1:game_horizon ]
push!(π21_α_list, π21_α_list[end])
u_list = [ x21.u[t] for t in 1:game_horizon ]
push!(u_list, u_list[end])
x_list = [ x21.x[t] for t in 1:game_horizon ]
push!(x_list, x_list[end])









# Problems: 1. u_t = \hat{u}_t + K_t(x - \hat{x}_t) + α_t, deep RL policy?
#           2. b(0) = 0.1, not working, now works.
#           3. b(0) = 0.2, not working, if 1/ΔT, now works.

















# TODO: substituting the above controller to player 1's mind for computing iLQR
# STEP 2:
# dynamics in the mind of the first player: iLQR, substituting player 2's control with player 2's varying belief
x01 = SVector(0, 0.5, pi/2, 1,       1, 0, pi/2, 1,      θ, 1.0, 1) # the last three states are: target lane, player 2's belief mean, player 2's belief variance

struct player1_dynamics <: ControlSystem{ΔT, 8+1+2, 4 } end
dx(cs::player1_dynamics, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), 0, 0, 
                                    0, 
                                    1/ΔT* x[11]*π21_P_list[Int(floor(t/ΔT))+1][1:2,10]'*inv(I(2)+π21_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]*π21_P_list[Int(floor(t/ΔT))+1][1:2,10]') * ( 
                                      [u[1];u[2]] - u_list[Int(floor(t/ΔT))+1][1:2]+ 
                                      π21_P_list[Int(floor(t/ΔT))+1][1:2,:]*(-x+x_list[Int(floor(t/ΔT))+1]) + π21_α_list[Int(floor(t/ΔT))+1][1:2] ),  # mean 
                                    - 1/ΔT* x[11]*π21_P_list[Int(floor(t/ΔT))+1][1:2,10]'*inv(I(2)+π21_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]*π21_P_list[Int(floor(t/ΔT))+1][1:2,10]') * 
                                    π21_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]                 # variance
                                    ) + SVector(0,0,0,0,0,0,     
                                    u_list[Int(floor(t/ΔT))+1][3] - π21_P_list[Int(floor(t/ΔT))+1][3,:]'*(x-x_list[Int(floor(t/ΔT))+1])-π21_α_list[Int(floor(t/ΔT))+1][3], 
                                    u_list[Int(floor(t/ΔT))+1][4] - π21_P_list[Int(floor(t/ΔT))+1][4,:]'*(x-x_list[Int(floor(t/ΔT))+1])-π21_α_list[Int(floor(t/ΔT))+1][4],
                                    0,0,0 );
dynamics1 = player1_dynamics();
costs1 = (FunctionPlayerCost((g, x, u, t) -> (10*(x[5]-x[9])^2  + (x[3]-pi/2)^2 + u[1]^2 + u[2]^2 )), # target lane is x[10], mean of player 2 belief
         FunctionPlayerCost((g, x, u, t) -> (  4*(x[5]-x[1])^2  + (x[7]-pi/2)^2 + u[3]^2 + u[4]^2 ))); 
g1 = GeneralGame(game_horizon, player_inputs, dynamics1, costs1);
solver1 = iLQSolver(g1, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")
c1, x1, π1 = solve(g1, solver1, x01);





x1_list = [ x1.x[t] for t in 1:game_horizon ];
u1_list = [ x1.u[t] for t in 1:game_horizon ];
push!(x1_list, x1.x[game_horizon]);
push!(u1_list, x1.u[game_horizon]);



π1_P_list = [ π1[t].P for t in 1:game_horizon ];
push!(π1_P_list, π1_P_list[end]);
π1_α_list = [ π1[t].α for t in 1:game_horizon ];
push!(π1_α_list, π1_α_list[end]);

belief_list = [ x1.x[t][10] for t in 1:game_horizon ]




















# STEP 3:
# dynamics in the mind of the second player: iLQGames, substituting player 1's control for player 2's belief update!
struct player2_dynamics <: ControlSystem{ΔT, 8+1+2, 4 } end
dx(cs::player2_dynamics, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), u[3], u[4],
                                    0, 
                                    1/ΔT* x[11]*π1_P_list[Int(floor(t/ΔT))+1][1:2,10]'*inv(I(2)+π1_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]*π1_P_list[Int(floor(t/ΔT))+1][1:2,10]') * ( 
                                      [u[1];u[2]] - u1_list[Int(floor(t/ΔT))+1][1:2]+ 
                                      π1_P_list[Int(floor(t/ΔT))+1][1:2,:]*(-x+x1_list[Int(floor(t/ΔT))+1]) + π1_α_list[Int(floor(t/ΔT))+1][1:2] ),  # mean 
                                    - 1/ΔT* x[11]*π1_P_list[Int(floor(t/ΔT))+1][1:2,10]'*inv(I(2)+π1_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]*π1_P_list[Int(floor(t/ΔT))+1][1:2,10]') * 
                                    π1_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]                 # variance
                                    );
dynamics2 = player2_dynamics();

costs = (FunctionPlayerCost((g, x, u, t) -> (10*(x[5]-x[10])^2 + 2*(x[4]-1)^2  + u[1]^2 + u[2]^2 )),
         FunctionPlayerCost((g, x, u, t) -> (  4*(x[5] - x[1])^2  +2*(x[8]-1)^2  + u[3]^2 + u[4]^2 )));
g2 = GeneralGame(game_horizon, player_inputs, dynamics2, costs);
solver2 = iLQSolver(g2, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE");
c2, x2, π2 = solve(g2, solver2, x021);



π21_P_list = [ π2[t].P for t in 1:game_horizon ];
push!(π21_P_list, π21_P_list[end]);
π21_α_list = [ π2[t].α for t in 1:game_horizon ];
push!(π21_α_list, π21_α_list[end]);
u_list = [ x2.u[t] for t in 1:game_horizon ];
push!(u_list, u_list[end]);
x_list = [ x2.x[t] for t in 1:game_horizon ];
push!(x_list, x_list[end]);

belief_list = [ x2.x[t][10] for t in 1:game_horizon ]














# STEP 4:
# dynamics in the mind of the first player: iLQR, substituting player 2's control with player 2's varying belief
x01 = SVector(0, 0.5, pi/2, 1,       1, 0, pi/2, 1,      θ, 1.0, 1) # the last three states are: target lane, player 2's belief mean, player 2's belief variance

struct player1_dynamics <: ControlSystem{ΔT, 8+1+2, 4 } end
dx(cs::player1_dynamics, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), 0, 0, 
                                    0, 
                                    1/ΔT* x[11]*π21_P_list[Int(floor(t/ΔT))+1][1:2,10]'*inv(I(2)+π21_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]*π21_P_list[Int(floor(t/ΔT))+1][1:2,10]') * ( 
                                      [u[1];u[2]] - u_list[Int(floor(t/ΔT))+1][1:2]+ 
                                      π21_P_list[Int(floor(t/ΔT))+1][1:2,:]*(-x+x_list[Int(floor(t/ΔT))+1]) + π21_α_list[Int(floor(t/ΔT))+1][1:2] ),  # mean 
                                    - 1/ΔT* x[11]*π21_P_list[Int(floor(t/ΔT))+1][1:2,10]'*inv(I(2)+π21_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]*π21_P_list[Int(floor(t/ΔT))+1][1:2,10]') * 
                                    π21_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]                 # variance
                                    ) + SVector(0,0,0,0,0,0,     
                                    u_list[Int(floor(t/ΔT))+1][3] - π21_P_list[Int(floor(t/ΔT))+1][3,:]'*(x-x_list[Int(floor(t/ΔT))+1])-π21_α_list[Int(floor(t/ΔT))+1][3], 
                                    u_list[Int(floor(t/ΔT))+1][4] - π21_P_list[Int(floor(t/ΔT))+1][4,:]'*(x-x_list[Int(floor(t/ΔT))+1])-π21_α_list[Int(floor(t/ΔT))+1][4],
                                    0,0,0 );
dynamics1 = player1_dynamics();
costs1 = (FunctionPlayerCost((g, x, u, t) -> (10*(x[5]-x[9])^2  + (x[3]-pi/2)^2 + u[1]^2 + u[2]^2 )), # target lane is x[10], mean of player 2 belief
         FunctionPlayerCost((g, x, u, t) -> (  4*(x[5]-x[1])^2  + (x[7]-pi/2)^2 + u[3]^2 + u[4]^2 ))) ;
g1 = GeneralGame(game_horizon, player_inputs, dynamics1, costs1);
solver1 = iLQSolver(g1, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE");
c1, x1, π1 = solve(g1, solver1, x01);

x1_list = [ x1.x[t] for t in 1:game_horizon ];
u1_list = [ x1.u[t] for t in 1:game_horizon ];
push!(x1_list, x1.x[game_horizon]);
push!(u1_list, x1.u[game_horizon]);



π1_P_list = [ π1[t].P for t in 1:game_horizon ];
push!(π1_P_list, π1_P_list[end]);
π1_α_list = [ π1[t].α for t in 1:game_horizon ];
push!(π1_α_list, π1_α_list[end]);

belief_list = [ x1.x[t][10] for t in 1:game_horizon ]





