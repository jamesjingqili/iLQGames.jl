using iLQGames
import iLQGames: dx
import BenchmarkTools
using Plots
using ForwardDiff
using LinearAlgebra

using Random
using Distributions

nx, nu, ΔT, game_horizon = 6, 2, 0.1, 6
# weird behavior: when horizon = 40, fine, 50 or 100 blows up

marker_list = LinRange(1, 2, game_horizon)
time_list = ΔT:ΔT:game_horizon*ΔT
θ = 1.0; # we classify θ to be 1 if θ > 0.5, 0 otherwise
initial_belief = 1
initial_state = SVector(-6, 1, -2,1)
initial_state_truth = vcat(initial_state, SVector(0.01,θ))
initial_state_1 = vcat(initial_state, SVector(initial_belief,initial_belief))
initial_state_2 = vcat(initial_state, SVector(θ,initial_belief))



# TODO: 

function ReLU(x)
  return [relu(item) for item in x]
end

function relu(x)
    if x > 0
        return x
    else
        return 0
    end
end


# ground truth:
struct player21_dynamics <: ControlSystem{ΔT, 6, 2 } end
dx(cs::player21_dynamics, x, u, t) = SVector(x[2], 
u[1], 
x[4], 
u[2], 
0.01,  # parameter of player 1, invisible to player 2
0,  # mean is not updated in the first iteration
)
dynamics = player21_dynamics()
costs = (FunctionPlayerCost((g, x, u, t) -> ( x[5]*(x[1]-10)^2 - log(x[3]-x[1]) + u[1]^2 )), # target lane is x[10], mean of player 2 belief
          FunctionPlayerCost((g, x, u, t) -> ( -log(x[3]-x[1])+ u[2]^2+ 20*relu(1.5-(x[3]+1)^2) ))) 

player_inputs = (SVector(1), SVector(2))
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
solver = iLQSolver(g, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")#Stackelberg_KKT_dynamic_factorization
c, x, π = solve(g, solver, SVector(-4,1,-2,1, 0.0, 0))

x1_FB, x2_FB = [x.x[i][1] for i in 1:game_horizon], [x.x[i][3] for i in 1:game_horizon];

scatter(x1_FB, 0*x1_FB,markersize=6*marker_list,label="player 1")
scatter!(x2_FB, 0*x2_FB,markersize=6*marker_list,label="player 2")
xlims!(-4,-1.5)

savefig("chill.png")
savefig("urgent.png")

π_P_list = [ π[t].P for t in 1:game_horizon ]
push!(π_P_list, π_P_list[end])
π_α_list = [ π[t].α for t in 1:game_horizon ]
push!(π_α_list, π_α_list[end])
u_list = [ x.u[t] for t in 1:game_horizon ]
push!(u_list, u_list[end])
x_list = [ x.x[t] for t in 1:game_horizon ]
push!(x_list, x_list[end])

plot(x1_FB, 0*x1_FB,label="player 1")
plot!(x2_FB, 0*x2_FB,label="player 2")
savefig("step0.png")



"""
#################
                    [Yellow light]
x1 -> x2 ->

#################
"""


# belief update for binary belief:
# next_belief_b1 = pdf.(Normal(control, 1), [control])*prior_b1/()



# TODO: try bigger variance for likelihood model variance. 



# posterior_of_b1  \propto  pdf.(Normal(optimal_control_under_b1, b1), [observed_control]) * prior_b1
# posterior_of_b2  \propto  pdf.(Normal(optimal_control_under_b2, b2), [observed_control]) * prior_b2




# STEP 1:
struct player_dynamics1 <: ControlSystem{ΔT, 6, 2 } end
dx(cs::player_dynamics1, x, u, t) = SVector(  
  x[2],
  u[1],
  x[4],
  u[2],
  1,  # parameter of player 1, invisible to player 2, binary variable
  0, #b1*pdf.(Normal(strategy1, b1), u[1]) / (b1*pdf.(Normal(strategy1, b1), u[1]) + (1-b1)*pdf.(Normal(strategy2, b2), u[1]) ),
)
dynamics1 = player_dynamics1()

costs1 = (FunctionPlayerCost((g, x, u, t) -> ( x[5]*(x[1]-10)^2 - log(x[3]-x[1]) + u[1]^2  )), # target lane is x[10], mean of player 2 belief
      FunctionPlayerCost((g, x, u, t) -> ( ReLU(1-(x[1]-1)^2) -log(x[3]-x[1])+ u[2]^2))) 

g1 = GeneralGame(game_horizon, player_inputs, dynamics1, costs1)
solver1 = iLQSolver(g1, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")#Stackelberg_KKT_dynamic_factorization
c1, x1, π1 = solve(g1, solver1, initial_state_1)

x11_FB, y11_FB = [x1.x[i][1] for i in 1:game_horizon], [x1.x[i][2] for i in 1:game_horizon];
x12_FB, y12_FB = [x1.x[i][5] for i in 1:game_horizon], [x1.x[i][6] for i in 1:game_horizon];


π1_P_list = [ π1[t].P for t in 1:game_horizon ]
push!(π1_P_list, π1_P_list[end])
π1_α_list = [ π1[t].α for t in 1:game_horizon ]
push!(π1_α_list, π1_α_list[end])
u_list = [ x1.u[t] for t in 1:game_horizon ]
push!(u_list, u_list[end])
x_list = [ x1.x[t] for t in 1:game_horizon ]
push!(x_list, x_list[end])
# step 1 finished!

plot(x11_FB, y11_FB,label="player 1")
plot!(x12_FB, y12_FB,label="player 2")
savefig("step1.png")





# Problems: 1. u_t = \hat{u}_t + K_t(x - \hat{x}_t) + α_t, deep RL policy?
#           2. b(0) = 0.1, not working, now works.
#           3. b(0) = 0.2, not working, if 1/ΔT, now works.

















# TODO: substituting the above controller to player 1's mind for computing iLQR
# STEP 2:
# dynamics in the mind of the first player: iLQR, substituting player 2's control with player 2's varying belief
x02 = SVector(0, 0.5, pi/2, 1,       1, 0, pi/2, 1,      θ, 1.0, 1) # the last three states are: target lane, player 2's belief mean, player 2's belief variance

struct player_dynamics2 <: ControlSystem{ΔT, 8+1+2, 4 } end
dx(cs::player_dynamics2, x, u, t) = SVector(x[2],
u[1],
x[4],
u[2],
0,
(pdf.(Normal(,1), u[2])*x[6])/(x[6]* + (1-x[6])* ), 
# -1/ΔT* x[11]*π1_P_list[Int(floor(t/ΔT))+1][1:2,10]'*inv(I(2)+π1_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]*π1_P_list[Int(floor(t/ΔT))+1][1:2,10]') * ( 
#   [u[1];u[2]] - u_list[Int(floor(t/ΔT))+1][1:2]+ 
#   π1_P_list[Int(floor(t/ΔT))+1][1:2,:]*(x-x_list[Int(floor(t/ΔT))+1]) + π1_α_list[Int(floor(t/ΔT))+1][1:2] ),  # mean 
# - 1/ΔT* x[11]*π1_P_list[Int(floor(t/ΔT))+1][1:2,10]'*inv(I(2)+π1_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]*π1_P_list[Int(floor(t/ΔT))+1][1:2,10]') * 
# π1_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]                 # variance
) + SVector(0,0,0,0,0,0,     
u_list[Int(floor(t/ΔT))+1][3] - π1_P_list[Int(floor(t/ΔT))+1][3,:]'*(x-x_list[Int(floor(t/ΔT))+1])-π1_α_list[Int(floor(t/ΔT))+1][3], 
u_list[Int(floor(t/ΔT))+1][4] - π1_P_list[Int(floor(t/ΔT))+1][4,:]'*(x-x_list[Int(floor(t/ΔT))+1])-π1_α_list[Int(floor(t/ΔT))+1][4],
0,0,0 );
dynamics2 = player_dynamics2();
costs2 = (FunctionPlayerCost((g, x, u, t) -> (10*(x[5]-x[9])^2  + (x[3]-pi/2)^2 + u[1]^2 + u[2]^2 + 50*(x[9]-x[10])^2 )), # target lane is x[10], mean of player 2 belief
          FunctionPlayerCost((g, x, u, t) -> (  4*(x[5]-x[1])^2  + (x[7]-pi/2)^2 + u[3]^2 + u[4]^2 ))); 
g2 = GeneralGame(game_horizon, player_inputs, dynamics2, costs1);
solver2 = iLQSolver(g2, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")
c2, x2, π2 = solve(g2, solver2, initial_state_2);





x2_list = [ x2.x[t] for t in 1:game_horizon ];
u2_list = [ x2.u[t] for t in 1:game_horizon ];
push!(x2_list, x2.x[game_horizon]);
push!(u2_list, x2.u[game_horizon]);



π2_P_list = [ π2[t].P for t in 1:game_horizon ];
push!(π2_P_list, π2_P_list[end]);
π2_α_list = [ π2[t].α for t in 1:game_horizon ];
push!(π2_α_list, π2_α_list[end]);

belief_list = [ x2.x[t][10] for t in 1:game_horizon ]
var_list = [x2.x[t][11] for t in 1:game_horizon]
# step 2 finished!



x21_FB, y21_FB = [x2.x[i][1] for i in 1:game_horizon], [x2.x[i][2] for i in 1:game_horizon];
x22_FB, y22_FB = [x2.x[i][5] for i in 1:game_horizon], [x2.x[i][6] for i in 1:game_horizon];

plot(x21_FB, y21_FB,label="player 1")
plot!(x22_FB, y22_FB,label="player 2")
savefig("step2.png")
















# TODO: stepsize: α = 0.1.  new_policy = old_policy + α * (new_policy - old_policy)



# STEP 3:
# dynamics in the mind of the second player: iLQGames, substituting player 1's control for player 2's belief update!
struct player_dynamics3 <: ControlSystem{ΔT, 8+1+2, 4 } end
dx(cs::player_dynamics3, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), u[3], u[4],
                                    0, 
                                    -1/ΔT* x[11]*π2_P_list[Int(floor(t/ΔT))+1][1:2,10]'*inv(I(2)+π2_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]*π2_P_list[Int(floor(t/ΔT))+1][1:2,10]') * ( 
                                      [u[1];u[2]] - u2_list[Int(floor(t/ΔT))+1][1:2]+ 
                                      π2_P_list[Int(floor(t/ΔT))+1][1:2,:]*(x-x2_list[Int(floor(t/ΔT))+1]) + π2_α_list[Int(floor(t/ΔT))+1][1:2] ),  # mean 
                                    - 1/ΔT* x[11]*π2_P_list[Int(floor(t/ΔT))+1][1:2,10]'*inv(I(2)+π2_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]*π2_P_list[Int(floor(t/ΔT))+1][1:2,10]') * 
                                    π2_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]                 # variance
                                    );
dynamics3 = player_dynamics3();

costs3 = (FunctionPlayerCost((g, x, u, t) -> (10*(x[5]-x[10])^2 + 2*(x[4]-1)^2  + u[1]^2 + u[2]^2 + 50*(x[9]-x[10])^2)),
          FunctionPlayerCost((g, x, u, t) -> (  4*(x[5] - x[1])^2  +2*(x[8]-1)^2  + u[3]^2 + u[4]^2 )));
g3 = GeneralGame(game_horizon, player_inputs, dynamics3, costs3);
solver3 = iLQSolver(g3, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE");
c3, x3, π3 = solve(g3, solver3, initial_state_1);



π3_P_list = [π3[t].P for t in 1:game_horizon ];
push!(π3_P_list, π3_P_list[end]);
π3_α_list = [π3[t].α for t in 1:game_horizon ];
push!(π3_α_list, π3_α_list[end]);

u3_list = [ x3.u[t] for t in 1:game_horizon ];
push!(u3_list, u3_list[end]);
x3_list = [ x3.x[t] for t in 1:game_horizon ];
push!(x3_list, x3_list[end]);

belief_list = [ x3.x[t][10] for t in 1:game_horizon ]
var_list = [x3.x[t][11] for t in 1:game_horizon]
# step 3 finished!



# x1_FB, y1_FB = [x3.x[i][1] for i in 1:game_horizon], [x3.x[i][2] for i in 1:game_horizon];
# x2_FB, y2_FB = [x3.x[i][5] for i in 1:game_horizon], [x3.x[i][6] for i in 1:game_horizon];

# plot(x1_FB, y1_FB,label="player 1")
# plot!(x2_FB, y2_FB,label="player 2")
# savefig("step3.png")














# STEP 4:
# dynamics in the mind of the first player: iLQR, substituting player 2's control with player 2's varying belief
# x01 = SVector(0, 0.5, pi/2, 1,       1, 0, pi/2, 1,      θ, 1.0, 1) # the last three states are: target lane, player 2's belief mean, player 2's belief variance

struct player_dynamics4 <: ControlSystem{ΔT, 8+1+2, 4 } end
dx(cs::player_dynamics4, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), 0, 0, 
                                    0, 
                                    -1/ΔT* x[11]*π3_P_list[Int(floor(t/ΔT))+1][1:2,10]'*inv(I(2)+π3_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]*π3_P_list[Int(floor(t/ΔT))+1][1:2,10]') * ( 
                                      [u[1];u[2]] - u3_list[Int(floor(t/ΔT))+1][1:2]+ 
                                      π3_P_list[Int(floor(t/ΔT))+1][1:2,:]*(x-x3_list[Int(floor(t/ΔT))+1]) + π3_α_list[Int(floor(t/ΔT))+1][1:2] ),  # mean 
                                    - 1/ΔT* x[11]*π3_P_list[Int(floor(t/ΔT))+1][1:2,10]'*inv(I(2)+π3_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]*π3_P_list[Int(floor(t/ΔT))+1][1:2,10]') * 
                                    π3_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]                 # variance
                                    ) + SVector(0,0,0,0,0,0,     
                                    u3_list[Int(floor(t/ΔT))+1][3] - π3_P_list[Int(floor(t/ΔT))+1][3,:]'*(x-x3_list[Int(floor(t/ΔT))+1])-π3_α_list[Int(floor(t/ΔT))+1][3], 
                                    u3_list[Int(floor(t/ΔT))+1][4] - π3_P_list[Int(floor(t/ΔT))+1][4,:]'*(x-x3_list[Int(floor(t/ΔT))+1])-π3_α_list[Int(floor(t/ΔT))+1][4],
                                    0,0,0 );
dynamics4 = player_dynamics4();
costs4 = (FunctionPlayerCost((g, x, u, t) -> (10*(x[5]-x[9])^2  + (x[3]-pi/2)^2 + u[1]^2 + u[2]^2 + 50*(x[9]-x[10])^2 )), # target lane is x[10], mean of player 2 belief
          FunctionPlayerCost((g, x, u, t) -> (  4*(x[5]-x[1])^2  + (x[7]-pi/2)^2 + u[3]^2 + u[4]^2 ))) ;
g4 = GeneralGame(game_horizon, player_inputs, dynamics4, costs4);
solver4 = iLQSolver(g4, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE");
c4, x4, π4 = solve(g4, solver4, initial_state_2);

x2_list = [ x4.x[t] for t in 1:game_horizon ];
u2_list = [ x4.u[t] for t in 1:game_horizon ];
push!(x2_list, x4.x[game_horizon]);
push!(u2_list, x4.u[game_horizon]);



π2_P_list = [ π4[t].P for t in 1:game_horizon ];
push!(π2_P_list, π2_P_list[end]);
π2_α_list = [ π4[t].α for t in 1:game_horizon ];
push!(π2_α_list, π2_α_list[end]);

belief_list = [ x4.x[t][10] for t in 1:game_horizon ]
var_list = [x4.x[t][11] for t in 1:game_horizon]
# step 4 finished!



p1_costs_list = zeros(game_horizon);
for t in 1:game_horizon
  p1_costs_list[t] = costs4[1](g4, x4.x[t], x4.u[t], t)
end
sum(p1_costs_list)

p2_costs_list = zeros(game_horizon);
for t in 1:game_horizon
  p2_costs_list[t] = costs4[2](g4, x4.x[t], x4.u[t], t)
end
sum(p2_costs_list)

# step 4, costs: 47.39, 82.05
# step 6, costs: 46.90, 83.89
# step 8, costs: 47.02，86.02
# step 10, costs: 



x1_FB, y1_FB = [x4.x[i][1] for i in 1:game_horizon], [x4.x[i][2] for i in 1:game_horizon];
x2_FB, y2_FB = [x4.x[i][5] for i in 1:game_horizon], [x4.x[i][6] for i in 1:game_horizon];

scatter(x1_FB, y1_FB,markersize=6*marker_list,label="player 1")
scatter!(x2_FB, y2_FB,markersize=6*marker_list,label="player 2")
vline!([θ],label="target lane")
savefig("step4tuned50.png")

plot(1:game_horizon, belief_list,ribbon=var_list,label="belief",title="belief update",ylabel="target lane",xlabel="t")
hline!([θ],label="ground truth")
savefig("compelling belief step 4 tuned50.png")









# STEP 5:
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


x1_FB, y1_FB = [x2.x[i][1] for i in 1:game_horizon], [x2.x[i][2] for i in 1:game_horizon];
x2_FB, y2_FB = [x2.x[i][5] for i in 1:game_horizon], [x2.x[i][6] for i in 1:game_horizon];

plot(x1_FB, y1_FB,label="player 1")
plot!(x2_FB, y2_FB,label="player 2")
savefig("step5.png")














# STEP 6:
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


x1_FB, y1_FB = [x1.x[i][1] for i in 1:game_horizon], [x1.x[i][2] for i in 1:game_horizon];
x2_FB, y2_FB = [x1.x[i][5] for i in 1:game_horizon], [x1.x[i][6] for i in 1:game_horizon];

plot(x1_FB, y1_FB,label="player 1")
plot!(x2_FB, y2_FB,label="player 2")
savefig("step6.png")




# STEP 7:
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


x1_FB, y1_FB = [x2.x[i][1] for i in 1:game_horizon], [x2.x[i][2] for i in 1:game_horizon];
x2_FB, y2_FB = [x2.x[i][5] for i in 1:game_horizon], [x2.x[i][6] for i in 1:game_horizon];

plot(x1_FB, y1_FB,label="player 1")
plot!(x2_FB, y2_FB,label="player 2")
savefig("step7.png")














# STEP 8:
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


x1_FB, y1_FB = [x1.x[i][1] for i in 1:game_horizon], [x1.x[i][2] for i in 1:game_horizon];
x2_FB, y2_FB = [x1.x[i][5] for i in 1:game_horizon], [x1.x[i][6] for i in 1:game_horizon];

plot(x1_FB, y1_FB,label="player 1")
plot!(x2_FB, y2_FB,label="player 2")
savefig("step8.png")



