using iLQGames
import iLQGames: dx
import BenchmarkTools
using Plots
using ForwardDiff
using LinearAlgebra

nx, nu, ΔT, game_horizon = 8+1+2, 4, 0.1, 40
# weird behavior: when horizon = 40, fine, 50 or 100 blows up


θ = 0.2;

# ground truth:
struct player_dynamics <: ControlSystem{ΔT, 8+1+2, 4 } end
dx(cs::player_dynamics, x, u, t) = SVector(x[4]cos(x[3]), 
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
dynamics = player_dynamics()
costs = (FunctionPlayerCost((g, x, u, t) -> (10*(x[5]-x[9])^2  + (x[3]-pi/2)^2 + u[1]^2 + u[2]^2 )), # target lane is x[10], mean of player 2 belief
            FunctionPlayerCost((g, x, u, t) -> (  4*(x[5] - x[1])^2   +(x[7]-pi/2)^2 + u[3]^2 + u[4]^2 ))) 

player_inputs = (SVector(1,2), SVector(3,4))
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
x0 = SVector(0, 0.5, pi/2, 1,       
                1, 0, pi/2, 1,  
                θ, 1, 1) # the last two states are: player 2's belief mean, player 2's belief variance
solver = iLQSolver(g, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")#Stackelberg_KKT_dynamic_factorization
c, x, π = solve(g, solver, x0)

x1_FB, y1_FB = [x.x[i][1] for i in 1:game_horizon], [x.x[i][2] for i in 1:game_horizon];
x2_FB, y2_FB = [x.x[i][5] for i in 1:game_horizon], [x.x[i][6] for i in 1:game_horizon];


π_P_list = [ π[t].P for t in 1:game_horizon ]
push!(π_P_list, π_P_list[end])
π_α_list = [ π[t].α for t in 1:game_horizon ]
push!(π_α_list, π_α_list[end])
u_list = [ x.u[t] for t in 1:game_horizon ]
push!(u_list, u_list[end])
x_list = [ x.x[t] for t in 1:game_horizon ]
push!(x_list, x_list[end])


plot(x1_FB, y1_FB,label="player 1")
plot!(x2_FB, y2_FB,label="player 2")
vline!([0.2],label="target lane" )
savefig("step0.png")








# STEP 1:
struct player_dynamics1 <: ControlSystem{ΔT, 8+1+2, 4 } end
dx(cs::player_dynamics1, x, u, t) = SVector(x[4]cos(x[3]), 
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
dynamics1 = player_dynamics1()
costs1 = (FunctionPlayerCost((g, x, u, t) -> (10*(x[5]-x[10])^2  + (x[3]-pi/2)^2 + u[1]^2 + u[2]^2 )), # target lane is x[10], mean of player 2 belief
            FunctionPlayerCost((g, x, u, t) -> (  4*(x[5] - x[1])^2   +(x[7]-pi/2)^2 + u[3]^2 + u[4]^2 ))) 

player_inputs = (SVector(1,2), SVector(3,4))
g1 = GeneralGame(game_horizon, player_inputs, dynamics1, costs1)
x01 = SVector(0, 0.5, pi/2, 1,       
                1, 0, pi/2, 1,  
                0, 1, 1) # the last two states are: player 2's belief mean, player 2's belief variance
solver1 = iLQSolver(g1, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")#Stackelberg_KKT_dynamic_factorization
c1, x1, π1 = solve(g1, solver1, x01)

x11_FB, y11_FB = [x1.x[i][1] for i in 1:game_horizon], [x1.x[i][2] for i in 1:game_horizon];
x12_FB, y12_FB = [x1.x[i][5] for i in 1:game_horizon], [x1.x[i][6] for i in 1:game_horizon];


π1_P_list = [ π1[t].P for t in 1:game_horizon ]
push!(π1_P_list, π1_P_list[end])
π1_α_list = [ π1[t].α for t in 1:game_horizon ]
push!(π1_α_list, π1_α_list[end])
u1_list = [ x1.u[t] for t in 1:game_horizon ]
push!(u1_list, u1_list[end])
x1_list = [ x1.x[t] for t in 1:game_horizon ]
push!(x1_list, x1_list[end])


plot(x11_FB, y11_FB,label="player 1")
plot!(x12_FB, y12_FB,label="player 2")
vline!([0.2],label="target lane" )

savefig("step1.png")





# Problems: 1. u_t = \hat{u}_t + K_t(x - \hat{x}_t) + α_t, deep RL policy?
#           2. b(0) = 0.1, not working, now works.
#           3. b(0) = 0.2, not working, if 1/ΔT, now works.

















# TODO: substituting the above controller to player 1's mind for computing iLQR
# STEP 2:
# dynamics in the mind of the first player: iLQR, substituting player 2's control with player 2's varying belief
x02 = SVector(0, 0.5, pi/2, 1,       1, 0, pi/2, 1,      θ, 1.0, 1) # the last three states are: target lane, player 2's belief mean, player 2's belief variance

struct player_dynamics2 <: ControlSystem{ΔT, 8+1+2, 4 } end
dx(cs::player_dynamics2, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), 
                                    0, 
                                    0, 
                                    0, 
                                    -1/ΔT* x[11]*π1_P_list[Int(floor(t/ΔT))+1][1:2,10]'*inv(I(2)+π1_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]*π1_P_list[Int(floor(t/ΔT))+1][1:2,10]') * ( 
                                        [u[1];u[2]] - u1_list[Int(floor(t/ΔT))+1][1:2]+ 
                                        π1_P_list[Int(floor(t/ΔT))+1][1:2,:]*(x-x1_list[Int(floor(t/ΔT))+1]) + π1_α_list[Int(floor(t/ΔT))+1][1:2] ),  # mean 
                                    - 1/ΔT* x[11]*π1_P_list[Int(floor(t/ΔT))+1][1:2,10]'*inv(I(2)+π1_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]*π1_P_list[Int(floor(t/ΔT))+1][1:2,10]') * 
                                    π1_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]                 # variance
                                    ) + SVector(0,0,0,0,0,0,     
                                    u1_list[Int(floor(t/ΔT))+1][3] - π1_P_list[Int(floor(t/ΔT))+1][3,:]'*(x-x1_list[Int(floor(t/ΔT))+1])-π1_α_list[Int(floor(t/ΔT))+1][3], 
                                    u1_list[Int(floor(t/ΔT))+1][4] - π1_P_list[Int(floor(t/ΔT))+1][4,:]'*(x-x1_list[Int(floor(t/ΔT))+1])-π1_α_list[Int(floor(t/ΔT))+1][4],
                                    0,0,0 );
dynamics2 = player_dynamics2();
costs2 = (FunctionPlayerCost((g, x, u, t) -> (10*(x[5]-x[9])^2  + (x[3]-pi/2)^2 + u[1]^2 + u[2]^2 )), # target lane is x[10], mean of player 2 belief
            FunctionPlayerCost((g, x, u, t) -> (  4*(x[5]-x[1])^2  + (x[7]-pi/2)^2 + u[3]^2 + u[4]^2 ))); 
g2 = GeneralGame(game_horizon, player_inputs, dynamics2, costs1);
solver2 = iLQSolver(g2, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")
c2, x2, π2 = solve(g2, solver2, x02);





x2_list = [ x2.x[t] for t in 1:game_horizon ];
u2_list = [ x2.u[t] for t in 1:game_horizon ];
push!(x2_list, x2.x[game_horizon]);
push!(u2_list, x2.u[game_horizon]);



π2_P_list = [ π2[t].P for t in 1:game_horizon ];
push!(π2_P_list, π2_P_list[end]);
π2_α_list = [ π2[t].α for t in 1:game_horizon ];
push!(π2_α_list, π2_α_list[end]);

belief_list = [ x2.x[t][10] for t in 1:game_horizon ]


x21_FB, y21_FB = [x2.x[i][1] for i in 1:game_horizon], [x2.x[i][2] for i in 1:game_horizon];
x22_FB, y22_FB = [x2.x[i][5] for i in 1:game_horizon], [x2.x[i][6] for i in 1:game_horizon];

# plot(x21_FB, y21_FB,label="player 1")
# plot!(x22_FB, y22_FB,label="player 2")
# savefig("step2.png")
















# TODO: stepsize: α = 0.1.  new_policy = old_policy + α * (new_policy - old_policy)



# STEP 3:
# dynamics in the mind of the second player: iLQGames, substituting player 1's control for player 2's belief update!
struct player_dynamics3 <: ControlSystem{ΔT, 8+1+2, 4 } end
dx(cs::player_dynamics3, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), 0*u[1], 0*u[2], 
                                    x[8]cos(x[7]), x[8]sin(x[7]), u[3], u[4],
                                    0, 
                                    -1/ΔT* x[11]*π2_P_list[Int(floor(t/ΔT))+1][1:2,10]'*inv(I(2)+π2_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]*π2_P_list[Int(floor(t/ΔT))+1][1:2,10]') * ( 
                                        [
                                            u2_list[Int(floor(t/ΔT))+1][1]- 
                                                π2_P_list[Int(floor(t/ΔT))+1][1,:]'*(x-x2_list[Int(floor(t/ΔT))+1]) - π2_α_list[Int(floor(t/ΔT))+1][1];
                                            u2_list[Int(floor(t/ΔT))+1][2]- 
                                                π2_P_list[Int(floor(t/ΔT))+1][2,:]'*(x-x2_list[Int(floor(t/ΔT))+1]) - π2_α_list[Int(floor(t/ΔT))+1][2]
                                        ] - u2_list[Int(floor(t/ΔT))+1][1:2]+ 
                                        π2_P_list[Int(floor(t/ΔT))+1][1:2,:]*(x-x2_list[Int(floor(t/ΔT))+1]) + π2_α_list[Int(floor(t/ΔT))+1][1:2] ),  # mean 
                                    - 1/ΔT* x[11]*π2_P_list[Int(floor(t/ΔT))+1][1:2,10]'*inv(I(2)+π2_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]*π2_P_list[Int(floor(t/ΔT))+1][1:2,10]') * 
                                    π2_P_list[Int(floor(t/ΔT))+1][1:2,10]*x[11]                 # variance
                                    ) + SVector(0,0,
                                    u2_list[Int(floor(t/ΔT))+1][1]- 
                                            π2_P_list[Int(floor(t/ΔT))+1][1,:]'*(x-x2_list[Int(floor(t/ΔT))+1]) - π2_α_list[Int(floor(t/ΔT))+1][1],
                                    u2_list[Int(floor(t/ΔT))+1][2]- 
                                            π2_P_list[Int(floor(t/ΔT))+1][2,:]'*(x-x2_list[Int(floor(t/ΔT))+1]) - π2_α_list[Int(floor(t/ΔT))+1][2],
                                    0,0,0,0,
                                    0,0,0);
dynamics3 = player_dynamics3();

costs3 = (FunctionPlayerCost((g, x, u, t) -> (10*(x[5]-x[10])^2 + 2*(x[4]-1)^2  + u[1]^2 + u[2]^2 )),
            FunctionPlayerCost((g, x, u, t) -> (  4*(x[5] - x[1])^2  +2*(x[8]-1)^2  + u[3]^2 + u[4]^2 )));
g3 = GeneralGame(game_horizon, player_inputs, dynamics3, costs3);
solver3 = iLQSolver(g3, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE");
c3, x3, π3 = solve(g3, solver3, x01);

step_size = 1;

π3_P_list = [step_size*π3[t].P for t in 1:game_horizon ];
π3_P_list = π3_P_list + (1-step_size)*π1_P_list[1:game_horizon];
push!(π3_P_list, π3_P_list[end]);
π3_α_list = [step_size*π3[t].α for t in 1:game_horizon ];
π3_α_list = π3_α_list + (1-step_size)*π1_α_list[1:game_horizon];
push!(π3_α_list, π3_α_list[end]);


u3_list = [step_size*x3.u[t] for t in 1:game_horizon ];
u3_list = u3_list + (1-step_size)*u1_list[1:game_horizon];
push!(u3_list, u3_list[end]);
x3_list = [step_size*x3.x[t] for t in 1:game_horizon ];
x3_list = x3_list + (1-step_size)*x1_list[1:game_horizon];
push!(x3_list, x3_list[end]);

π1_P_list = deepcopy(π3_P_list);
π1_α_list = deepcopy(π3_α_list);
u1_list = deepcopy(u3_list);
x1_list = deepcopy(x3_list);


belief_list = [ x3.x[t][10] for t in 1:game_horizon ]




# x1_FB, y1_FB = [x3.x[i][1] for i in 1:game_horizon], [x3.x[i][2] for i in 1:game_horizon];
# x2_FB, y2_FB = [x3.x[i][5] for i in 1:game_horizon], [x3.x[i][6] for i in 1:game_horizon];

# plot(x1_FB, y1_FB,label="player 1")
# plot!(x2_FB, y2_FB,label="player 2")

# savefig("step3.png")

# savefig("step5.png")

# savefig("step7.png")

# savefig("step9.png")

# savefig("step11.png")

# savefig("step13.png")

# savefig("step15.png")









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
costs4 = (FunctionPlayerCost((g, x, u, t) -> (10*(x[5]-x[9])^2  + (x[3]-pi/2)^2 + u[1]^2 + u[2]^2 )), # target lane is x[10], mean of player 2 belief
            FunctionPlayerCost((g, x, u, t) -> (  4*(x[5]-x[1])^2  + (x[7]-pi/2)^2 + u[3]^2 + u[4]^2 ))) ;
g4 = GeneralGame(game_horizon, player_inputs, dynamics4, costs4);
solver4 = iLQSolver(g4, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE");
c4, x4, π4 = solve(g4, solver4, x02);

x4_list = [ step_size*x4.x[t] for t in 1:game_horizon ];
x4_list = x4_list + (1-step_size)*x2_list[1:end-1];
u4_list = [ step_size*x4.u[t] for t in 1:game_horizon ];
u4_list = u4_list + (1-step_size)*u2_list[1:end-1];
push!(x4_list, x4_list[game_horizon]);
push!(u4_list, u4_list[game_horizon]);



π4_P_list = [ step_size*π4[t].P for t in 1:game_horizon ];
π4_P_list = π4_P_list + (1-step_size)*π2_P_list[1:end-1];
push!(π4_P_list, π4_P_list[end]);
π4_α_list = [ step_size*π4[t].α for t in 1:game_horizon ];
π4_α_list = π4_α_list + (1-step_size)*π2_α_list[1:end-1];
push!(π4_α_list, π4_α_list[end]);

π2_P_list = deepcopy(π4_P_list);
π2_α_list = deepcopy(π4_α_list);
x2_list = deepcopy(x4_list);
u2_list = deepcopy(u4_list);

belief_list = [ x4.x[t][10] for t in 1:game_horizon ]
plot(1:game_horizon, belief_list, xlabel="t",ylabel="mean of belief", label="")
savefig("mean_active_inference.png")

x1_FB, y1_FB = [x4.x[i][1] for i in 1:game_horizon], [x4.x[i][2] for i in 1:game_horizon];
x2_FB, y2_FB = [x4.x[i][5] for i in 1:game_horizon], [x4.x[i][6] for i in 1:game_horizon];

plot(x1_FB, y1_FB,label="player 1", xlabel="x", ylabel="y")
plot!(x2_FB, y2_FB,label="player 2")
vline!([0.2],label="target lane")
savefig("active_inference.png")



savefig("step4.png")

# conclusion: step_size = 1/2, can stabilize the progress. However, the policy in step 3 is different from step 4. How to justify this?


savefig("step6.png")

savefig("step8.png")

savefig("step10.png")

savefig("step12.png")

savefig("step14.png")

savefig("step16.png")

