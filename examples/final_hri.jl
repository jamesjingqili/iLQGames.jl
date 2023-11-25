using iLQGames
import iLQGames: dx
import BenchmarkTools
using Plots
using ForwardDiff
using LinearAlgebra

using Random
using Distributions


nx, nu, ΔT, game_horizon = 5+3, 4, 0.1, 30
# weird behavior: when horizon = 40, fine, 50 or 100 blows up
L = 1;

marker_list = LinRange(1, 2, game_horizon)
time_list = ΔT:ΔT:game_horizon*ΔT
θ = 0.5; # we initialize θ to be 0.1, but the true θ can be π/4
initial_belief = 0.1
left_point_x = 1.0
left_point_y = -1.0
initial_state = SVector(
    left_point_x,
    left_point_y,
    0.2,
    left_point_x + L*cos(θ),
    left_point_y + L*sin(θ),
)
initial_state_truth = vcat(initial_state, SVector(θ, θ, 1.0))
initial_state_1 = vcat(initial_state, SVector(initial_belief, initial_belief, 1.0))
initial_state_2 = vcat(initial_state, SVector(θ, initial_belief, 1.0))

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

# notice that the dynamics should consider the Δt in the dynamics

# we define the following model parameters:
σ1 = 2;
σ2 = 2;
# ground truth:

struct player21_dynamics <: ControlSystem{ΔT, nx, nu } end
dx(cs::player21_dynamics, x, u, t) = SVector(
u[1], # x[1] is the x position of the robot
u[2], # x[2] is the y position of the robot
(u[4]-u[2])*cos(x[3]) + (u[1]-u[3])*sin(x[3]), # x[3] is the heading angle of the robot
-1/ΔT*x[4] + 1/ΔT*x[1] + 1/ΔT * L*cos(x[3]) + u[3], # x[4] is the x position of human
-1/ΔT*x[5] + 1/ΔT*x[2] + 1/ΔT * L*sin(x[3]) + u[4], # x[5] is the y position of human
0, # x[6] is the ground truth state
0, # x[7] is the mean of the  belief state
0 # x[8] is the variance of the belief state
);
dynamics = player21_dynamics();
costs = (
    FunctionPlayerCost((g, x, u, t) -> ( (x[1])^2 + (x[2])^2 +  0*(x[3] - x[6])^2 + σ1*(u[1]^2 + u[2]^2) )),  # robot cost
    FunctionPlayerCost((g, x, u, t) -> ( 10*(x[3] - x[6])^2 + σ2*((u[3]-u[1])^2 + (u[4]-u[2])^2) )) # human cost
);
player_inputs = (SVector(1,2), SVector(3,4));

g = GeneralGame(game_horizon, player_inputs, dynamics, costs);

solver = iLQSolver(g, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE"); #Stackelberg_KKT_dynamic_factorization
c_true, x_true, π_true = solve(g, solver, initial_state_truth);

π1_P_list = [ π_true[t].P for t in 1:game_horizon ]
push!(π1_P_list, π1_P_list[end])
π1_α_list = [ π_true[t].α for t in 1:game_horizon ]
push!(π1_α_list, π1_α_list[end])
u_list = [ x_true.u[t] for t in 1:game_horizon ]
push!(u_list, u_list[end])
x_list = [ x_true.x[t] for t in 1:game_horizon ]
push!(x_list, x_list[end])
# ground truth solved!



# STEP iLQR:
# dynamics in the mind of the second player: iLQR, substituting player 1's control with player 1's varying belief
x02 = vcat(initial_state, SVector(θ, initial_belief, 1.0))
struct player_dynamics2 <: ControlSystem{ΔT, nx, nu } end
dx(cs::player_dynamics2, x, u, t) = SVector(
0, # x[1] is the x position of the robot
0, # x[2] is the y position of the robot
(u_list[Int(floor(t/ΔT))+1][2] - π1_P_list[Int(floor(t/ΔT))+1][2,[1,2,3,6]]'*[x-x_list[Int(floor(t/ΔT))+1]][1][[1,2,3,7]]-π1_α_list[Int(floor(t/ΔT))+1][2]-u[4])*cos(x[3]) + 
(u_list[Int(floor(t/ΔT))+1][1] - π1_P_list[Int(floor(t/ΔT))+1][1,[1,2,3,6]]'*[x-x_list[Int(floor(t/ΔT))+1]][1][[1,2,3,7]]-π1_α_list[Int(floor(t/ΔT))+1][1]-u[3])*sin(x[3]), # x[3] is the heading angle of the robot
-1/ΔT*x[4] + 1/ΔT*x[1] + 1/ΔT * L*cos(x[3]) + u[3], # x[4] is the x position of human
-1/ΔT*x[5] + 1/ΔT*x[2] + 1/ΔT * L*sin(x[3]) + u[4], # x[5] is the y position of human
0, # x[6] is the ground truth state 
- 1/ΔT* x[end]*π1_P_list[Int(floor(t/ΔT))+1][3:4,6]'*inv(I(2)+π1_P_list[Int(floor(t/ΔT))+1][3:4,6]*x[end]*π1_P_list[Int(floor(t/ΔT))+1][3:4,6]') * ( 
    [u[3];u[4]] - u_list[Int(floor(t/ΔT))+1][3:4]+ 
    π1_P_list[Int(floor(t/ΔT))+1][3:4,:]*(x-x_list[Int(floor(t/ΔT))+1]) + π1_α_list[Int(floor(t/ΔT))+1][3:4] ),  # mean 
- 1/ΔT* x[end]*π1_P_list[Int(floor(t/ΔT))+1][3:4,6]'*inv(I(2)+π1_P_list[Int(floor(t/ΔT))+1][3:4,6]*x[end]*π1_P_list[Int(floor(t/ΔT))+1][3:4,6]') * 
π1_P_list[Int(floor(t/ΔT))+1][3:4,6]*x[end]                 # variance
) + SVector(  
u_list[Int(floor(t/ΔT))+1][1] - π1_P_list[Int(floor(t/ΔT))+1][1,[1,2,3,6]]'*[x-x_list[Int(floor(t/ΔT))+1]][1][[1,2,3,7]]-π1_α_list[Int(floor(t/ΔT))+1][1], 
u_list[Int(floor(t/ΔT))+1][2] - π1_P_list[Int(floor(t/ΔT))+1][2,[1,2,3,6]]'*[x-x_list[Int(floor(t/ΔT))+1]][1][[1,2,3,7]]-π1_α_list[Int(floor(t/ΔT))+1][2],
0,0,0,
0,0,0
);
dynamics2 = player_dynamics2();
costs2 = (
    FunctionPlayerCost((g, x, u, t) -> ( (x[1])^2 + (x[2])^2 +  0*(x[3] - x[7])^2 + (u[1])^2 + (u[2])^2 + σ1*(
        (u_list[Int(floor(t/ΔT))+1][1] - π1_P_list[Int(floor(t/ΔT))+1][1,[1,2,3,6]]'*[x-x_list[Int(floor(t/ΔT))+1]][1][[1,2,3,7]]-π1_α_list[Int(floor(t/ΔT))+1][1])^2 + 
        (u_list[Int(floor(t/ΔT))+1][2] - π1_P_list[Int(floor(t/ΔT))+1][2,[1,2,3,6]]'*[x-x_list[Int(floor(t/ΔT))+1]][1][[1,2,3,7]]-π1_α_list[Int(floor(t/ΔT))+1][2])^2) )),  # robot cost
    FunctionPlayerCost((g, x, u, t) -> ( 10*(x[3] - x[6])^2 + 10*(x[end-1] - x[end-2])^2 + σ2*((
        u[3]
        )^2 + (
        u[4]
        )^2) )) # human cost
); 
g2 = GeneralGame(game_horizon, player_inputs, dynamics2, costs2);
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

belief_list = [ x2.x[t][7] for t in 1:game_horizon ]
var_list = [x2.x[t][end] for t in 1:game_horizon]
x2_costs_player_1 = sum([costs2[1](g2, x2.x[t], x2.u[t], t*ΔT)[1] for t in 1:game_horizon])
x2_costs_player_2 = sum([costs2[2](g2, x2.x[t], x2.u[t], t*ΔT)[1] for t in 1:game_horizon])
# step iLQR finished!

x21_FB, y21_FB = [x2.x[i][1] for i in 1:game_horizon], [x2.x[i][2] for i in 1:game_horizon];
x22_FB, y22_FB = [x2.x[i][4] for i in 1:game_horizon], [x2.x[i][5] for i in 1:game_horizon];
plot(1:game_horizon, belief_list, ribbon=var_list, xlabel="time", ylabel="mean of belief target lane",label="")
hline!([θ],label="ground truth")
savefig("final_hri_belief.png")

plot(1:game_horizon, [x2_list[t][3] for t in 1:game_horizon], xlabel="time", ylabel="theta",label="")
hline!([θ],label="target")
savefig("final_hri_theta.png")

scatter(x21_FB, y21_FB,color=:black,label="robot")
scatter!(x22_FB, y22_FB,color=:red,label="human")
for t in 1:game_horizon
    if t == 1
        plot_label = "desk"
    else
        plot_label = ""
    end
    plot!([x21_FB[t], x22_FB[t]], [y21_FB[t], y22_FB[t]], color=:orange, label=plot_label)
end
savefig("final_hri_traj.png")














# passive
x01 = vcat(initial_state, SVector(θ, initial_belief, 1.0))
# STEP 0:
# dynamics in the mind of the second player: iLQGames, substituting player 1's control for player 2's belief update!
struct player_dynamics3 <: ControlSystem{ΔT, nx, nu } end
dx(cs::player_dynamics3, x, u, t) = SVector(
    u_list[Int(floor(t/ΔT))+1][1] - π1_P_list[Int(floor(t/ΔT))+1][1,:]'*(x-x_list[Int(floor(t/ΔT))+1])-π1_α_list[Int(floor(t/ΔT))+1][1], # x[1] is the x position of the robot
    u_list[Int(floor(t/ΔT))+1][2] - π1_P_list[Int(floor(t/ΔT))+1][2,:]'*(x-x_list[Int(floor(t/ΔT))+1])-π1_α_list[Int(floor(t/ΔT))+1][2], # x[2] is the y position of the robot
    (    (u_list[Int(floor(t/ΔT))+1][2] - π1_P_list[Int(floor(t/ΔT))+1][2,[1,2,3,6]]'*[x-x_list[Int(floor(t/ΔT))+1]][1][[1,2,3,7]]-π1_α_list[Int(floor(t/ΔT))+1][2])-(u_list[Int(floor(t/ΔT))+1][4] - π1_P_list[Int(floor(t/ΔT))+1][4,:]'*(x-x_list[Int(floor(t/ΔT))+1])-π1_α_list[Int(floor(t/ΔT))+1][4]))*cos(x[3]) +
        ((u_list[Int(floor(t/ΔT))+1][1] - π1_P_list[Int(floor(t/ΔT))+1][1,[1,2,3,6]]'*[x-x_list[Int(floor(t/ΔT))+1]][1][[1,2,3,7]]-π1_α_list[Int(floor(t/ΔT))+1][1])-(u_list[Int(floor(t/ΔT))+1][3] - π1_P_list[Int(floor(t/ΔT))+1][3,:]'*(x-x_list[Int(floor(t/ΔT))+1])-π1_α_list[Int(floor(t/ΔT))+1][3]))*sin(x[3]), # x[3] is the heading angle of the robot
    -1/ΔT*x[4] + 1/ΔT*x[1] + 1/ΔT * L*cos(x[3]) + (u_list[Int(floor(t/ΔT))+1][3] - π1_P_list[Int(floor(t/ΔT))+1][3,:]'*(x-x_list[Int(floor(t/ΔT))+1])-π1_α_list[Int(floor(t/ΔT))+1][3]), # x[4] is the x position of human
    -1/ΔT*x[5] + 1/ΔT*x[2] + 1/ΔT * L*sin(x[3]) + (u_list[Int(floor(t/ΔT))+1][4] - π1_P_list[Int(floor(t/ΔT))+1][4,:]'*(x-x_list[Int(floor(t/ΔT))+1])-π1_α_list[Int(floor(t/ΔT))+1][4]), # x[5] is the y position of human
    0, 
    -1/ΔT* x[end]*π1_P_list[Int(floor(t/ΔT))+1][3:4,6]'*inv(I(2)+π1_P_list[Int(floor(t/ΔT))+1][3:4,6]*x[end]*π1_P_list[Int(floor(t/ΔT))+1][3:4,6]') * ( 
        [
            u_list[Int(floor(t/ΔT))+1][3] - π1_P_list[Int(floor(t/ΔT))+1][3,:]'*(x-x_list[Int(floor(t/ΔT))+1])-π1_α_list[Int(floor(t/ΔT))+1][3];
            u_list[Int(floor(t/ΔT))+1][4] - π1_P_list[Int(floor(t/ΔT))+1][4,:]'*(x-x_list[Int(floor(t/ΔT))+1])-π1_α_list[Int(floor(t/ΔT))+1][4]
        ] - u_list[Int(floor(t/ΔT))+1][3:4]+ 
        π1_P_list[Int(floor(t/ΔT))+1][3:4,1:6]*(x-x_list[Int(floor(t/ΔT))+1])[[1,2,3,4,5,7]] + π1_α_list[Int(floor(t/ΔT))+1][3:4] ),  # mean 
    - 1/ΔT* x[end]*π1_P_list[Int(floor(t/ΔT))+1][3:4,6]'*inv(I(2)+π1_P_list[Int(floor(t/ΔT))+1][3:4,6]*x[end]*π1_P_list[Int(floor(t/ΔT))+1][3:4,6]') * 
        π1_P_list[Int(floor(t/ΔT))+1][3:4,6]*x[end]                 # variance
);
dynamics3 = player_dynamics3();

costs3 = (
    FunctionPlayerCost((g, x, u, t) -> ( (x[1])^2 + (x[2])^2 +  0*(x[3] - x[7])^2 + σ1*(u[1]^2 + u[2]^2) )),  # robot cost
    FunctionPlayerCost((g, x, u, t) -> ( 
        10*(x[3] - x[6])^2 + (u[3])^2 + (u[4])^2
        # σ2*(((u_true_list[Int(floor(t/ΔT))+1][3] - π_true_P_list[Int(floor(t/ΔT))+1][3,:]'*(x-x_true_list[Int(floor(t/ΔT))+1])-π_true_α_list[Int(floor(t/ΔT))+1][3])-u[1])^2 + ((u_true_list[Int(floor(t/ΔT))+1][4] - π_true_P_list[Int(floor(t/ΔT))+1][4,:]'*(x-x_true_list[Int(floor(t/ΔT))+1])-π_true_α_list[Int(floor(t/ΔT))+1][4])-u[2])^2) 
        )) # human cost
);
g3 = GeneralGame(game_horizon, player_inputs, dynamics3, costs3);
solver3 = iLQSolver(g3, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE");
c3, x3, π3 = solve(g3, solver3, x01);



π3_P_list = [π3[t].P for t in 1:game_horizon ];
push!(π3_P_list, π3_P_list[end]);
π3_α_list = [π3[t].α for t in 1:game_horizon ];
push!(π3_α_list, π3_α_list[end]);

u3_list = [ x3.u[t] for t in 1:game_horizon ];
push!(u3_list, u3_list[end]);
x3_list = [ x3.x[t] for t in 1:game_horizon ];
push!(x3_list, x3_list[end]);

belief_list = [ x3.x[t][7] for t in 1:game_horizon ]
var_list = [x3.x[t][end] for t in 1:game_horizon]

# eval_costs = (
#     FunctionPlayerCost((g, x, u, t) -> ( (x[1])^2 + (x[2])^2 +  0*(x[3] - x[7])^2 + σ1*(u[1]^2 + u[2]^2) )),  # robot cost
#     FunctionPlayerCost((g, x, u, t) -> ( 10*(x[3] - x[6])^2 + σ2*((u[3]-u[1])^2 + (u[4]-u[2])^2) )) # human cost
# );
# baseline_costs_player_1 = sum([eval_costs[1](g3, x3.x[t], x3.u[t], t*ΔT)[1] for t in 1:game_horizon])
# baseline_costs_player_2 = sum([eval_costs[2](g3, x3.x[t], x3.u[t], t*ΔT)[1] for t in 1:game_horizon])
# step 0 finished!




x11_FB, y11_FB = [x3.x[i][1] for i in 1:game_horizon], [x3.x[i][2] for i in 1:game_horizon];
x12_FB, y12_FB = [x3.x[i][4] for i in 1:game_horizon], [x3.x[i][5] for i in 1:game_horizon];

scatter(x11_FB, y11_FB,color=:black,label="robot")
scatter!(x12_FB, y12_FB,color=:red,label="human")
for t in 1:game_horizon
    if t == 1
        plot_label = "desk"
    else
        plot_label = ""
    end
    plot!([x11_FB[t], x12_FB[t]], [y11_FB[t], y12_FB[t]], color=:orange, label=plot_label)
end
savefig("final_hri_traj_baseline.png")

plot(1:game_horizon, [x3_list[t][3] for t in 1:game_horizon], xlabel="time", ylabel="theta",label="")
hline!([θ],label="target")
savefig("final_hri_angle_baseline.png")


plot(1:game_horizon, belief_list, ribbon=var_list, xlabel="time", ylabel="mean of belief target lane",label="")
hline!([θ],label="ground truth")
savefig("final_hri_belief_baseline.png")
