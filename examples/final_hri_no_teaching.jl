using iLQGames
import iLQGames: dx
import BenchmarkTools
using Plots
using ForwardDiff
using LinearAlgebra
# player 1 is the student, player 2 is the teacher
using Random
using Distributions
active_color = "#ff910a"
passive_color = "#828282"
complete_color = "#1c9993"
plot_path="hri_figure/"
nx, nu, ΔT, game_horizon = 5+3, 4, 0.1, 20
# weird behavior: when horizon = 40, fine, 50 or 100 blows up
L = 1;

robot_care = 4.0;
human_care = 4.0;
task_weight = 1.0;
intent_weight = 0.0;



# 



# [1, 0], check, outdated
# [1, 20], check, outdated

marker_list = LinRange(1, 2, game_horizon)
time_list = ΔT:ΔT:game_horizon*ΔT
# θ = 0.7; # we initialize θ to be 0.1, but the true θ can be π/4
θ = 0.3; # 0.3
initial_belief =0.1
initial_variance = 0.3
left_point_x =  2.0
left_point_y = -2.0
initial_state = SVector(
    left_point_x,
    left_point_y,
    0.6,
    left_point_x + L*cos(θ),
    left_point_y + L*sin(θ),
)
initial_state_truth = vcat(initial_state, SVector(θ, θ, initial_variance))
initial_state_1 = vcat(initial_state, SVector(initial_belief, initial_belief, initial_variance))
initial_state_2 = vcat(initial_state, SVector(θ, initial_belief, initial_variance))

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
(u[2]-u[4])*cos(x[3]) + (u[1]-u[3])*sin(x[3]), # x[3] is the heading angle of the robot
-1/ΔT*x[4] + 1/ΔT*x[1] + 1/ΔT * L*cos(x[3]) + u[3], # x[4] is the x position of human
-1/ΔT*x[5] + 1/ΔT*x[2] + 1/ΔT * L*sin(x[3]) + u[4], # x[5] is the y position of human
0, # x[6] is the ground truth state
0, # x[7] is the mean of the  belief state
0 # x[8] is the variance of the belief state
);
dynamics = player21_dynamics();
costs = (
    FunctionPlayerCost((g, x, u, t) -> ( (x[1])^2 + (x[2])^2 +  robot_care*(x[3] - x[6])^2 + σ1*(u[1]^2 + u[2]^2) )),  # robot cost
    FunctionPlayerCost((g, x, u, t) -> ( human_care*(x[3] - x[6])^2 + σ2*((u[3]-0*u[1])^2 + (u[4]-0*u[2])^2) )) # human cost, teacher
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

x_true_list = [ x_true.x[t] for t in 1:game_horizon ];
u_true_list = [ x_true.u[t] for t in 1:game_horizon ];
push!(x_true_list, x_true.x[game_horizon]);
push!(u_true_list, x_true.u[game_horizon]);
π_true_P_list = [ π_true[t].P for t in 1:game_horizon ];
push!(π_true_P_list, π_true_P_list[end]);
π_true_α_list = [ π_true[t].α for t in 1:game_horizon ];
push!(π_true_α_list, π_true_α_list[end]);




x11_ground_truth, y11_ground_truth = [x_true.x[i][1] for i in 1:game_horizon], [x_true.x[i][2] for i in 1:game_horizon];
x12_ground_truth, y12_ground_truth = [x_true.x[i][4] for i in 1:game_horizon], [x_true.x[i][5] for i in 1:game_horizon];

marker_alpha_list = LinRange(0.3, 1.0, game_horizon)
scatter(x11_ground_truth, y11_ground_truth,markeralpha=marker_alpha_list,color=:black,label="robot")
scatter!(x12_ground_truth, y12_ground_truth,markeralpha=marker_alpha_list,color=:red,label="human")
for t in 1:game_horizon
    if t == 1
        plot_label = "desk"
    else
        plot_label = ""
    end
    plot!([x11_ground_truth[t], x12_ground_truth[t]], [y11_ground_truth[t], y12_ground_truth[t]], linealpha = marker_alpha_list[t], color=:orange, label=plot_label)
end
savefig(plot_path*"final_hri_traj_ground_truth_no_teaching.png")

plot(0:game_horizon-1, [x_list[t][3] for t in 1:game_horizon], xlabel="time", ylabel="theta",label="")
hline!([θ],label="target")
savefig(plot_path*"final_hri_theta_ground_truth_no_teaching.png")
# ground truth solved!




function control1(x,τ)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][1]-π1_P_list[t][1,[1,2,3,6]]'*[x-x_list[t]][1][[1,2,3,7]]-π1_α_list[t][1]
end
function control2(x,τ)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][2]-π1_P_list[t][2,[1,2,3,6]]'*[x-x_list[t]][1][[1,2,3,7]]-π1_α_list[t][2]
end
function control3(x,τ)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][3]-π1_P_list[t][3,[1,2,3,6]]'*[x-x_list[t]][1][[1,2,3,6]]-π1_α_list[t][3]
end
function control4(x,τ)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][4]-π1_P_list[t][4,[2,2,3,6]]'*[x-x_list[t]][1][[1,2,3,6]]-π1_α_list[t][4]
end
# player 1 is the student, player 2 is the teacher
function player1_imagined_control(x, τ, index)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][index] - π1_P_list[t][index,[1,2,3,6]]'*[x-x_list[t]][1][[1,2,3,7]]-π1_α_list[t][index]
end
function belief_mean_update(x,u,τ)
    t= Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[end]*π1_P_list[t][3:4,6]'*inv(I(2)+π1_P_list[t][3:4,6]*x[end]*π1_P_list[t][3:4,6]') * ( 
        [u[3] - player1_imagined_control(x,τ,3);  u[4] - player1_imagined_control(x,τ,4)] )
end
function belief_variance_update(x,u,τ)
    t = Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[end]*π1_P_list[t][3:4,6]'*inv(I(2)+π1_P_list[t][3:4,6]*x[end]*π1_P_list[t][3:4,6]') * 
        π1_P_list[t][3:4,6]*x[end]
end

# function belief_mean_update(x,u,τ)
#     t= Int(floor(τ/ΔT))+1
#     return - 1/ΔT* x[end]*π1_P_list[Int(floor(t/ΔT))+1][3:4,6]'*inv(I(2)+π1_P_list[Int(floor(t/ΔT))+1][3:4,6]*x[end]*π1_P_list[Int(floor(t/ΔT))+1][3:4,6]') * ( 
#         [control3(x,t)+u[3];control4(x,t)+u[4]] - u_list[Int(floor(t/ΔT))+1][3:4] + π1_P_list[Int(floor(t/ΔT))+1][3:4,[1,2,3,6]]*(x-x_list[Int(floor(t/ΔT))+1])[[1,2,3,7]] + π1_α_list[Int(floor(t/ΔT))+1][3:4] )
# end



# STEP iLQR:
# u[3] and u[4] are the control of the human
# dynamics in the mind of the second player: iLQR, substituting player 1's control with player 1's varying belief
x02 = vcat(initial_state, SVector(θ, initial_belief, initial_variance))
struct player_dynamics2 <: ControlSystem{ΔT, nx, nu } end
dx(cs::player_dynamics2, x, u, t) = SVector(
control1(x,t), 
control2(x,t), 
(control2(x,t) - u[4])*cos(x[3]) + (control1(x,t) -u[3])*sin(x[3]), # x[3] is the heading angle of the robot
-1/ΔT*x[4] + 1/ΔT*x[1] + 1/ΔT * L*cos(x[3])  + u[3], # x[4] is the x position of human, teacher
-1/ΔT*x[5] + 1/ΔT*x[2] + 1/ΔT * L*sin(x[3])  + u[4], # x[5] is the y position of human
0, # x[6] is the ground truth state 
belief_mean_update(x,u,t),
belief_variance_update(x,u,t)
# - 1/ΔT* x[end]*π1_P_list[Int(floor(t/ΔT))+1][3:4,6]'*inv(I(2)+π1_P_list[Int(floor(t/ΔT))+1][3:4,6]*x[end]*π1_P_list[Int(floor(t/ΔT))+1][3:4,6]') * 
# π1_P_list[Int(floor(t/ΔT))+1][3:4,6]*x[end]                 # variance
);
dynamics2 = player_dynamics2();
costs2 = (
    FunctionPlayerCost((g, x, u, t) -> ( (x[1])^2 + (x[2])^2 +  robot_care*(x[3] - x[7])^2 + σ1*(u[1])^2 + (u[2])^2)), #+ σ1*(
        # (u_list[Int(floor(t/ΔT))+1][1] - π1_P_list[Int(floor(t/ΔT))+1][1,[1,2,3,6]]'*[x-x_list[Int(floor(t/ΔT))+1]][1][[1,2,3,7]]-π1_α_list[Int(floor(t/ΔT))+1][1])^2 + 
        # (u_list[Int(floor(t/ΔT))+1][2] - π1_P_list[Int(floor(t/ΔT))+1][2,[1,2,3,6]]'*[x-x_list[Int(floor(t/ΔT))+1]][1][[1,2,3,7]]-π1_α_list[Int(floor(t/ΔT))+1][2])^2) )),  # robot cost
    FunctionPlayerCost((g, x, u, t) -> ( task_weight*(human_care*(x[3] - x[6])^2+ σ2*((u[3] - 0*control1(x,t))^2 + (u[4]-0*control2(x,t))^2)) + intent_weight*(x[end-1] - x[end-2])^2  )) # human cost
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

active_belief_list = [ x2.x[t][7] for t in 1:game_horizon ]
active_var_list = [x2.x[t][end] for t in 1:game_horizon]
x2_costs_player_1 = sum([costs2[1](g2, x2.x[t], x2.u[t], t*ΔT)[1] for t in 1:game_horizon])
x2_costs_player_2 = sum([costs2[2](g2, x2.x[t], x2.u[t], t*ΔT)[1] for t in 1:game_horizon])
# step iLQR finished!

x21_FB, y21_FB = [x2.x[i][1] for i in 1:game_horizon], [x2.x[i][2] for i in 1:game_horizon];
x22_FB, y22_FB = [x2.x[i][4] for i in 1:game_horizon], [x2.x[i][5] for i in 1:game_horizon];
plot(0:game_horizon-1, active_belief_list, ribbon=active_var_list, xlabel="time", ylabel="mean of belief target lane",label="")
hline!([θ],label="ground truth")
savefig(plot_path*"final_hri_belief_no_teaching.png")

plot(0:game_horizon-1, [x2_list[t][3] for t in 1:game_horizon], xlabel="time", ylabel="theta",label="")
hline!([θ],label="target")
savefig(plot_path*"final_hri_theta_no_teaching.png")

scatter(x21_FB, y21_FB,color=:black,markeralpha=marker_alpha_list,label="robot")
scatter!(x22_FB, y22_FB,color=:red,markeralpha=marker_alpha_list,label="human")
for t in 1:game_horizon
    if t == 1
        plot_label = "desk"
    else
        plot_label = ""
    end
    plot!([x21_FB[t], x22_FB[t]], [y21_FB[t], y22_FB[t]], linealpha = marker_alpha_list[t], color=:orange, label=plot_label)
end
savefig(plot_path*"final_hri_traj_no_teaching.png")













function passive_belief_mean_update(x,τ)
    t= Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[end]*π1_P_list[t][3:4,6]'*inv(I(2)+π1_P_list[t][3:4,6]*x[end]*π1_P_list[t][3:4,6]') * ( 
        [control3(x,τ) - player1_imagined_control(x,τ,3);  control4(x,τ) - player1_imagined_control(x,τ,4)] )
end
function passive_belief_variance_update(x,τ)
    t = Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[end]*π1_P_list[t][3:4,6]'*inv(I(2)+π1_P_list[t][3:4,6]*x[end]*π1_P_list[t][3:4,6]') * 
        π1_P_list[t][3:4,6]*x[end]
end


# passive inference:
# the main idea is the following: 
# player 1 is robot and player 2 is human. Human knows the true θ, but robot doesn't know the true θ
# robot uses human's action to infer what's the true θ

x01 = vcat(initial_state, SVector(θ, initial_belief, initial_variance))
# STEP 0:
# dynamics in the mind of the second player: iLQGames, substituting player 1's control for player 2's belief update!
struct player_dynamics3 <: ControlSystem{ΔT, nx, nu } end
dx(cs::player_dynamics3, x, u, t) = SVector(
    control1(x,t), # x[1] is the x position of the robot
    control2(x,t), # x[2] is the y position of the robot
    (control2(x,t)-control4(x,t))*cos(x[3]) + (control1(x,t)-control3(x,t))*sin(x[3]), # x[3] is the heading angle of the robot
    -1/ΔT*x[4] + 1/ΔT*x[1] + 1/ΔT * L*cos(x[3]) + control3(x,t), # x[4] is the x position of human
    -1/ΔT*x[5] + 1/ΔT*x[2] + 1/ΔT * L*sin(x[3]) + control4(x,t), # x[5] is the y position of human
    0,
    passive_belief_mean_update(x,t),
    # passive_belief_variance_update(x,t) 
    # -1/ΔT* x[end]*π_true_P_list[Int(floor(t/ΔT))+1][3:4,6]'*inv(I(2)+π_true_P_list[Int(floor(t/ΔT))+1][3:4,6]*x[end]*π_true_P_list[Int(floor(t/ΔT))+1][3:4,6]') * ( 
    #     [
    #         u_true_list[Int(floor(t/ΔT))+1][3] - π_true_P_list[Int(floor(t/ΔT))+1][3,:]'*(x-x_true_list[Int(floor(t/ΔT))+1])-π_true_α_list[Int(floor(t/ΔT))+1][3];
    #         u_true_list[Int(floor(t/ΔT))+1][4] - π_true_P_list[Int(floor(t/ΔT))+1][4,:]'*(x-x_true_list[Int(floor(t/ΔT))+1])-π_true_α_list[Int(floor(t/ΔT))+1][4]
    #     ] - u_true_list[Int(floor(t/ΔT))+1][3:4]+ 
    #     π_true_P_list[Int(floor(t/ΔT))+1][3:4,1:6]*(x-x_true_list[Int(floor(t/ΔT))+1])[[1,2,3,4,5,7]] + π_true_α_list[Int(floor(t/ΔT))+1][3:4] ),  # mean 
    - 1/ΔT* x[end]*π_true_P_list[Int(floor(t/ΔT))+1][3:4,6]'*inv(I(2)+π_true_P_list[Int(floor(t/ΔT))+1][3:4,6]*x[end]*π_true_P_list[Int(floor(t/ΔT))+1][3:4,6]') * 
        π_true_P_list[Int(floor(t/ΔT))+1][3:4,6]*x[end]                 # variance
);
dynamics3 = player_dynamics3();

costs3 = (
    FunctionPlayerCost((g, x, u, t) -> ( (x[1])^2 + (x[2])^2 +  robot_care*(x[3] - x[7])^2 + σ1*(u[1]^2 + u[2]^2) )),  # robot cost
    FunctionPlayerCost((g, x, u, t) -> ( 
        human_care*(x[3] - x[6])^2 + (u[3])^2 + (u[4])^2
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

passive_belief_list = [ x3.x[t][7] for t in 1:game_horizon ]
passive_var_list = [x3.x[t][end] for t in 1:game_horizon]

# eval_costs = (
#     FunctionPlayerCost((g, x, u, t) -> ( (x[1])^2 + (x[2])^2 +  0*(x[3] - x[7])^2 + σ1*(u[1]^2 + u[2]^2) )),  # robot cost
#     FunctionPlayerCost((g, x, u, t) -> ( 10*(x[3] - x[6])^2 + σ2*((u[3]-u[1])^2 + (u[4]-u[2])^2) )) # human cost
# );
# baseline_costs_player_1 = sum([eval_costs[1](g3, x3.x[t], x3.u[t], t*ΔT)[1] for t in 1:game_horizon])
# baseline_costs_player_2 = sum([eval_costs[2](g3, x3.x[t], x3.u[t], t*ΔT)[1] for t in 1:game_horizon])
# step 0 finished!




x11_FB, y11_FB = [x3.x[i][1] for i in 1:game_horizon], [x3.x[i][2] for i in 1:game_horizon];
x12_FB, y12_FB = [x3.x[i][4] for i in 1:game_horizon], [x3.x[i][5] for i in 1:game_horizon];

scatter(x11_FB, y11_FB,color=:black,markeralpha=marker_alpha_list,label="robot")
scatter!(x12_FB, y12_FB,color=:red,markeralpha=marker_alpha_list,label="human")
for t in 1:game_horizon
    if t == 1
        plot_label = "desk"
    else
        plot_label = ""
    end
    plot!([x11_FB[t], x12_FB[t]], [y11_FB[t], y12_FB[t]], linealpha = marker_alpha_list[t], color=:orange, label=plot_label)
end
savefig(plot_path*"final_hri_traj_baseline_no_teaching.png")

plot(0:game_horizon-1, [x3_list[t][3] for t in 1:game_horizon], xlabel="time", ylabel="theta",label="")
hline!([θ],label="target")
savefig(plot_path*"final_hri_angle_baseline_no_teaching.png")


plot(0:game_horizon-1, passive_belief_list, ribbon=passive_var_list, xlabel="time", ylabel="mean of belief target lane",label="")
hline!([θ],label="ground truth")
savefig(plot_path*"final_hri_belief_baseline_no_teaching.png")


plot(0:game_horizon-1, [x3_list[t][3] for t in 1:game_horizon], xlabel="time", ylabel="theta",label="passive teaching",legend=:bottomright,color=passive_color, linewidth=4 )
plot!(0:game_horizon-1, [x2_list[t][3] for t in 1:game_horizon], xlabel="time", ylabel="theta",label="active teaching", color=active_color, linewidth=4)
plot!(0:game_horizon-1, [x_list[t][3] for t in 1:game_horizon], xlabel="time", ylabel="theta",label="complete information", color=complete_color, linewidth=4)
plot!([0,game_horizon-1],[θ,θ],label="target",color=:black)
savefig(plot_path*"final_hri_angle_comparison_no_teaching.png")


plot(0:game_horizon-1, passive_belief_list, ribbon=passive_var_list, xlabel="time", ylabel="mean of the belief",label="passive teaching",legend=:bottomright, color=passive_color, linewidth=4)
plot!(0:game_horizon-1, active_belief_list, ribbon=active_var_list, xlabel="time", ylabel="mean of the belief",label="active teaching", color=active_color, linewidth=4)
# hline!([θ],label="ground truth",color=:black)
plot!([0,game_horizon-1],[θ,θ],label="target",color=:black)
savefig(plot_path*"final_hri_belief_comparison_no_teaching.png")


# Arxiv version:
# struct player_dynamics3 <: ControlSystem{ΔT, nx, nu } end
# dx(cs::player_dynamics3, x, u, t) = SVector(
#     (u_true_list[Int(floor(t/ΔT))+1][1] - π_true_P_list[Int(floor(t/ΔT))+1][1,[1,2,3,6]]'*(x-x_true_list[Int(floor(t/ΔT))+1])[[1,2,3,7]]-π_true_α_list[Int(floor(t/ΔT))+1][1]), # x[1] is the x position of the robot
#     (u_true_list[Int(floor(t/ΔT))+1][2] - π_true_P_list[Int(floor(t/ΔT))+1][2,[1,2,3,6]]'*(x-x_true_list[Int(floor(t/ΔT))+1])[[1,2,3,7]]-π_true_α_list[Int(floor(t/ΔT))+1][2]), # x[2] is the y position of the robot
#     ((u_true_list[Int(floor(t/ΔT))+1][2] - π_true_P_list[Int(floor(t/ΔT))+1][2,[1,2,3,6]]'*(x-x_true_list[Int(floor(t/ΔT))+1])[[1,2,3,7]]-π_true_α_list[Int(floor(t/ΔT))+1][2])-(u_true_list[Int(floor(t/ΔT))+1][4] - π_true_P_list[Int(floor(t/ΔT))+1][4,:]'*(x-x_true_list[Int(floor(t/ΔT))+1])-π_true_α_list[Int(floor(t/ΔT))+1][4]))*cos(x[3]) + 
#         ((u_true_list[Int(floor(t/ΔT))+1][1] - π_true_P_list[Int(floor(t/ΔT))+1][1,[1,2,3,6]]'*(x-x_true_list[Int(floor(t/ΔT))+1])[[1,2,3,7]]-π_true_α_list[Int(floor(t/ΔT))+1][1])-(u_true_list[Int(floor(t/ΔT))+1][3] - π_true_P_list[Int(floor(t/ΔT))+1][3,:]'*(x-x_true_list[Int(floor(t/ΔT))+1])-π_true_α_list[Int(floor(t/ΔT))+1][3]))*sin(x[3]), # x[3] is the heading angle of the robot
#     -1/ΔT*x[4] + 1/ΔT*x[1] + 1/ΔT * L*cos(x[3]) + (u_true_list[Int(floor(t/ΔT))+1][3] - π_true_P_list[Int(floor(t/ΔT))+1][3,:]'*(x-x_true_list[Int(floor(t/ΔT))+1])-π_true_α_list[Int(floor(t/ΔT))+1][3]), # x[4] is the x position of human
#     -1/ΔT*x[5] + 1/ΔT*x[2] + 1/ΔT * L*sin(x[3]) + (u_true_list[Int(floor(t/ΔT))+1][4] - π_true_P_list[Int(floor(t/ΔT))+1][4,:]'*(x-x_true_list[Int(floor(t/ΔT))+1])-π_true_α_list[Int(floor(t/ΔT))+1][4]), # x[5] is the y position of human
#     0, 
#     -1/ΔT* x[end]*π_true_P_list[Int(floor(t/ΔT))+1][3:4,6]'*inv(I(2)+π_true_P_list[Int(floor(t/ΔT))+1][3:4,6]*x[end]*π_true_P_list[Int(floor(t/ΔT))+1][3:4,6]') * ( 
#         [
#             u_true_list[Int(floor(t/ΔT))+1][3] - π_true_P_list[Int(floor(t/ΔT))+1][3,:]'*(x-x_true_list[Int(floor(t/ΔT))+1])-π_true_α_list[Int(floor(t/ΔT))+1][3];
#             u_true_list[Int(floor(t/ΔT))+1][4] - π_true_P_list[Int(floor(t/ΔT))+1][4,:]'*(x-x_true_list[Int(floor(t/ΔT))+1])-π_true_α_list[Int(floor(t/ΔT))+1][4]
#         ] - u_true_list[Int(floor(t/ΔT))+1][3:4]+ 
#         π_true_P_list[Int(floor(t/ΔT))+1][3:4,1:6]*(x-x_true_list[Int(floor(t/ΔT))+1])[[1,2,3,4,5,7]] + π_true_α_list[Int(floor(t/ΔT))+1][3:4] ),  # mean 
#     - 1/ΔT* x[end]*π_true_P_list[Int(floor(t/ΔT))+1][3:4,6]'*inv(I(2)+π_true_P_list[Int(floor(t/ΔT))+1][3:4,6]*x[end]*π_true_P_list[Int(floor(t/ΔT))+1][3:4,6]') * 
#         π_true_P_list[Int(floor(t/ΔT))+1][3:4,6]*x[end]                 # variance
# );
# dynamics3 = player_dynamics3();

# costs3 = (
#     FunctionPlayerCost((g, x, u, t) -> ( (x[1])^2 + (x[2])^2 +  10*(x[3] - x[7])^2 + σ1*(u[1]^2 + u[2]^2) )),  # robot cost
#     FunctionPlayerCost((g, x, u, t) -> ( 
#         10*(x[3] - x[6])^2 + (u[3])^2 + (u[4])^2
#         # σ2*(((u_true_list[Int(floor(t/ΔT))+1][3] - π_true_P_list[Int(floor(t/ΔT))+1][3,:]'*(x-x_true_list[Int(floor(t/ΔT))+1])-π_true_α_list[Int(floor(t/ΔT))+1][3])-u[1])^2 + ((u_true_list[Int(floor(t/ΔT))+1][4] - π_true_P_list[Int(floor(t/ΔT))+1][4,:]'*(x-x_true_list[Int(floor(t/ΔT))+1])-π_true_α_list[Int(floor(t/ΔT))+1][4])-u[2])^2) 
#         )) # human cost
# );
# g3 = GeneralGame(game_horizon, player_inputs, dynamics3, costs3);
# solver3 = iLQSolver(g3, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE");
# c3, x3, π3 = solve(g3, solver3, x01);



function passive_control(x,t)
    return [control1(x,t); control2(x,t); control3(x,t); control4(x,t)]
end
function active_control(x,u,t)
    return [control1(x,t); control2(x,t); u[3]; u[4]]
end


# evaluate the task costs:


list_complete_costs_player_1 = [costs[1](g, x_true.x[t], x_true.u[t], t*ΔT)[1] for t in 1:game_horizon]
list_complete_costs_player_2 = [costs[2](g, x_true.x[t], x_true.u[t], t*ΔT)[1] for t in 1:game_horizon]
list_passive_costs_player_1 = [costs[1](g3, x3.x[t], passive_control(x3.x[t],t*ΔT), t*ΔT)[1] for t in 1:game_horizon]
list_passive_costs_player_2 = [costs[2](g3, x3.x[t], passive_control(x3.x[t],t*ΔT), t*ΔT)[1] for t in 1:game_horizon]
list_active_costs_player_1 = [costs[1](g2, x2.x[t], active_control(x2.x[t],x2.u[t],t*ΔT), t*ΔT)[1] for t in 1:game_horizon]
list_active_costs_player_2 = [costs[2](g2, x2.x[t], active_control(x2.x[t],x2.u[t],t*ΔT), t*ΔT)[1] for t in 1:game_horizon]

complete_costs_player_1 = sum(list_complete_costs_player_1)
complete_costs_player_2 = sum(list_complete_costs_player_2)
passive_costs_player_1 = sum(list_passive_costs_player_1)
passive_costs_player_2 = sum(list_passive_costs_player_2)

active_costs_player_1 = sum(list_active_costs_player_1)
active_costs_player_2 = sum(list_active_costs_player_2)




player1_c = [complete_costs_player_1, active_costs_player_1, passive_costs_player_1]
player2_c = [complete_costs_player_2, active_costs_player_2, passive_costs_player_2]




line_width = 2




# the below is for the paper:

# plot the cost of each player versus time:
plot(0:game_horizon-1, list_active_costs_player_2,linewidth = line_width,color=active_color, xlabel="time",label="",legend=:topright)
plot!(0:game_horizon-1, list_passive_costs_player_2,linewidth = line_width, color = passive_color, xlabel="time", ylabel="cost",label="")
plot!(0:game_horizon-1, list_complete_costs_player_2,linewidth = line_width, color = complete_color, xlabel="time", ylabel="cost",label="")
plot!(size = (400,200), grid = false)
# savefig(plot_path*"hri_costs_teacher_no_teaching_1_0.pdf")
savefig(plot_path*"hri_costs_teacher_$θ.pdf")




# plot the belief of uncertain agent versus time:
plot(0:game_horizon-1, active_belief_list, ribbon=active_var_list, xlabel="time", ylabel="belief",label="", color=active_color, linewidth=line_width)
plot!(0:game_horizon-1, passive_belief_list, ribbon=passive_var_list, xlabel="time", ylabel="belief",label="",legend=:bottomright, color=passive_color, linewidth=line_width)
plot!([0,game_horizon-1],[θ, θ],label="",color=complete_color, linewidth=line_width)
plot!(size = (400,200), grid = false)
# savefig(plot_path*"hri_belief_teacher_no_teaching_1_0.pdf")
savefig(plot_path*"hri_belief_teacher_$θ.pdf")





# plot the theta of uncertain agent versus time:
plot(0:game_horizon-1, [x2_list[t][3] for t in 1:game_horizon], xlabel="time", ylabel="angle",label="active", color=active_color, linewidth=line_width) # active teaching
plot!(0:game_horizon-1, [x3_list[t][3] for t in 1:game_horizon], xlabel="time", ylabel="angle",label="passive",legend=:bottomright, color=passive_color, linewidth=line_width) # passive teaching
plot!(0:game_horizon-1, [x_list[t][3] for t in 1:game_horizon], xlabel="time", ylabel="angle",label="oracal",legend=:bottomright, color=complete_color, linewidth=line_width) # complete information
plot!([0,game_horizon-1],[θ,θ],label="θ*",color=:black, linewidth=line_width) # θ*
plot!(size = (400,200), legend=:topright, grid = false)
# savefig(plot_path*"hri_theta_teacher_no_teaching_1_0.pdf")
savefig(plot_path*"hri_theta_teacher_$θ.pdf")



