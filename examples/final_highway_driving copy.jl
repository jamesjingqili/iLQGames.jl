
using iLQGames
import iLQGames: dx
import BenchmarkTools
using Plots
using ForwardDiff
using LinearAlgebra
using Statistics
# using FileIO
# using Images
plot_path = "highway_figure/"
nx, nu, ΔT, game_horizon = 12+1+2*2, 6, 0.1, 40
# weird behavior: when horizon = 40, fine, 50 or 100 blows up



# # Load the image
# marker_path = "/mnt/2312dd0d-0348-41b0-8596-b79a2cc21f6b/Dropbox/ppt_download_logos/"
# teacher_marker = load(marker_path*"noun-top-view-car-5105420.png")
# student1_marker = load(marker_path*"noun-top-view-car-5105420-3B6BF9.png")
# student2_marker = load(marker_path*"noun-top-view-car-5105420-FF001C.png")
# # Convert the image to an array
# teacher_marker_array = channelview(teacher_marker)
# student1_marker_array = channelview(student1_marker)
# student2_marker_array = channelview(student2_marker)
# # # Use the array as a marker in the scatter function
# scatter([1, 2, 3], [4, 5, 6], marker_z = teacher_marker)



marker_alpha_list = LinRange(0.3, 1.0, game_horizon)
time_list = ΔT:ΔT:game_horizon*ΔT
θ = 0.3; # NOTE!!! change to 0.3 for the paper trajectory, but 0.5 for the regret!!!
initial_belief = 1;
initial_state = SVector(
    0., 1.5, pi/2, 1.5, 
    1., 0.5, pi/2, 1.5, 
    1., 0., pi/2, 1.5
)
initial_state_truth = vcat(initial_state, SVector(θ,        θ,1,θ,1))
active_color = "#ff910a"
passive_color = "#828282"
complete_color = "#1c9993"
# ground truth:
struct ThreeUnicycles <: ControlSystem{ΔT, nx, nu } end
dx(cs::ThreeUnicycles, x, u, t) = SVector(
x[4]cos(x[3]), 
x[4]sin(x[3]), 
u[1], 
u[2], 
x[8]cos(x[7]), 
x[8]sin(x[7]), 
u[3], 
u[4], 
x[12]cos(x[11]), 
x[12]sin(x[11]), 
u[5], 
u[6],
0,  # parameter of player 1, invisible to player 2
0,  # mean of player 2
0,  # variance player 2
0,  # mean of player 3
0   # variance player 3
)
dynamics = ThreeUnicycles()
costs = (FunctionPlayerCost((g, x, u, t) -> (10*(x[5]-x[13])^2 + 10*(x[9]-x[13])^2  + (x[3]-pi/2)^2 + u[1]^2 + u[2]^2 )), 
        FunctionPlayerCost((g, x, u, t) -> (  4*(x[5] - x[1])^2  + (x[7]-pi/2)^2 + u[3]^2 + u[4]^2 )),
        FunctionPlayerCost((g, x, u, t) -> (  4*(x[9] - x[5])^2  + (x[11]-pi/2)^2 + u[5]^2 + u[6]^2 ))
        ) 

player_inputs = (SVector(1,2), SVector(3,4), SVector(5,6))
g = GeneralGame(game_horizon, player_inputs, dynamics, costs)
solver = iLQSolver(g, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")
c, x, π = solve(g, solver, initial_state_truth)



x1_FB, y1_FB = [x.x[i][1] for i in 1:game_horizon], [x.x[i][2] for i in 1:game_horizon];
x2_FB, y2_FB = [x.x[i][5] for i in 1:game_horizon], [x.x[i][6] for i in 1:game_horizon];
x3_FB, y3_FB = [x.x[i][9] for i in 1:game_horizon], [x.x[i][10] for i in 1:game_horizon];

π_P_list = [ π[t].P for t in 1:game_horizon ]
push!(π_P_list, π_P_list[end])
π_α_list = [ π[t].α for t in 1:game_horizon ]
push!(π_α_list, π_α_list[end])
u_list = [ x.u[t] for t in 1:game_horizon ]
push!(u_list, u_list[end])
x_list = [ x.x[t] for t in 1:game_horizon ]
push!(x_list, x_list[end])



scatter(x1_FB, y1_FB,color=:black,markeralpha=marker_alpha_list,label="teacher", size=(200,600),legend=:bottomleft)
scatter!(x2_FB, y2_FB,color=:blue,markeralpha=marker_alpha_list,label="student 1")
scatter!(x3_FB, y3_FB,color=:red,markeralpha=marker_alpha_list,label="student 2")
xlims!(-0.3, 1.3)
vline!([θ],label="target",linestyle=:dash,color=:black)
vline!([-0.2], label="lane boundary", color=:black, linewidth = 4)
vline!([1.2], label="", color=:black, linewidth = 4)
savefig(plot_path*"final_highway_traj_complete.pdf")






player2_info_list = [1,2,3,4,5,6,7,8,9,10,11,12,14] # mean is 14, var is 15
player3_info_list = [1,2,3,4,5,6,7,8,9,10,11,12,16] # mean is 16, var is 17


function control1(x,τ)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][1]-π_P_list[t][1,1:13]'*[x-x_list[t]][1][1:13]-π_α_list[t][1]
end
function control2(x,τ)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][2]-π_P_list[t][2,1:13]'*[x-x_list[t]][1][1:13]-π_α_list[t][2]
end
function control3(x,τ)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][3]-π_P_list[t][3,1:13]'*[x-x_list[t]][1][player2_info_list]-π_α_list[t][3]
end
function control4(x,τ)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][4]-π_P_list[t][4,1:13]'*[x-x_list[t]][1][player2_info_list]-π_α_list[t][4]
end
function control5(x,τ)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][5]-π_P_list[t][5,1:13]'*[x-x_list[t]][1][player3_info_list]-π_α_list[t][5]
end
function control6(x,τ)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][6]-π_P_list[t][6,1:13]'*[x-x_list[t]][1][player3_info_list]-π_α_list[t][6]
end




# player 1 is the teacher, players 2 and 3 are students
function player2_imagined_control(x, τ, index)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][index] - π_P_list[t][index,1:13]'*[x-x_list[t]][1][player2_info_list]-π_α_list[t][index]
end
function player3_imagined_control(x, τ, index)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][index] - π_P_list[t][index,1:13]'*[x-x_list[t]][1][player3_info_list]-π_α_list[t][index]
end

# belief update:
function player2_belief_mean_update(x,u,τ)
    t= Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[15]*π_P_list[t][1:2,13]'*inv(I(2)+π_P_list[t][1:2,13]*x[15]*π_P_list[t][1:2,13]') * ( 
        [control1(x,τ)+u[1] - player2_imagined_control(x,τ,1);  control2(x,τ)+u[2] - player2_imagined_control(x,τ,2)] )
end
function player2_belief_variance_update(x,u,τ)
    t = Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[15]*π_P_list[t][1:2,13]'*inv(I(2)+π_P_list[t][1:2,13]*x[15]*π_P_list[t][1:2,13]') * π_P_list[t][1:2,13]*x[15]
end
function player3_belief_mean_update(x,u,τ)
    t= Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[17]*π_P_list[t][1:2,13]'*inv(I(2)+π_P_list[t][1:2,13]*x[17]*π_P_list[t][1:2,13]') * ( 
        [control1(x,τ)+u[1] - player3_imagined_control(x,τ,1);  control2(x,τ)+u[2] - player3_imagined_control(x,τ,2)] )
end
function player3_belief_variance_update(x,u,τ)
    t = Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[17]*π_P_list[t][1:2,13]'*inv(I(2)+π_P_list[t][1:2,13]*x[17]*π_P_list[t][1:2,13]') * π_P_list[t][1:2,13]*x[17]
end



# active teaching
x02 = vcat(initial_state, SVector(θ, initial_belief,θ, initial_belief, 1.0))
struct ThreeUnicycles2 <: ControlSystem{ΔT, nx, nu } end
dx(cs::ThreeUnicycles2, x, u, t) = SVector(
x[4]cos(x[3]), 
x[4]sin(x[3]), 
control1(x,t) + u[1], 
control2(x,t) + u[2], 
x[8]cos(x[7]), 
x[8]sin(x[7]), 
control3(x,t), 
control4(x,t),
x[12]cos(x[11]), 
x[12]sin(x[11]), 
control5(x,t), 
control6(x,t),
0,  # parameter of player 1, invisible to students
player2_belief_mean_update(x,u,t),
player2_belief_variance_update(x,u,t),
player3_belief_mean_update(x,u,t),
player3_belief_variance_update(x,u,t),
)
dynamics2 = ThreeUnicycles2();
costs2 = (
    FunctionPlayerCost((g, x, u, t) -> (5*(x[14]-x[13])^2 + 5*(x[16]-x[13])^2 + 
    10*(x[5]-x[13])^2 + 10*(x[9]-x[13])^2  + (x[3]-pi/2)^2 + u[1]^2 + u[2]^2 )), # teaching cost!
    FunctionPlayerCost((g, x, u, t) -> (  4*(x[5] - x[1])^2  + (x[7]-pi/2)^2 + u[3]^2 + u[4]^2 )),
    FunctionPlayerCost((g, x, u, t) -> (  4*(x[9] - x[5])^2  + (x[11]-pi/2)^2 + u[5]^2 + u[6]^2 ))
) 
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


x21_FB, y21_FB = [x2.x[i][1] for i in 1:game_horizon], [x2.x[i][2] for i in 1:game_horizon];
x22_FB, y22_FB = [x2.x[i][5] for i in 1:game_horizon], [x2.x[i][6] for i in 1:game_horizon];
x23_FB, y23_FB = [x2.x[i][9] for i in 1:game_horizon], [x2.x[i][10] for i in 1:game_horizon];

player2_active_belief_list = [ x2.x[t][14] for t in 1:game_horizon ]
player2_active_var_list = [x2.x[t][15] for t in 1:game_horizon]
player3_active_belief_list = [ x2.x[t][16] for t in 1:game_horizon ]
player3_active_var_list = [x2.x[t][17] for t in 1:game_horizon]

# x2_costs_player_1 = sum([costs2[1](g2, x2.x[t], x2.u[t], t*ΔT)[1] for t in 1:game_horizon])
# x2_costs_player_2 = sum([costs2[2](g2, x2.x[t], x2.u[t], t*ΔT)[1] for t in 1:game_horizon])

scatter(x21_FB, y21_FB,color=:black,markeralpha=marker_alpha_list,label="",size=(200,600))
scatter!(x22_FB, y22_FB,color=:blue,markeralpha=marker_alpha_list,label="")
scatter!(x23_FB, y23_FB,color=:red,markeralpha=marker_alpha_list,label="")
vline!([θ],label="",linestyle=:dash, color=:black)
xlims!(-0.3, 1.3)
vline!([-0.2], label="", color=:black, linewidth = 4)
vline!([1.2], label="", color=:black, linewidth = 4)
savefig(plot_path*"final_highway_traj.pdf")




























# passive teaching
function player2_passive_belief_mean_update(x,τ)
    t= Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[15]*π_P_list[t][1:2,13]'*inv(I(2)+π_P_list[t][1:2,13]*x[15]*π_P_list[t][1:2,13]') * ( 
        [control1(x,τ) - player2_imagined_control(x,τ,1);  control2(x,τ) - player2_imagined_control(x,τ,2)] )
end
function player2_passive_belief_variance_update(x,τ)
    t = Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[15]*π_P_list[t][1:2,13]'*inv(I(2)+π_P_list[t][1:2,13]*x[15]*π_P_list[t][1:2,13]') * π_P_list[t][1:2,13]*x[15]
end
function player3_passive_belief_mean_update(x,τ)
    t= Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[17]*π_P_list[t][1:2,13]'*inv(I(2)+π_P_list[t][1:2,13]*x[17]*π_P_list[t][1:2,13]') * ( 
        [control1(x,τ) - player3_imagined_control(x,τ,1);  control2(x,τ) - player3_imagined_control(x,τ,2)] )
end
function player3_passive_belief_variance_update(x,τ)
    t = Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[17]*π_P_list[t][1:2,13]'*inv(I(2)+π_P_list[t][1:2,13]*x[17]*π_P_list[t][1:2,13]') * π_P_list[t][1:2,13]*x[17]
end
struct ThreeUnicycles3 <: ControlSystem{ΔT, nx, nu } end
dx(cs::ThreeUnicycles3, x, u, t) = SVector(
x[4]cos(x[3]), 
x[4]sin(x[3]), 
control1(x,t), 
control2(x,t), 
x[8]cos(x[7]), 
x[8]sin(x[7]), 
control3(x,t), 
control4(x,t),
x[12]cos(x[11]), 
x[12]sin(x[11]), 
control5(x,t), 
control6(x,t),
0,  # parameter of player 1, invisible to students
player2_passive_belief_mean_update(x,t),
player2_passive_belief_variance_update(x,t),
player3_passive_belief_mean_update(x,t),
player3_passive_belief_variance_update(x,t),
)
dynamics3 = ThreeUnicycles3();
costs3 = (
    FunctionPlayerCost((g, x, u, t) -> (20*(x[14]-x[13])^2 + 20*(x[16]-x[13])^2 + 
        10*(x[5]-x[13])^2 + 10*(x[9]-x[13])^2  + (x[3]-pi/2)^2 + u[1]^2 + u[2]^2 )),
    FunctionPlayerCost((g, x, u, t) -> (  4*(x[5] - x[1])^2  + (x[7]-pi/2)^2 + u[3]^2 + u[4]^2 )),
    FunctionPlayerCost((g, x, u, t) -> (  4*(x[9] - x[5])^2  + (x[11]-pi/2)^2 + u[5]^2 + u[6]^2 ))
) 
g3 = GeneralGame(game_horizon, player_inputs, dynamics3, costs3);
solver3 = iLQSolver(g3, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")
c3, x3, π3 = solve(g3, solver3, x02);








# data processing:
π3_P_list = [π3[t].P for t in 1:game_horizon ];
push!(π3_P_list, π3_P_list[end]);
π3_α_list = [π3[t].α for t in 1:game_horizon ];
push!(π3_α_list, π3_α_list[end]);

u3_list = [ x3.u[t] for t in 1:game_horizon ];
push!(u3_list, u3_list[end]);
x3_list = [ x3.x[t] for t in 1:game_horizon ];
push!(x3_list, x3_list[end]);

player2_passive_belief_list = [ x3.x[t][14] for t in 1:game_horizon ]
player2_passive_var_list = [x3.x[t][15] for t in 1:game_horizon]
player3_passive_belief_list = [ x3.x[t][16] for t in 1:game_horizon ]
player3_passive_var_list = [x3.x[t][17] for t in 1:game_horizon]


x31_FB, y31_FB = [x3.x[i][1] for i in 1:game_horizon], [x3.x[i][2] for i in 1:game_horizon];
x32_FB, y32_FB = [x3.x[i][5] for i in 1:game_horizon], [x3.x[i][6] for i in 1:game_horizon];
x33_FB, y33_FB = [x3.x[i][9] for i in 1:game_horizon], [x3.x[i][10] for i in 1:game_horizon];


scatter(x31_FB, y31_FB,color=:black,  markeralpha=marker_alpha_list,label="",size=(200,600))
scatter!(x32_FB, y32_FB,color=:blue,  markeralpha=marker_alpha_list,label="")
scatter!(x33_FB, y33_FB,color=:red,   markeralpha=marker_alpha_list,label="")
vline!([θ],label="", linestyle=:dash,color=:black)
vline!([-0.2], label="", color=:black, linewidth = 4)
vline!([1.2], label="", color=:black, linewidth = 4)
xlims!(-0.3, 1.3)
savefig(plot_path*"final_highway_traj_baseline.pdf")


# plot(0:game_horizon-1, [x3_list[t][14] for t in 1:game_horizon], xlabel="time", ylabel="theta",label="")
# hline!([θ],label="target")
# savefig(plot_path*"final_highway_angle_baseline.png")


# plot(0:game_horizon-1, player2_passive_belief_list, ribbon=player2_passive_var_list, xlabel="time", ylabel="mean of belief target lane",label="student 1")
# plot!(0:game_horizon-1, player3_passive_belief_list, ribbon=player3_passive_var_list, xlabel="time", ylabel="mean of belief target lane",label="student 2")
# hline!([θ],label="ground truth")
# savefig(plot_path*"final_highway_belief_baseline.png")


plot(0:game_horizon-1, [x3_list[t][5] for t in 1:game_horizon], xlabel="time", ylabel="theta",label="passive teaching, student 1",legend=:topright, color=passive_color)
plot!(0:game_horizon-1, [x2_list[t][5] for t in 1:game_horizon], xlabel="time", ylabel="theta",label="active teaching, student 1", color = active_color)
plot!(0:game_horizon-1, [x_list[t][5] for t in 1:game_horizon], xlabel="time", ylabel="theta",label="complete information, student 1", color = complete_color)
plot!([0,game_horizon-1],[θ,θ],label="target",color=:black)
savefig(plot_path*"final_highway_x_comparison_student1.png")

plot(0:game_horizon-1, [x3_list[t][9] for t in 1:game_horizon], xlabel="time", ylabel="theta",label="passive teaching, student 2",legend=:topright, color=passive_color)
plot!(0:game_horizon-1, [x2_list[t][9] for t in 1:game_horizon], xlabel="time", ylabel="theta",label="active teaching, student 2", color = active_color)
plot!(0:game_horizon-1, [x_list[t][9] for t in 1:game_horizon], xlabel="time", ylabel="theta",label="complete information, student 2", color = complete_color)
plot!([0,game_horizon-1],[θ,θ],label="target",color=:black)
savefig(plot_path*"final_highway_x_comparison_student2.png")


plot(0:game_horizon-1, player2_passive_belief_list, ribbon=player2_passive_var_list, xlabel="time", ylabel="mean of the belief",label="student 1, passive teaching",legend=:topright, color=passive_color, linewidth=2)
plot!(0:game_horizon-1, player2_active_belief_list, ribbon=player2_active_var_list, xlabel="time", ylabel="mean of the belief",label="student 1, active teaching", color = active_color, linewidth=2)
plot!([0,game_horizon-1],[θ,θ],label="target",color=:black)
savefig(plot_path*"final_hri_belief_comparison_student1.pdf")

plot(0:game_horizon-1, player3_passive_belief_list, ribbon=player3_passive_var_list, xlabel="time", ylabel="mean of the belief",label="student 2, passive teaching",legend=:topright, color=passive_color, linewidth=2)
plot!(0:game_horizon-1, player3_active_belief_list, ribbon=player3_active_var_list, xlabel="time", ylabel="mean of the belief",label="student 2, active teaching", color = active_color, linewidth=2)
plot!([0,game_horizon-1],[θ,θ],label="target",color=:black)
savefig(plot_path*"final_hri_belief_comparison_student2.pdf")


plot(0:game_horizon-1, player3_passive_belief_list, ribbon=player3_passive_var_list,linestyle=:dash, 
    xlabel="time", ylabel="mean of the belief",label="student 2, passive teaching",legend=:topright, color=passive_color, linewidth=2)
plot!(0:game_horizon-1, player3_active_belief_list, ribbon=player3_active_var_list, 
    xlabel="time", ylabel="mean of the belief",label="student 2, active teaching", color = active_color, linewidth=2)
plot!(0:game_horizon-1, player2_passive_belief_list, ribbon=player2_passive_var_list,linestyle=:dash, 
    xlabel="time", ylabel="mean of the belief",label="student 1, passive teaching",legend=:topright, color=passive_color, linewidth=2)
plot!(0:game_horizon-1, player2_active_belief_list, ribbon=player2_active_var_list, 
    xlabel="time", ylabel="mean of the belief",label="student 1, active teaching", color = active_color, linewidth=2)
plot!([0,game_horizon-1],[θ,θ],label="target",color=:black)
savefig(plot_path*"final_hri_belief_comparison_students.png")






# evaluate task costs:
complete_costs_player_1 = sum([costs[1](g, x.x[t], x.u[t], t*ΔT)[1] for t in 1:game_horizon])
complete_costs_player_2 = sum([costs[2](g, x.x[t], x.u[t], t*ΔT)[1] for t in 1:game_horizon])
complete_costs_player_3 = sum([costs[3](g, x.x[t], x.u[t], t*ΔT)[1] for t in 1:game_horizon])

function passive_control(x,t)
    return [control1(x,t); control2(x,t); control3(x,t); control4(x,t); control5(x,t); control6(x,t)]
end
function active_control(x,u,t)
    return [control1(x,t)+u[1]; control2(x,t)+u[2]; control3(x,t); control4(x,t); control5(x,t); control6(x,t)]
end
passive_costs_player_1 = sum([costs[1](g3, x3.x[t], passive_control(x3.x[t],t*ΔT), t*ΔT)[1] for t in 1:game_horizon])
passive_costs_player_2 = sum([costs[2](g3, x3.x[t], passive_control(x3.x[t],t*ΔT), t*ΔT)[1] for t in 1:game_horizon])
passive_costs_player_3 = sum([costs[3](g3, x3.x[t], passive_control(x3.x[t],t*ΔT), t*ΔT)[1] for t in 1:game_horizon])

active_costs_player_1 = sum([costs[1](g2, x2.x[t], active_control(x2.x[t],x2.u[t],t*ΔT), t*ΔT)[1] for t in 1:game_horizon])
active_costs_player_2 = sum([costs[2](g2, x2.x[t], active_control(x2.x[t],x2.u[t],t*ΔT), t*ΔT)[1] for t in 1:game_horizon])
active_costs_player_3 = sum([costs[3](g2, x2.x[t], active_control(x2.x[t],x2.u[t],t*ΔT), t*ΔT)[1] for t in 1:game_horizon])



player1_c = [complete_costs_player_1, active_costs_player_1, passive_costs_player_1]
player2_c = [complete_costs_player_2, active_costs_player_2, passive_costs_player_2]
player3_c = [complete_costs_player_3, active_costs_player_3, passive_costs_player_3]




















