# continue with final_hri.jl

function control1_policy(x,τ,π_P_list,π_α_list,u_list,x_list)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][1]-π_P_list[t][1,[1,2,3,6]]'*[x-x_list[t]][1][[1,2,3,7]]-π_α_list[t][1]
end
function control2_policy(x,τ,π_P_list,π_α_list,u_list,x_list)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][2]-π_P_list[t][2,[1,2,3,6]]'*[x-x_list[t]][1][[1,2,3,7]]-π_α_list[t][2]
end
function control3_policy(x,τ,π_P_list,π_α_list,u_list,x_list)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][3]-π_P_list[t][3,[1,2,3,6]]'*[x-x_list[t]][1][[1,2,3,6]]-π_α_list[t][3]
end
function control4_policy(x,τ,π_P_list,π_α_list,u_list,x_list)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][4]-π_P_list[t][4,[1,2,3,6]]'*[x-x_list[t]][1][[1,2,3,6]]-π_α_list[t][4]
end
# randomize experiments:
# x0_lower_limit = vcat(initial_state - 0.2*rand(12,), SVector(θ, 0.0,θ, 0.0, 1.0))
num_theta = 10
num_samples = 10
θ_samples = range(0.1,stop=0.9,length=num_theta)

x0_samples = [[vcat(initial_state + 0.2*rand(nx-3,) - 0.1*ones(nx-3,), SVector(θ_samples[j], 0.5+0.5*(rand(1,)-ones(1,))[1], 1.0)) for i in 1:num_samples] for j in 1:num_theta]



Log_p1_costs = []
Log_p2_costs = []
Log_x = []
Log_π = []
Log_x2 = []
Log_π2 = []
Log_x3 = []
Log_π3 = []




key_info_list = [1,2,3,6]
player1_info_list = [1,2,3,7]

# player 2 is the teacher, player 1 is student
function player1_imagined_control_policy(x, τ, index, π_P_list, π_α_list, u_list, x_list)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][index] - π_P_list[t][index,key_info_list]'*[x-x_list[t]][1][player1_info_list]-π_α_list[t][index]
end

# belief update:
function belief_mean_update_policy(x,u,τ,π_P_list,π_α_list,u_list,x_list)
    t= Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[8]*π_P_list[t][3:4,6]'*inv(I(2)+π_P_list[t][3:4,6]*x[8]*π_P_list[t][3:4,6]') * ( 
        [u[3] - player1_imagined_control_policy(x,τ,3,π_P_list,π_α_list,u_list,x_list);  
        u[4] - player1_imagined_control_policy(x,τ,4,π_P_list, π_α_list,u_list,x_list)] )
end

function belief_variance_update_policy(x,u,τ,π_P_list,π_α_list,u_list,x_list)
    t = Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[8]*π_P_list[t][3:4,6]'*inv(I(2)+π_P_list[t][3:4,6]*x[8]*π_P_list[t][3:4,6]') * π_P_list[t][3:4,6]*x[8]
end


# passive teaching related functions:
function passive_belief_mean_update_policy(x,τ,π_P_list,π_α_list,u_list,x_list)
    t= Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[8]*π_P_list[t][3:4,6]'*inv(I(2)+π_P_list[t][3:4,6]*x[8]*π_P_list[t][3:4,6]') * ( 
        [control3_policy(x,τ,π_P_list,π_α_list,u_list,x_list) - player1_imagined_control_policy(x,τ,3,π_P_list,π_α_list,u_list,x_list);  
        control4_policy(x,τ,π_P_list,π_α_list,u_list,x_list) - player1_imagined_control_policy(x,τ,4,π_P_list,π_α_list,u_list,x_list)] )
end
function passive_belief_variance_update_policy(x,τ,π_P_list,π_α_list,u_list,x_list)
    t = Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[8]*π_P_list[t][3:4,6]'*inv(I(2)+π_P_list[t][3:4,6]*x[8]*π_P_list[t][3:4,6]') * π_P_list[t][3:4,6]*x[8]
end



# policy evaluation:
function passive_control_policy(x,t,π_P_list,π_α_list,u_list,x_list)
    return [
        control1_policy(x,t,π_P_list,π_α_list,u_list,x_list); 
        control2_policy(x,t,π_P_list,π_α_list,u_list,x_list); 
        control3_policy(x,t,π_P_list,π_α_list,u_list,x_list); 
        control4_policy(x,t,π_P_list,π_α_list,u_list,x_list)]
end
function active_control_policy(x,u,t,π_P_list,π_α_list,u_list,x_list)
    return [
        control1_policy(x,t,π_P_list,π_α_list,u_list,x_list); 
        control2_policy(x,t,π_P_list,π_α_list,u_list,x_list); 
        u[3]; 
        u[4];]
end



for outer_iter in 1:num_theta
    player1_costs = zeros(3,num_samples) # 1 is complete, 2 is active, 3 is passsive
    player2_costs = zeros(3,num_samples) # 1 is complete, 2 is active, 3 is passsive
    log_x = []
    log_π = []
    log_x2 = []
    log_π2 = []
    log_x3 = []
    log_π3 = []
    for iter in 1:num_samples
        # ground truth:
        # global π_P_list_tmp, π_α_list_tmp, u_list_tmp, x_list_tmp
        c_tmp, x_tmp, π_tmp = solve(g, solver, x0_samples[outer_iter][iter])
        push!(log_x, x_tmp)
        push!(log_π, π_tmp)
        
        π_P_list_tmp = [ π_tmp[t].P for t in 1:game_horizon ]
        push!(π_P_list_tmp, π_P_list_tmp[end])
        π_α_list_tmp = [ π_tmp[t].α for t in 1:game_horizon ]
        push!(π_α_list_tmp, π_α_list_tmp[end])
        u_list_tmp = [ x_tmp.u[t] for t in 1:game_horizon ]
        push!(u_list_tmp, u_list_tmp[end])
        x_list_tmp = [ x_tmp.x[t] for t in 1:game_horizon ]
        push!(x_list_tmp, x_list_tmp[end])
        
        # active:
        dx(cs::player_dynamics2, x, u, t) = SVector(
            control1_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), 
            control2_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
            (control2_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp) -u[4])*cos(x[3]) +
                (control1_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp) -u[3])*sin(x[3]), # x[3] is the heading angle of the robot
            -1/ΔT*x[4] + 1/ΔT*x[1] + 1/ΔT * L*cos(x[3])  + u[3], # x[4] is the x position of human
            -1/ΔT*x[5] + 1/ΔT*x[2] + 1/ΔT * L*sin(x[3])  + u[4], # x[5] is the y position of human
            0, # x[6] is the ground truth state 
            belief_mean_update_policy(x,u,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
            belief_variance_update_policy(x,u,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp)
        );
        dynamics2_policy = player_dynamics2();
        g2_policy = GeneralGame(game_horizon, player_inputs, dynamics2_policy, costs2);
        solver2_policy = iLQSolver(g2_policy, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")

        c2_tmp, x2_tmp, π2_tmp = solve(g2_policy, solver2_policy, x0_samples[outer_iter][iter]);
        push!(log_x2, x2_tmp)
        push!(log_π2, π2_tmp)
        
        # passive:
        dx(cs::player_dynamics3, x, u, t) = SVector(
            control1_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), # x[1] is the x position of the robot
            control2_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), # x[2] is the y position of the robot
            (control2_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp)-control4_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp))*cos(x[3]) + 
                (control1_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp)-control3_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp))*sin(x[3]), # x[3] is the heading angle of the robot
            -1/ΔT*x[4] + 1/ΔT*x[1] + 1/ΔT * L*cos(x[3]) + control3_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), # x[4] is the x position of human
            -1/ΔT*x[5] + 1/ΔT*x[2] + 1/ΔT * L*sin(x[3]) + control4_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), # x[5] is the y position of human
            0,
            passive_belief_mean_update_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
            passive_belief_variance_update_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp)
        );
        dynamics3_policy = player_dynamics3();
        g3_policy = GeneralGame(game_horizon, player_inputs, dynamics3_policy, costs3);
        solver3_policy = iLQSolver(g3_policy, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")
        c3_tmp, x3_tmp, π3_tmp = solve(g3_policy, solver3_policy, x0_samples[outer_iter][iter]);
        push!(log_x3, x3_tmp)
        push!(log_π3, π3_tmp)
        
        # record costs:
        

        complete_costs_player_1_tmp = sum([costs[1](g, x_tmp.x[t], x_tmp.u[t], t*ΔT)[1] for t in 1:game_horizon])
        complete_costs_player_2_tmp = sum([costs[2](g, x_tmp.x[t], x_tmp.u[t], t*ΔT)[1] for t in 1:game_horizon])

        active_costs_player_1_tmp = sum([costs[1](g2, x2_tmp.x[t], active_control_policy(x2_tmp.x[t],x2.u[t],t*ΔT,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), t*ΔT)[1] for t in 1:game_horizon])
        active_costs_player_2_tmp = sum([costs[2](g2, x2_tmp.x[t], active_control_policy(x2_tmp.x[t],x2.u[t],t*ΔT,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), t*ΔT)[1] for t in 1:game_horizon])

        passive_costs_player_1_tmp = sum([costs[1](g3, x3_tmp.x[t], passive_control_policy(x3_tmp.x[t],t*ΔT,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), t*ΔT)[1] for t in 1:game_horizon])
        passive_costs_player_2_tmp = sum([costs[2](g3, x3_tmp.x[t], passive_control_policy(x3_tmp.x[t],t*ΔT,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), t*ΔT)[1] for t in 1:game_horizon])

        player1_costs[:,iter] = [complete_costs_player_1_tmp, active_costs_player_1_tmp, passive_costs_player_1_tmp]
        player2_costs[:,iter] = [complete_costs_player_2_tmp, active_costs_player_2_tmp, passive_costs_player_2_tmp]
    end
    push!(Log_p1_costs, player1_costs)
    push!(Log_p2_costs, player2_costs)
    push!(Log_x, log_x)
    push!(Log_π, log_π)
    push!(Log_x2, log_x2)
    push!(Log_π2, log_π2)
    push!(Log_x3, log_x3)
    push!(Log_π3, log_π3)
end

p1_active_regret = [Log_p1_costs[ii][2,:] .- Log_p1_costs[ii][1,:] for ii in 1:num_theta]
p1_passive_regret = [Log_p1_costs[ii][3,:] .- Log_p1_costs[ii][1,:] for ii in 1:num_theta]

p1_mean_active_regret = [mean(p1_active_regret[ii]) for ii in 1:num_theta]
p1_var_active_regret = [std(p1_active_regret[ii]) for ii in 1:num_theta]
p1_mean_passive_regret = [mean(p1_passive_regret[ii]) for ii in 1:num_theta]
p1_var_passive_regret = [std(p1_passive_regret[ii]) for ii in 1:num_theta]


plot(θ_samples, p1_mean_passive_regret, ribbon=p1_var_passive_regret, xlabel="target θ", 
    ylabel="regret of player 1",label="passive teaching",legend=:bottomright, color = passive_color,
    linewidth=2)
plot!(θ_samples, p1_mean_active_regret, ribbon=p1_var_passive_regret, xlabel="target θ", 
    ylabel="regret of player 1",label="active teaching",legend=:bottomright, color = active_color,
    linewidth=2)
savefig(plot_path*"final_hri_costs_p1_comparison_no_teaching.pdf")


p2_active_regret = [Log_p2_costs[ii][2,:] .- Log_p2_costs[ii][1,:] for ii in 1:num_theta]
p2_passive_regret = [Log_p2_costs[ii][3,:] .- Log_p2_costs[ii][1,:] for ii in 1:num_theta]

p2_mean_active_regret = [mean(p2_active_regret[ii]) for ii in 1:num_theta]
p2_var_active_regret = [std(p2_active_regret[ii]) for ii in 1:num_theta]
p2_mean_passive_regret = [mean(p2_passive_regret[ii]) for ii in 1:num_theta]
p2_var_passive_regret = [std(p2_passive_regret[ii]) for ii in 1:num_theta]


plot(θ_samples, p2_mean_passive_regret, ribbon=p2_var_passive_regret, xlabel="target θ", 
    ylabel="regret of player 2",label="passive teaching",legend=:bottomright, color = passive_color, linewidth=2)
plot!(θ_samples, p2_mean_active_regret, ribbon=p2_var_passive_regret, xlabel="target θ", 
    ylabel="regret of player 2",label="active teaching",legend=:bottomright, color = active_color, linewidth=2)
savefig(plot_path*"final_hri_costs_p2_comparison_no_teaching.pdf")

