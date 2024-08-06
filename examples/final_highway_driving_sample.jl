# continue with final_highway_driving.jl with θ = 0.3

# NOTE! run theta = 0.5, and then run the following experiments!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 11/27



using Plots.PlotMeasures














# We have  the following set up!
# x0_samples = [[vcat(initial_state_tmp + 0.4*rand(12,) - 0.4*ones(12,), SVector(θ_samples[j], 0.9+0.1*(rand(1,)-ones(1,))[1],1.0, 0.9+0.1*(rand(1,)-ones(1,))[1], 1.0)) 
#     for i in 1:num_samples] for j in 1:num_theta]
# θ_samples = range(0.3, 0.7, length=num_theta)
# initial_state_tmp=SVector(
#     0.5, 1.5, pi/2, 1.5, 
#     1., 0.5, pi/2, 1.5, 
#     1., 0.0, pi/2, 1.5
# )
# θ=0.5





# randomize experiments:
# x0_lower_limit = vcat(initial_state - 0.2*rand(12,), SVector(θ, 0.0,θ, 0.0, 1.0))
num_samples = 10
num_theta = 10
θ_samples = range(0.1, 0.9, length=num_theta)
initial_state_tmp=SVector(
    0., 1.5, pi/2, 1.5, 
    1., 0.5, pi/2, 1.5, 
    1., 0.0, pi/2, 1.5
)
x0_samples = [[vcat(initial_state_tmp + 0.4*rand(12,) - 0.4*ones(12,), SVector(θ_samples[j], 1+0.1*(rand(1,)-ones(1,))[1],1.0, 1+0.1*(rand(1,)-ones(1,))[1], 1.0)) 
    for i in 1:num_samples] for j in 1:num_theta]

Log_p1_costs = []
Log_p2_costs = []
Log_p3_costs = []
Log_x = []
Log_π = []
Log_x2 = []
Log_π2 = []
Log_x3 = []
Log_π3 = []


function control1_policy(x,τ,π_P_list,π_α_list,u_list,x_list)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][1]-π_P_list[t][1,1:13]'*[x-x_list[t]][1][1:13]-π_α_list[t][1]
end
function control2_policy(x,τ,π_P_list,π_α_list,u_list,x_list)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][2]-π_P_list[t][2,1:13]'*[x-x_list[t]][1][1:13]-π_α_list[t][2]
end
function control3_policy(x,τ,π_P_list,π_α_list,u_list,x_list)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][3]-π_P_list[t][3,1:13]'*[x-x_list[t]][1][player2_info_list]-π_α_list[t][3]
end
function control4_policy(x,τ,π_P_list,π_α_list,u_list,x_list)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][4]-π_P_list[t][4,1:13]'*[x-x_list[t]][1][player2_info_list]-π_α_list[t][4]
end
function control5_policy(x,τ,π_P_list,π_α_list,u_list,x_list)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][5]-π_P_list[t][5,1:13]'*[x-x_list[t]][1][player3_info_list]-π_α_list[t][5]
end
function control6_policy(x,τ,π_P_list,π_α_list,u_list,x_list)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][6]-π_P_list[t][6,1:13]'*[x-x_list[t]][1][player3_info_list]-π_α_list[t][6]
end




# player 1 is the teacher, players 2 and 3 are students
function player2_imagined_control_policy(x, τ, index, π_P_list, π_α_list, u_list, x_list)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][index] - π_P_list[t][index,1:13]'*[x-x_list[t]][1][player2_info_list]-π_α_list[t][index]
end
function player3_imagined_control_policy(x, τ, index, π_P_list, π_α_list, u_list, x_list)
    t= Int(floor(τ/ΔT))+1
    return u_list[t][index] - π_P_list[t][index,1:13]'*[x-x_list[t]][1][player3_info_list]-π_α_list[t][index]
end

# belief update:
function player2_belief_mean_update_policy(x,u,τ,π_P_list,π_α_list,u_list,x_list)
    t= Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[15]*π_P_list[t][1:2,13]'*inv(I(2)+π_P_list[t][1:2,13]*x[15]*π_P_list[t][1:2,13]') * ( 
        [control1_policy(x,τ,π_P_list,π_α_list,u_list,x_list)+u[1] - player2_imagined_control_policy(x,τ,1,π_P_list,π_α_list,u_list,x_list);  
        control2_policy(x,τ,π_P_list,π_α_list,u_list,x_list)+u[2] - player2_imagined_control_policy(x,τ,2,π_P_list, π_α_list,u_list,x_list)] )
end

function player2_belief_variance_update_policy(x,u,τ,π_P_list,π_α_list,u_list,x_list)
    t = Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[15]*π_P_list[t][1:2,13]'*inv(I(2)+π_P_list[t][1:2,13]*x[15]*π_P_list[t][1:2,13]') * π_P_list[t][1:2,13]*x[15]
end

function player3_belief_mean_update_policy(x,u,τ,π_P_list,π_α_list,u_list,x_list)
    t= Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[17]*π_P_list[t][1:2,13]'*inv(I(2)+π_P_list[t][1:2,13]*x[17]*π_P_list[t][1:2,13]') * ( 
        [control1_policy(x,τ,π_P_list,π_α_list,u_list,x_list)+u[1] - player3_imagined_control_policy(x,τ,1,π_P_list,π_α_list,u_list,x_list);2
        control2_policy(x,τ,π_P_list,π_α_list,u_list,x_list)+u[2] - player3_imagined_control_policy(x,τ,2,π_P_list,π_α_list,u_list,x_list)] )
end

function player3_belief_variance_update_policy(x,u,τ,π_P_list,π_α_list,u_list,x_list)
    t = Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[17]*π_P_list[t][1:2,13]'*inv(I(2)+π_P_list[t][1:2,13]*x[17]*π_P_list[t][1:2,13]') * π_P_list[t][1:2,13]*x[17]
end



# active teaching related functions:
# struct ThreeUnicycles2_policy <: ControlSystem{ΔT, nx, nu } end
# dx(cs::ThreeUnicycles2_policy, x, u, t) = SVector(
# x[4]cos(x[3]), 
# x[4]sin(x[3]), 
# control1_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp) + u[1], 
# control2_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp) + u[2], 
# x[8]cos(x[7]), 
# x[8]sin(x[7]), 
# control3_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), 
# control4_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
# x[12]cos(x[11]), 
# x[12]sin(x[11]), 
# control5_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), 
# control6_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
# 0,  # parameter of player 1, invisible to students
# player2_belief_mean_update_policy(x,u,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
# player2_belief_variance_update_policy(x,u,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
# player3_belief_mean_update_policy(x,u,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
# player3_belief_variance_update_policy(x,u,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
# )
# dynamics2_policy = ThreeUnicycles2_policy();
# g2_policy = GeneralGame(game_horizon, player_inputs, dynamics2_policy, costs2);
# solver2_policy = iLQSolver(g2_policy, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")



# passive teaching related functions:
function player2_passive_belief_mean_update_policy(x,τ,π_P_list,π_α_list,u_list,x_list)
    t= Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[15]*π_P_list[t][1:2,13]'*inv(I(2)+π_P_list[t][1:2,13]*x[15]*π_P_list[t][1:2,13]') * ( 
        [control1_policy(x,τ,π_P_list,π_α_list,u_list,x_list) - player2_imagined_control_policy(x,τ,1,π_P_list,π_α_list,u_list,x_list);  
        control2_policy(x,τ,π_P_list,π_α_list,u_list,x_list) - player2_imagined_control_policy(x,τ,2,π_P_list,π_α_list,u_list,x_list)] )
end
function player2_passive_belief_variance_update_policy(x,τ,π_P_list,π_α_list,u_list,x_list)
    t = Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[15]*π_P_list[t][1:2,13]'*inv(I(2)+π_P_list[t][1:2,13]*x[15]*π_P_list[t][1:2,13]') * π_P_list[t][1:2,13]*x[15]
end
function player3_passive_belief_mean_update_policy(x,τ,π_P_list,π_α_list,u_list,x_list)
    t= Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[17]*π_P_list[t][1:2,13]'*inv(I(2)+π_P_list[t][1:2,13]*x[17]*π_P_list[t][1:2,13]') * ( 
        [control1_policy(x,τ,π_P_list,π_α_list,u_list,x_list) - player3_imagined_control_policy(x,τ,1,π_P_list,π_α_list,u_list,x_list);
        control2_policy(x,τ,π_P_list,π_α_list,u_list,x_list) - player3_imagined_control_policy(x,τ,2,π_P_list,π_α_list,u_list,x_list)] )
end
function player3_passive_belief_variance_update_policy(x,τ,π_P_list,π_α_list,u_list,x_list)
    t = Int(floor(τ/ΔT))+1
    return - 1/ΔT* x[17]*π_P_list[t][1:2,13]'*inv(I(2)+π_P_list[t][1:2,13]*x[17]*π_P_list[t][1:2,13]') * π_P_list[t][1:2,13]*x[17]
end
# struct ThreeUnicycles3_policy <: ControlSystem{ΔT, nx, nu } end
# dx(cs::ThreeUnicycles3_policy, x, u, t) = SVector(
# x[4]cos(x[3]), 
# x[4]sin(x[3]), 
# control1_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), 
# control2_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), 
# x[8]cos(x[7]), 
# x[8]sin(x[7]), 
# control3_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), 
# control4_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
# x[12]cos(x[11]), 
# x[12]sin(x[11]), 
# control5_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), 
# control6_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
# 0,  # parameter of player 1, invisible to students
# player2_passive_belief_mean_update_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
# player2_passive_belief_variance_update_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
# player3_passive_belief_mean_update_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
# player3_passive_belief_variance_update_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
# )
# dynamics3_policy = ThreeUnicycles3_policy();
# g3_policy = GeneralGame(game_horizon, player_inputs, dynamics3_policy, costs3);
# solver3_policy = iLQSolver(g3_policy, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")



# policy evaluation:
function complete_control_policy(x,t,π_P_list,π_α_list,u_list,x_list)
    return [control1_policy(x,t,π_P_list,π_α_list,u_list,x_list); 
    control2_policy(x,t,π_P_list,π_α_list,u_list,x_list); 
    control3_policy(x,t,π_P_list,π_α_list,u_list,x_list); 
    control4_policy(x,t,π_P_list,π_α_list,u_list,x_list); 
    control5_policy(x,t,π_P_list,π_α_list,u_list,x_list); 
    control6_policy(x,t,π_P_list,π_α_list,u_list,x_list)]
end
function passive_control_policy(x,t,π_P_list,π_α_list,u_list,x_list)
    return [control1_policy(x,t,π_P_list,π_α_list,u_list,x_list); 
    control2_policy(x,t,π_P_list,π_α_list,u_list,x_list); 
    control3_policy(x,t,π_P_list,π_α_list,u_list,x_list); 
    control4_policy(x,t,π_P_list,π_α_list,u_list,x_list); 
    control5_policy(x,t,π_P_list,π_α_list,u_list,x_list); 
    control6_policy(x,t,π_P_list,π_α_list,u_list,x_list)]
end
function active_control_policy(x,u,t,π_P_list,π_α_list,u_list,x_list)
    return [control1_policy(x,t,π_P_list,π_α_list,u_list,x_list)+u[1]; 
    control2_policy(x,t,π_P_list,π_α_list,u_list,x_list)+u[2]; 
    control3_policy(x,t,π_P_list,π_α_list,u_list,x_list); 
    control4_policy(x,t,π_P_list,π_α_list,u_list,x_list); 
    control5_policy(x,t,π_P_list,π_α_list,u_list,x_list); 
    control6_policy(x,t,π_P_list,π_α_list,u_list,x_list)]
end



# compute costs for complete information games
for outer_iter in 1:num_theta
    player1_costs = zeros(3,num_samples) # 1 is complete, 2 is active, 3 is passsive
    player2_costs = zeros(3,num_samples)
    player3_costs = zeros(3,num_samples)
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
        dx(cs::ThreeUnicycles2, x, u, t) = SVector(
        x[4]cos(x[3]), 
        x[4]sin(x[3]), 
        control1_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp) + u[1], 
        control2_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp) + u[2], 
        x[8]cos(x[7]), 
        x[8]sin(x[7]), 
        control3_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), 
        control4_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
        x[12]cos(x[11]), 
        x[12]sin(x[11]), 
        control5_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), 
        control6_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
        0,  # parameter of player 1, invisible to students
        player2_belief_mean_update_policy(x,u,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
        player2_belief_variance_update_policy(x,u,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
        player3_belief_mean_update_policy(x,u,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
        player3_belief_variance_update_policy(x,u,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
        )
        dynamics2_policy = ThreeUnicycles2();
        g2_policy = GeneralGame(game_horizon, player_inputs, dynamics2_policy, costs2);
        solver2_policy = iLQSolver(g2_policy, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")

        c2_tmp, x2_tmp, π2_tmp = solve(g2_policy, solver2_policy, x0_samples[outer_iter][iter]);
        push!(log_x2, x2_tmp)
        push!(log_π2, π2_tmp)
        
        # passive:
        dx(cs::ThreeUnicycles3, x, u, t) = SVector(
        x[4]cos(x[3]), 
        x[4]sin(x[3]), 
        control1_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), 
        control2_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), 
        x[8]cos(x[7]), 
        x[8]sin(x[7]), 
        control3_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), 
        control4_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
        x[12]cos(x[11]), 
        x[12]sin(x[11]), 
        control5_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), 
        control6_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
        0,  # parameter of player 1, invisible to students
        player2_passive_belief_mean_update_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
        player2_passive_belief_variance_update_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
        player3_passive_belief_mean_update_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
        player3_passive_belief_variance_update_policy(x,t,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp),
        )
        dynamics3_policy = ThreeUnicycles3();
        g3_policy = GeneralGame(game_horizon, player_inputs, dynamics3_policy, costs3);
        solver3_policy = iLQSolver(g3_policy, max_scale_backtrack=5, max_n_iter=10, max_elwise_diff_step=Inf, equilibrium_type="FBNE")
        c3_tmp, x3_tmp, π3_tmp = solve(g3_policy, solver3_policy, x0_samples[outer_iter][iter]);
        push!(log_x3, x3_tmp)
        push!(log_π3, π3_tmp)
        
        # record costs:
        complete_costs_player_1_tmp = sum([costs[1](g, x_tmp.x[t], x_tmp.u[t], t*ΔT)[1] for t in 1:game_horizon])
        complete_costs_player_2_tmp = sum([costs[2](g, x_tmp.x[t], x_tmp.u[t], t*ΔT)[1] for t in 1:game_horizon])
        complete_costs_player_3_tmp = sum([costs[3](g, x_tmp.x[t], x_tmp.u[t], t*ΔT)[1] for t in 1:game_horizon])

        active_costs_player_1_tmp = sum([costs[1](g2, x2_tmp.x[t], active_control_policy(x2_tmp.x[t],x2.u[t],t*ΔT,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), t*ΔT)[1] for t in 1:game_horizon])
        active_costs_player_2_tmp = sum([costs[2](g2, x2_tmp.x[t], active_control_policy(x2_tmp.x[t],x2.u[t],t*ΔT,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), t*ΔT)[1] for t in 1:game_horizon])
        active_costs_player_3_tmp = sum([costs[3](g2, x2_tmp.x[t], active_control_policy(x2_tmp.x[t],x2.u[t],t*ΔT,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), t*ΔT)[1] for t in 1:game_horizon])

        passive_costs_player_1_tmp = sum([costs[1](g3, x3_tmp.x[t], passive_control_policy(x3_tmp.x[t],t*ΔT,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), t*ΔT)[1] for t in 1:game_horizon])
        passive_costs_player_2_tmp = sum([costs[2](g3, x3_tmp.x[t], passive_control_policy(x3_tmp.x[t],t*ΔT,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), t*ΔT)[1] for t in 1:game_horizon])
        passive_costs_player_3_tmp = sum([costs[3](g3, x3_tmp.x[t], passive_control_policy(x3_tmp.x[t],t*ΔT,π_P_list_tmp,π_α_list_tmp,u_list_tmp,x_list_tmp), t*ΔT)[1] for t in 1:game_horizon])

        player1_costs[:,iter] = [complete_costs_player_1_tmp, active_costs_player_1_tmp, passive_costs_player_1_tmp]
        player2_costs[:,iter] = [complete_costs_player_2_tmp, active_costs_player_2_tmp, passive_costs_player_2_tmp]
        player3_costs[:,iter] = [complete_costs_player_3_tmp, active_costs_player_3_tmp, passive_costs_player_3_tmp]
    end
    push!(Log_p1_costs, player1_costs)
    push!(Log_p2_costs, player2_costs)
    push!(Log_p3_costs, player3_costs)
    push!(Log_x, log_x)
    push!(Log_π, log_π)
    push!(Log_x2, log_x2)
    push!(Log_π2, log_π2)
    push!(Log_x3, log_x3)
    push!(Log_π3, log_π3)
end



p1_active_regret = [Log_p1_costs[i][2,:] - Log_p1_costs[i][1,:] for i in 1:num_theta]
p1_passive_regret = [Log_p1_costs[i][3,:] - Log_p1_costs[i][1,:] for i in 1:num_theta]
p1_mean_active_regret = [mean(p1_active_regret[i]) for i in 1:num_theta]
p1_mean_passive_regret = [mean(p1_passive_regret[i]) for i in 1:num_theta]
p1_std_active_regret = [std(p1_active_regret[i]) for i in 1:num_theta]
p1_std_passive_regret = [std(p1_passive_regret[i]) for i in 1:num_theta]

fill_alpha = 0.4

plot(θ_samples, p1_mean_passive_regret, ribbon=p1_std_passive_regret, label="", xlabel="", ylabel="", color = passive_color, fillalpha = fill_alpha)
plot!(θ_samples, p1_mean_active_regret, ribbon=p1_std_active_regret, label="", xlabel="", ylabel="", color = active_color, fillalpha = fill_alpha)
plot!(size = (450,320), grid=false,tickfontsize=14, left_margin = 1cm, bottom_margin = 0.5cm, )
savefig(plot_path*"player1_regret.pdf")

p2_active_regret = [Log_p2_costs[i][2,:] - Log_p2_costs[i][1,:] for i in 1:num_theta]
p2_passive_regret = [Log_p2_costs[i][3,:] - Log_p2_costs[i][1,:] for i in 1:num_theta]
p2_mean_active_regret = [mean(p2_active_regret[i]) for i in 1:num_theta]
p2_mean_passive_regret = [mean(p2_passive_regret[i]) for i in 1:num_theta]
p2_std_active_regret = [std(p2_active_regret[i]) for i in 1:num_theta]
p2_std_passive_regret = [std(p2_passive_regret[i]) for i in 1:num_theta]

plot(θ_samples, p2_mean_passive_regret, ribbon=p2_std_passive_regret, label="", xlabel="", ylabel="", color = passive_color, fillalpha = fill_alpha)
plot!(θ_samples, p2_mean_active_regret, ribbon=p2_std_active_regret, label="", xlabel="", ylabel="", color = active_color, fillalpha = fill_alpha)
plot!(size=(450,160), grid=false, tickfontsize=14,left_margin = 1cm, bottom_margin = 0.5cm,)
savefig(plot_path*"player2_regret.pdf")

p3_active_regret = [Log_p3_costs[i][2,:] - Log_p3_costs[i][1,:] for i in 1:num_theta]
p3_passive_regret = [Log_p3_costs[i][3,:] - Log_p3_costs[i][1,:] for i in 1:num_theta]
p3_mean_active_regret = [mean(p3_active_regret[i]) for i in 1:num_theta]
p3_mean_passive_regret = [mean(p3_passive_regret[i]) for i in 1:num_theta]
p3_std_active_regret = [std(p3_active_regret[i]) for i in 1:num_theta]
p3_std_passive_regret = [std(p3_passive_regret[i]) for i in 1:num_theta]

plot(θ_samples, p3_mean_passive_regret, ribbon=p3_std_passive_regret, label="", xlabel="", ylabel="", color = passive_color, fillalpha = fill_alpha)
plot!(θ_samples, p3_mean_active_regret, ribbon=p3_std_active_regret, label="", xlabel="", ylabel="", color = active_color, fillalpha = fill_alpha)
plot!(size = (450,160), grid=false,tickfontsize=14,left_margin = 1cm, bottom_margin = 0.5cm,)
savefig(plot_path*"player3_regret.pdf")



