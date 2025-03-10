
function level_1_KKT_x0(obs_x, obs_time_list, obs_state_list)
    #obs_x should be 1:g.h, correspond to x[end],x[1:g.h-1]
    #obs_state_list should be 1:39 at most
    model = Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    @variable(model, x[1:nx, 1:g.h+1]) # x0[end] is treated as x0
    @variable(model, u[1:nu, 1:g.h])
    @variable(model, λ[1:2, 1:nx, 1:g.h])
    set_start_value.(x[1:nx, 1:g.h], obs_x)
    #-1e-4*sum(log((x[5,t]-x[9,t])^2 +(x[6,t]-x[10,t])^2+0.1) for t in 1:g.h-1) + 
    @NLobjective(model, Min,  sum((x[ii,end] - obs_x[ii,1])^2 for ii in obs_state_list)+sum(sum((x[ii,t-1] - obs_x[ii,t])^2 for ii in obs_state_list) for t in obs_time_list.+1))
    for t in 1:g.h # for each time t within the game horizon
        # for convention, we model x[:,end] as the x0
        if t == 1
            @NLconstraint(model, x[1,1] == x[1,end] + ΔT * x[4,end]*cos(x[3,end]))
            @NLconstraint(model, x[2,1] == x[2,end] + ΔT * x[4,end]*sin(x[3,end]))
            @NLconstraint(model, x[3,1] == x[3,end] + ΔT * u[1,1])
            @NLconstraint(model, x[4,1] == x[4,end] + ΔT * u[2,1])
            @NLconstraint(model, x[5,1] == x[5,end] + ΔT * x[8,end]*cos(x[7,end]))
            @NLconstraint(model, x[6,1] == x[6,end] + ΔT * x[8,end]*sin(x[7,end]))
            @NLconstraint(model, x[7,1] == x[7,end] + ΔT * u[3,1])
            @NLconstraint(model, x[8,1] == x[8,end] + ΔT * u[4,1])
            @NLconstraint(model, x[9,1] == x[9,end] + ΔT * x[12,end]*cos(x[11,end]))
            @NLconstraint(model, x[10,1] == x[10,end] + ΔT * x[12,end]*sin(x[11,end]))
            @NLconstraint(model, x[11,1] == x[11,end] + ΔT * u[5,1])
            @NLconstraint(model, x[12,1] == x[12,end] + ΔT * u[6,1])
            @NLconstraint(model, x[13,1] == x[13,end])
        else
            @NLconstraint(model, x[1,t] == x[1,t-1] + ΔT * x[4,t-1]*cos(x[3,t-1]))
            @NLconstraint(model, x[2,t] == x[2,t-1] + ΔT * x[4,t-1]*sin(x[3,t-1]))
            @NLconstraint(model, x[3,t] == x[3,t-1] + ΔT * u[1,t])
            @NLconstraint(model, x[4,t] == x[4,t-1] + ΔT * u[2,t])
            @NLconstraint(model, x[5,t] == x[5,t-1] + ΔT * x[8,t-1]*cos(x[7,t-1]))
            @NLconstraint(model, x[6,t] == x[6,t-1] + ΔT * x[8,t-1]*sin(x[7,t-1]))
            @NLconstraint(model, x[7,t] == x[7,t-1] + ΔT * u[3,t])
            @NLconstraint(model, x[8,t] == x[8,t-1] + ΔT * u[4,t])
            @NLconstraint(model, x[9,t] == x[9,t-1] + ΔT * x[12,t-1]*cos(x[11,t-1]))
            @NLconstraint(model, x[10,t] == x[10,t-1] + ΔT * x[12,t-1]*sin(x[11,t-1]))
            @NLconstraint(model, x[11,t] == x[11,t-1] + ΔT * u[5,t])
            @NLconstraint(model, x[12,t] == x[12,t-1] + ΔT * u[6,t])
            @NLconstraint(model, x[13,t] == x[13,t-1])
        end
    end
    optimize!(model)
    return value.(x), value.(u), model
end
# costs = (FunctionPlayerCost((g,x,u,t) -> ( θ[1]*x[1]^2 +θ[2]*(x[5]-x[13])^2   +4*(x[3]-pi/2)^2  +2*(x[4]-2)^2       +2*(u[1]^2 + u[2]^2)    )),
#          FunctionPlayerCost((g,x,u,t) -> ( θ[3]*(x[5]-x[1])^2 + θ[4]*x[5]^2   +4*(x[7]-pi/2)^2  +2*(x[8]-2)^2       -log((x[5]-x[9])^2+(x[6]-x[10])^2)    +2*(u[3]^2+u[4]^2)    )),
#          FunctionPlayerCost((g,x,u,t) -> ( 2*(x[9]-x0[9])^2   + 2*(u[5]^2+u[6]^2)  ))
#     )
function level_2_KKT_x0(x_init, u_init, obs_x, θ₀, obs_time_list, obs_state_index_list)
    model = Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    @variable(model, x[1:nx, 1:g.h+1])
    @variable(model, u[1:nu, 1:g.h])
    @variable(model, λ[1:3, 1:nx, 1:g.h])
    @variable(model, θ[1:5]) #-
    set_start_value.(θ, θ₀)
    set_start_value.(x, x_init)
    set_start_value.(u, u_init)
    @constraint(model, sum(θ) <= 20)
    @constraint(model, θ.>=0)
    @objective(model, Min, 1e-4*sum(θ.*θ)+sum((x[ii,end]-obs_x[ii,1])^2 for ii in obs_state_index_list)+sum(sum((x[ii,t-1] - obs_x[ii,t])^2 for ii in obs_state_index_list ) for t in obs_time_list.+1))
    for t in 1:g.h # for each time t within the game horizon
        # λ[t] correspond to x[t] - x[t-1]-u[t-1]
        if t != g.h # dJ1/dx
            @constraint(model, λ[1,1,t] +2*θ[1]*x[1,t]              - λ[1,1,t+1] == 0)
            @constraint(model, λ[1,2,t]               - λ[1,2,t+1] == 0)
            @NLconstraint(model, λ[1,3,t] + 8*(x[3,t]-pi/2)              - λ[1,3,t+1] + λ[1,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[1,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
            @NLconstraint(model, λ[1,4,t] + 4*(x[4,t]-2)                - λ[1,4,t+1] - λ[1,1,t+1]*ΔT*cos(x[3,t]) - λ[1,2,t+1]*ΔT*sin(x[3,t]) == 0)
            @constraint(model, λ[1,5,t] + 2*θ[2]*(x[5,t]-x[13,t])          - λ[1,5,t+1] == 0)
            @constraint(model, λ[1,6,t]               - λ[1,6,t+1] == 0)
            @NLconstraint(model, λ[1,7,t]               - λ[1,7,t+1] + λ[1,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[1,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
            @NLconstraint(model, λ[1,8,t]               - λ[1,8,t+1] - λ[1,5,t+1]*ΔT*cos(x[7,t]) - λ[1,6,t+1]*ΔT*sin(x[7,t]) == 0)
            @constraint(model, λ[1,9,t]               - λ[1,9,t+1] == 0)
            @constraint(model, λ[1,10,t]              - λ[1,10,t+1] == 0)
            @NLconstraint(model, λ[1,11,t]              - λ[1,11,t+1] + λ[1,9,t+1]*ΔT*x[12,t]*sin(x[11,t]) - λ[1,10,t+1]*ΔT*x[12,t]*cos(x[11,t]) == 0)
            @NLconstraint(model, λ[1,12,t]              - λ[1,12,t+1] - λ[1,9,t+1]*ΔT*cos(x[11,t]) - λ[1,10,t+1]*ΔT*sin(x[11,t]) == 0)
            @NLconstraint(model, λ[1,13,t] + 2*θ[2]*(x[13,t]-x[5,t])              - λ[1,13,t+1] == 0)
        else
            @constraint(model, λ[1,1,t] +2*θ[1]*x[1,t]== 0)
            @constraint(model, λ[1,2,t] == 0)
            @NLconstraint(model, λ[1,3,t] + 8*(x[3,t]-pi/2)== 0)
            @NLconstraint(model, λ[1,4,t] + 4*(x[4,t]-2)== 0)
            @constraint(model, λ[1,5,t] + 2*θ[2]*(x[5,t]-x[13,t]) == 0)
            @constraint(model, λ[1,6,t] == 0)
            @NLconstraint(model, λ[1,7,t] == 0)
            @NLconstraint(model, λ[1,8,t] == 0)
            @constraint(model, λ[1,9,t] == 0)
            @constraint(model, λ[1,10,t] == 0)
            @NLconstraint(model, λ[1,11,t] == 0)
            @NLconstraint(model, λ[1,12,t] == 0)
            @NLconstraint(model, λ[1,13,t] + 2*θ[2]*(x[13,t]-x[5,t]) == 0)
        end
        if t != g.h # dJ2/dx
            @constraint(model, λ[2,1,t] +2*θ[3]*(x[1,t]-x[5,t])              - λ[2,1,t+1] == 0)
            @constraint(model, λ[2,2,t]                 - λ[2,2,t+1] == 0)
            @NLconstraint(model, λ[2,3,t]               - λ[2,3,t+1] + λ[2,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[2,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
            @NLconstraint(model, λ[2,4,t]                 - λ[2,4,t+1] - λ[2,1,t+1]*ΔT*cos(x[3,t]) - λ[2,2,t+1]*ΔT*sin(x[3,t]) == 0)
            @NLconstraint(model, λ[2,5,t] +2*θ[4]*x[5,t] +2*θ[3]*(x[5,t]-x[1,t])   -2*(x[5,t]-x[9,t])/((x[5,t]-x[9,t])^2+(x[6,t]-x[10,t])^2)          - λ[2,5,t+1] == 0)
            @NLconstraint(model, λ[2,6,t] -2*(x[6,t]-x[10,t])/((x[5,t]-x[9,t])^2 + (x[6,t]-x[10,t])^2)              - λ[2,6,t+1] == 0)
            @NLconstraint(model, λ[2,7,t] +8*(x[7,t]-pi/2)              - λ[2,7,t+1] + λ[2,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[2,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
            @NLconstraint(model, λ[2,8,t] +4*(x[8,t]-2)              - λ[2,8,t+1] - λ[2,5,t+1]*ΔT*cos(x[7,t]) - λ[2,6,t+1]*ΔT*sin(x[7,t]) == 0)
            @NLconstraint(model, λ[2,9,t] -2*(x[9,t]-x[5,t])/((x[5,t]-x[9,t])^2+(x[6,t]-x[10,t])^2)              - λ[2,9,t+1] == 0)
            @NLconstraint(model, λ[2,10,t] -2*(x[10,t]-x[6,t])/((x[5,t]-x[9,t])^2+(x[6,t]-x[10,t])^2)             - λ[2,10,t+1] == 0)
            @NLconstraint(model, λ[2,11,t]              - λ[2,11,t+1] + λ[2,9,t+1]*ΔT*x[12,t]*sin(x[11,t]) - λ[2,10,t+1]*ΔT*x[12,t]*cos(x[11,t]) == 0)
            @NLconstraint(model, λ[2,12,t]              - λ[2,12,t+1] - λ[2,9,t+1]*ΔT*cos(x[11,t]) - λ[2,10,t+1]*ΔT*sin(x[11,t]) == 0)
            @NLconstraint(model, λ[2,13,t]              - λ[2,13,t+1] == 0)
        else
            @constraint(model, λ[2,1,t] +2*θ[3]*(x[1,t]-x[5,t])== 0)
            @constraint(model, λ[2,2,t] == 0)
            @NLconstraint(model, λ[2,3,t] == 0)
            @NLconstraint(model, λ[2,4,t] == 0)
            @NLconstraint(model, λ[2,5,t] +2*θ[4]*x[5,t] +2*θ[3]*(x[5,t]-x[1,t])  -2*(x[5,t]-x[9,t])/((x[5,t]-x[9,t])^2+(x[6,t]-x[10,t])^2) == 0)
            @NLconstraint(model, λ[2,6,t] -2*(x[6,t]-x[10,t])/((x[5,t]-x[9,t])^2 + (x[6,t]-x[10,t])^2)== 0)
            @NLconstraint(model, λ[2,7,t] +8*(x[7,t]-pi/2)== 0)
            @NLconstraint(model, λ[2,8,t] +4*(x[8,t]-2)== 0)
            @NLconstraint(model, λ[2,9,t] -2*(x[9,t]-x[5,t])/((x[5,t]-x[9,t])^2+(x[6,t]-x[10,t])^2)== 0)
            @NLconstraint(model, λ[2,10,t] -2*(x[10,t]-x[6,t])/((x[5,t]-x[9,t])^2+(x[6,t]-x[10,t])^2)== 0)
            @NLconstraint(model, λ[2,11,t] == 0)
            @NLconstraint(model, λ[2,12,t] == 0)
            @NLconstraint(model, λ[2,13,t] == 0)
        end
        if t != g.h # dJ3/dx
            @constraint(model,   λ[3,1,t]               - λ[3,1,t+1] == 0)
            @constraint(model,   λ[3,2,t]               - λ[3,2,t+1] == 0)
            @NLconstraint(model, λ[3,3,t]               - λ[3,3,t+1] + λ[3,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[3,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
            @NLconstraint(model, λ[3,4,t]               - λ[3,4,t+1] - λ[3,1,t+1]*ΔT*cos(x[3,t]) - λ[3,2,t+1]*ΔT*sin(x[3,t]) == 0)
            @constraint(model,   λ[3,5,t]               - λ[3,5,t+1] == 0)
            @constraint(model,   λ[3,6,t]               - λ[3,6,t+1] == 0)
            @NLconstraint(model, λ[3,7,t]               - λ[3,7,t+1] + λ[3,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[3,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
            @NLconstraint(model, λ[3,8,t]               - λ[3,8,t+1] - λ[3,5,t+1]*ΔT*cos(x[7,t]) - λ[3,6,t+1]*ΔT*sin(x[7,t]) == 0)
            # @constraint(model,   λ[3,9,t] +4*(x[9,t]-x0[9])              - λ[3,9,t+1] == 0)
            @constraint(model,   λ[3,9,t] +2*θ[5]*(x[9,t]-x0[9])              - λ[3,9,t+1] == 0) #-
            @constraint(model,   λ[3,10,t]              - λ[3,10,t+1] == 0)
            @NLconstraint(model, λ[3,11,t]              - λ[3,11,t+1] + λ[3,9,t+1]*ΔT*x[12,t]*sin(x[11,t]) - λ[3,10,t+1]*ΔT*x[12,t]*cos(x[11,t]) == 0)
            @NLconstraint(model, λ[3,12,t]              - λ[3,12,t+1] - λ[3,9,t+1]*ΔT*cos(x[11,t]) - λ[3,10,t+1]*ΔT*sin(x[11,t]) == 0)
            @NLconstraint(model, λ[3,13,t]              - λ[3,13,t+1] == 0)
        else
            @constraint(model,   λ[3,1,t] == 0)
            @constraint(model,   λ[3,2,t] == 0)
            @NLconstraint(model, λ[3,3,t] == 0)
            @NLconstraint(model, λ[3,4,t] == 0)
            @constraint(model,   λ[3,5,t] == 0)
            @constraint(model,   λ[3,6,t] == 0)
            @NLconstraint(model, λ[3,7,t] == 0)
            @NLconstraint(model, λ[3,8,t] == 0)
            # @constraint(model,   λ[3,9,t] +4*(x[9,t]-x0[9])== 0)
            @constraint(model,   λ[3,9,t] +2*θ[5]*(x[9,t]-x0[9])== 0) #-
            @constraint(model,   λ[3,10,t] == 0)
            @NLconstraint(model, λ[3,11,t] == 0)
            @NLconstraint(model, λ[3,12,t] == 0)
            @NLconstraint(model, λ[3,13,t] == 0)
        end
        # dJ1/du, dJ2/du and dJ3/du
        @constraint(model, 4*u[1,t] - λ[1,3,t]*ΔT == 0)
        @constraint(model, 4*u[2,t] - λ[1,4,t]*ΔT == 0)
        @constraint(model, 4*u[3,t] - λ[2,7,t]*ΔT == 0)
        @constraint(model, 4*u[4,t] - λ[2,8,t]*ΔT == 0)
        @constraint(model, 4*u[5,t] - λ[3,11,t]*ΔT == 0)
        @constraint(model, 4*u[6,t] - λ[3,12,t]*ΔT == 0)
        if t == 1
            @NLconstraint(model, x[1,1] == x[1,end] + ΔT * x[4,end]*cos(x[3,end]))
            @NLconstraint(model, x[2,1] == x[2,end] + ΔT * x[4,end]*sin(x[3,end]))
            @NLconstraint(model, x[3,1] == x[3,end] + ΔT * u[1,1])
            @NLconstraint(model, x[4,1] == x[4,end] + ΔT * u[2,1])
            @NLconstraint(model, x[5,1] == x[5,end] + ΔT * x[8,end]*cos(x[7,end]))
            @NLconstraint(model, x[6,1] == x[6,end] + ΔT * x[8,end]*sin(x[7,end]))
            @NLconstraint(model, x[7,1] == x[7,end] + ΔT * u[3,1])
            @NLconstraint(model, x[8,1] == x[8,end] + ΔT * u[4,1])
            @NLconstraint(model, x[9,1] == x[9,end] + ΔT * x[12,end]*cos(x[11,end]))
            @NLconstraint(model, x[10,1] == x[10,end] + ΔT * x[12,end]*sin(x[11,end]))
            @NLconstraint(model, x[11,1] == x[11,end] + ΔT * u[5,1])
            @NLconstraint(model, x[12,1] == x[12,end] + ΔT * u[6,1])
            @NLconstraint(model, x[13,1] == x[13,end])
        else
            @NLconstraint(model, x[1,t] == x[1,t-1] + ΔT * x[4,t-1]*cos(x[3,t-1]))
            @NLconstraint(model, x[2,t] == x[2,t-1] + ΔT * x[4,t-1]*sin(x[3,t-1]))
            @NLconstraint(model, x[3,t] == x[3,t-1] + ΔT * u[1,t])
            @NLconstraint(model, x[4,t] == x[4,t-1] + ΔT * u[2,t])
            @NLconstraint(model, x[5,t] == x[5,t-1] + ΔT * x[8,t-1]*cos(x[7,t-1]))
            @NLconstraint(model, x[6,t] == x[6,t-1] + ΔT * x[8,t-1]*sin(x[7,t-1]))
            @NLconstraint(model, x[7,t] == x[7,t-1] + ΔT * u[3,t])
            @NLconstraint(model, x[8,t] == x[8,t-1] + ΔT * u[4,t])
            @NLconstraint(model, x[9,t] == x[9,t-1] + ΔT * x[12,t-1]*cos(x[11,t-1]))
            @NLconstraint(model, x[10,t] == x[10,t-1] + ΔT * x[12,t-1]*sin(x[11,t-1]))
            @NLconstraint(model, x[11,t] == x[11,t-1] + ΔT * u[5,t])
            @NLconstraint(model, x[12,t] == x[12,t-1] + ΔT * u[6,t])
            @NLconstraint(model, x[13,t] == x[13,t-1])
        end
    end
    optimize!(model)
    return value.(x), value.(u), value.(θ), model
end



# ------------------------------------ Optimization problem begin ------------------------------------------- #

# function KKT_highway_forward_game_solve( x0,g)
#     model = Model(Ipopt.Optimizer)
#     @variable(model, x[1:nx, 1:g.h+1])
#     @variable(model, u[1:nu, 1:g.h])
#     @variable(model, θ[1:4])
#     set_start_value.(θ, θ_true)
#     set_start_value.(x[:,1:g.h], noisy_obs_x_OL)
#     set_start_value.(u[:,1:g.h], noisy_obs_u_OL)
#     @variable(model, λ[1:3, 1:nx, 1:g.h])
#     @objective(model, Min, 0)
#     ΔT = 0.1
#     for t in 1:g.h # for each time t within the game horizon
#         if t != g.h # dJ1/dx
#             @constraint(model, λ[1,1,t] +θ[1]*x[1,t]              - λ[1,1,t+1] == 0)
#             @constraint(model, λ[1,2,t]               - λ[1,2,t+1] == 0)
#             @NLconstraint(model, λ[1,3,t] + 8*(x[3,t]-pi/2)              - λ[1,3,t+1] + λ[1,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[1,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
#             @NLconstraint(model, λ[1,4,t] + 16*(x[4,t]-2)                - λ[1,4,t+1] - λ[1,1,t+1]*ΔT*cos(x[3,t]) - λ[1,2,t+1]*ΔT*sin(x[3,t]) == 0)
#             @constraint(model, λ[1,5,t] + 2*θ[2]*(x[5,t]-x[13,t])          - λ[1,5,t+1] == 0)
#             @constraint(model, λ[1,6,t]               - λ[1,6,t+1] == 0)
#             @NLconstraint(model, λ[1,7,t]               - λ[1,7,t+1] + λ[1,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[1,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
#             @NLconstraint(model, λ[1,8,t]               - λ[1,8,t+1] - λ[1,5,t+1]*ΔT*cos(x[7,t]) - λ[1,6,t+1]*ΔT*sin(x[7,t]) == 0)
#             @constraint(model, λ[1,9,t]               - λ[1,9,t+1] == 0)
#             @constraint(model, λ[1,10,t]              - λ[1,10,t+1] == 0)
#             @NLconstraint(model, λ[1,11,t]              - λ[1,11,t+1] + λ[1,9,t+1]*ΔT*x[12,t]*sin(x[11,t]) - λ[1,10,t+1]*ΔT*x[12,t]*cos(x[11,t]) == 0)
#             @NLconstraint(model, λ[1,12,t]              - λ[1,12,t+1] - λ[1,9,t+1]*ΔT*cos(x[11,t]) - λ[1,10,t+1]*ΔT*sin(x[11,t]) == 0)
#             @NLconstraint(model, λ[1,13,t] + 2*θ[2]*(x[13,t]-x[5,t])              - λ[1,13,t+1] == 0)
#         else
#             @constraint(model, λ[1,1,t] +θ[1]*x[1,t]== 0)
#             @constraint(model, λ[1,2,t] == 0)
#             @NLconstraint(model, λ[1,3,t] + 8*(x[3,t]-pi/2)== 0)
#             @NLconstraint(model, λ[1,4,t] + 16*(x[4,t]-2)== 0)
#             @constraint(model, λ[1,5,t] + 2*θ[2]*(x[5,t]-x[13,t]) == 0)
#             @constraint(model, λ[1,6,t] == 0)
#             @NLconstraint(model, λ[1,7,t] == 0)
#             @NLconstraint(model, λ[1,8,t] == 0)
#             @constraint(model, λ[1,9,t] == 0)
#             @constraint(model, λ[1,10,t] == 0)
#             @NLconstraint(model, λ[1,11,t] == 0)
#             @NLconstraint(model, λ[1,12,t] == 0)
#             @NLconstraint(model, λ[1,13,t] + 2*θ[2]*(x[13,t]-x[5,t]) == 0)
#         end
#         if t != g.h # dJ2/dx
#             @constraint(model, λ[2,1,t] +2*θ[4]*(x[1,t]-x[5,t])              - λ[2,1,t+1] == 0)
#             @constraint(model, λ[2,2,t]               - λ[2,2,t+1] == 0)
#             @NLconstraint(model, λ[2,3,t]               - λ[2,3,t+1] + λ[2,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[2,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
#             @NLconstraint(model, λ[2,4,t]                 - λ[2,4,t+1] - λ[2,1,t+1]*ΔT*cos(x[3,t]) - λ[2,2,t+1]*ΔT*sin(x[3,t]) == 0)
#             @NLconstraint(model, λ[2,5,t] +θ[3]*x[5,t] +2*θ[4]*(x[5,t]-x[1,t]) -2*(x[5,t]-x[9,t])/((x[5,t]-x[9,t])^2+(x[6,t]-x[10,t])^2)          - λ[2,5,t+1] == 0)
#             @NLconstraint(model, λ[2,6,t] -2*(x[6,t]-x[10,t])/((x[5,t]-x[9,t])^2 + (x[6,t]-x[10,t])^2)              - λ[2,6,t+1] == 0)
#             @NLconstraint(model, λ[2,7,t] +8*(x[7,t]-pi/2)              - λ[2,7,t+1] + λ[2,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[2,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
#             @NLconstraint(model, λ[2,8,t] +16*(x[8,t]-2)              - λ[2,8,t+1] - λ[2,5,t+1]*ΔT*cos(x[7,t]) - λ[2,6,t+1]*ΔT*sin(x[7,t]) == 0)
#             @NLconstraint(model, λ[2,9,t] -2*(x[9,t]-x[5,t])/((x[5,t]-x[9,t])^2+(x[6,t]-x[10,t])^2)              - λ[2,9,t+1] == 0)
#             @NLconstraint(model, λ[2,10,t] -2*(x[10,t]-x[6,t])/((x[5,t]-x[9,t])^2+(x[6,t]-x[10,t])^2)             - λ[2,10,t+1] == 0)
#             @NLconstraint(model, λ[2,11,t]              - λ[2,11,t+1] + λ[2,9,t+1]*ΔT*x[12,t]*sin(x[11,t]) - λ[2,10,t+1]*ΔT*x[12,t]*cos(x[11,t]) == 0)
#             @NLconstraint(model, λ[2,12,t]              - λ[2,12,t+1] - λ[2,9,t+1]*ΔT*cos(x[11,t]) - λ[2,10,t+1]*ΔT*sin(x[11,t]) == 0)
#             @NLconstraint(model, λ[2,13,t]              - λ[2,13,t+1] == 0)
#         else
#             @constraint(model, λ[2,1,t] +2*θ[4]*(x[1,t]-x[5,t])== 0)
#             @constraint(model, λ[2,2,t] == 0)
#             @NLconstraint(model, λ[2,3,t] == 0)
#             @NLconstraint(model, λ[2,4,t] == 0)
#             @NLconstraint(model, λ[2,5,t] +θ[3]*x[5,t] +2*θ[4]*(x[5,t]-x[1,t]) -2*(x[5,t]-x[9,t])/((x[5,t]-x[9,t])^2+(x[6,t]-x[10,t])^2) == 0)
#             @NLconstraint(model, λ[2,6,t] -2*(x[6,t]-x[10,t])/((x[5,t]-x[9,t])^2 + (x[6,t]-x[10,t])^2)== 0)
#             @NLconstraint(model, λ[2,7,t] +8*(x[7,t]-pi/2)== 0)
#             @NLconstraint(model, λ[2,8,t] +16*(x[8,t]-2)== 0)
#             @NLconstraint(model, λ[2,9,t] -2*(x[9,t]-x[5,t])/((x[5,t]-x[9,t])^2+(x[6,t]-x[10,t])^2)== 0)
#             @NLconstraint(model, λ[2,10,t] -2*(x[10,t]-x[6,t])/((x[5,t]-x[9,t])^2+(x[6,t]-x[10,t])^2)== 0)
#             @NLconstraint(model, λ[2,11,t] == 0)
#             @NLconstraint(model, λ[2,12,t] == 0)
#             @NLconstraint(model, λ[2,13,t] == 0)
#         end
#         if t != g.h # dJ3/dx
#             @constraint(model,   λ[3,1,t]               - λ[3,1,t+1] == 0)
#             @constraint(model,   λ[3,2,t]               - λ[3,2,t+1] == 0)
#             @NLconstraint(model, λ[3,3,t]               - λ[3,3,t+1] + λ[3,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[3,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
#             @NLconstraint(model, λ[3,4,t]               - λ[3,4,t+1] - λ[3,1,t+1]*ΔT*cos(x[3,t]) - λ[3,2,t+1]*ΔT*sin(x[3,t]) == 0)
#             @constraint(model,   λ[3,5,t]               - λ[3,5,t+1] == 0)
#             @constraint(model,   λ[3,6,t]               - λ[3,6,t+1] == 0)
#             @NLconstraint(model, λ[3,7,t]               - λ[3,7,t+1] + λ[3,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[3,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
#             @NLconstraint(model, λ[3,8,t]               - λ[3,8,t+1] - λ[3,5,t+1]*ΔT*cos(x[7,t]) - λ[3,6,t+1]*ΔT*sin(x[7,t]) == 0)
#             @constraint(model,   λ[3,9,t] +4*(x[9,t]-x0[9])              - λ[3,9,t+1] == 0)
#             @constraint(model,   λ[3,10,t]              - λ[3,10,t+1] == 0)
#             @NLconstraint(model, λ[3,11,t]              - λ[3,11,t+1] + λ[3,9,t+1]*ΔT*x[12,t]*sin(x[11,t]) - λ[3,10,t+1]*ΔT*x[12,t]*cos(x[11,t]) == 0)
#             @NLconstraint(model, λ[3,12,t]              - λ[3,12,t+1] - λ[3,9,t+1]*ΔT*cos(x[11,t]) - λ[3,10,t+1]*ΔT*sin(x[11,t]) == 0)
#             @NLconstraint(model, λ[3,13,t]              - λ[3,13,t+1] == 0)
#         else
#             @constraint(model,   λ[3,1,t] == 0)
#             @constraint(model,   λ[3,2,t] == 0)
#             @NLconstraint(model, λ[3,3,t] == 0)
#             @NLconstraint(model, λ[3,4,t] == 0)
#             @constraint(model,   λ[3,5,t] == 0)
#             @constraint(model,   λ[3,6,t] == 0)
#             @NLconstraint(model, λ[3,7,t] == 0)
#             @NLconstraint(model, λ[3,8,t] == 0)
#             @constraint(model,   λ[3,9,t] +4*(x[9,t]-x0[9])== 0)
#             @constraint(model,   λ[3,10,t] == 0)
#             @NLconstraint(model, λ[3,11,t] == 0)
#             @NLconstraint(model, λ[3,12,t] == 0)
#             @NLconstraint(model, λ[3,13,t] == 0)
#         end
#         # dJ1/du, dJ2/du and dJ3/du
#         @constraint(model, 4*u[1,t] - λ[1,3,t]*ΔT == 0)
#         @constraint(model, 4*u[2,t] - λ[1,4,t]*ΔT == 0)
#         @constraint(model, 4*u[3,t] - λ[2,7,t]*ΔT == 0)
#         @constraint(model, 4*u[4,t] - λ[2,8,t]*ΔT == 0)
#         @constraint(model, 4*u[5,t] - λ[3,11,t]*ΔT == 0)
#         @constraint(model, 4*u[6,t] - λ[3,12,t]*ΔT == 0)
#         if t == 1
#             @NLconstraint(model, x[1,1] == x0[1] + ΔT * x0[4]*cos(x0[3]))
#             @NLconstraint(model, x[2,1] == x0[2] + ΔT * x0[4]*sin(x0[3]))
#             @NLconstraint(model, x[3,1] == x0[3] + ΔT * u[1,t])
#             @NLconstraint(model, x[4,1] == x0[4] + ΔT * u[2,t])
#             @NLconstraint(model, x[5,1] == x0[5] + ΔT * x0[8]*cos(x0[7]))
#             @NLconstraint(model, x[6,1] == x0[6] + ΔT * x0[8]*sin(x0[7]))
#             @NLconstraint(model, x[7,1] == x0[7] + ΔT * u[3,t])
#             @NLconstraint(model, x[8,1] == x0[8] + ΔT * u[4,t])
#             @NLconstraint(model, x[9,1] == x0[9] + ΔT * x0[12]*cos(x0[11]))
#             @NLconstraint(model, x[10,1] == x0[10] + ΔT * x0[12]*sin(x0[11]))
#             @NLconstraint(model, x[11,1] == x0[11] + ΔT * u[5,t])
#             @NLconstraint(model, x[12,1] == x0[12] + ΔT * u[6,t])
#             @NLconstraint(model, x[13,1] == x0[13])
#         else
#             @NLconstraint(model, x[1,t] == x[1,t-1] + ΔT * x[4,t-1]*cos(x[3,t-1]))
#             @NLconstraint(model, x[2,t] == x[2,t-1] + ΔT * x[4,t-1]*sin(x[3,t-1]))
#             @NLconstraint(model, x[3,t] == x[3,t-1] + ΔT * u[1,t])
#             @NLconstraint(model, x[4,t] == x[4,t-1] + ΔT * u[2,t])
#             @NLconstraint(model, x[5,t] == x[5,t-1] + ΔT * x[8,t-1]*cos(x[7,t-1]))
#             @NLconstraint(model, x[6,t] == x[6,t-1] + ΔT * x[8,t-1]*sin(x[7,t-1]))
#             @NLconstraint(model, x[7,t] == x[7,t-1] + ΔT * u[3,t])
#             @NLconstraint(model, x[8,t] == x[8,t-1] + ΔT * u[4,t])
#             @NLconstraint(model, x[9,t] == x[9,t-1] + ΔT * x[12,t-1]*cos(x[11,t-1]))
#             @NLconstraint(model, x[10,t] == x[10,t-1] + ΔT * x[12,t-1]*sin(x[11,t-1]))
#             @NLconstraint(model, x[11,t] == x[11,t-1] + ΔT * u[5,t])
#             @NLconstraint(model, x[12,t] == x[12,t-1] + ΔT * u[6,t])
#             @NLconstraint(model, x[13,t] == x[13,t-1])
#         end
#     end
#     # @constraint(model, θ .>= -0.1*ones(3))
#     optimize!(model)
#     return value.(x), value.(u), value.(θ), model
# end

# for_sol=KKT_highway_forward_game_solve(x0, g)
