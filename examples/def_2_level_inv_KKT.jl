
function level_1_KKT(obs_x, x0, obs_time_list, obs_state_list)
    model = Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    @variable(model, x[1:nx, 1:g.h])
    @variable(model, u[1:nu, 1:g.h])
    @variable(model, λ[1:2, 1:nx, 1:g.h])
    set_start_value.(x[1:nx, 1:g.h-1], obs_x)
    @objective(model, Min,0)
        for t in 1:g.h # for each time t within the game horizon
            if t == 1
            @NLconstraint(model, x[1,1] == x0[1] + ΔT * x0[4]*cos(x0[3]))
            @NLconstraint(model, x[2,1] == x0[2] + ΔT * x0[4]*sin(x0[3]))
            @NLconstraint(model, x[3,1] == x0[3] + ΔT * u[1,t])
            @NLconstraint(model, x[4,1] == x0[4] + ΔT * u[2,t])
            @NLconstraint(model, x[5,1] == x0[5] + ΔT * x0[8]*cos(x0[7]))
            @NLconstraint(model, x[6,1] == x0[6] + ΔT * x0[8]*sin(x0[7]))
            @NLconstraint(model, x[7,1] == x0[7] + ΔT * u[3,t])
            @NLconstraint(model, x[8,1] == x0[8] + ΔT * u[4,t])
            @NLconstraint(model, x[9,1] == x0[9])
        else
            @NLconstraint(model, x[1,t] == x[1,t-1] + ΔT * x[4,t-1]*cos(x[3,t-1]))
            @NLconstraint(model, x[2,t] == x[2,t-1] + ΔT * x[4,t-1]*sin(x[3,t-1]))
            @NLconstraint(model, x[3,t] == x[3,t-1] + ΔT * u[1,t])
            @NLconstraint(model, x[4,t] == x[4,t-1] + ΔT * u[2,t])
            @NLconstraint(model, x[5,t] == x[5,t-1] + ΔT * x[8,t-1]*cos(x[7,t-1]))
            @NLconstraint(model, x[6,t] == x[6,t-1] + ΔT * x[8,t-1]*sin(x[7,t-1]))
            @NLconstraint(model, x[7,t] == x[7,t-1] + ΔT * u[3,t])
            @NLconstraint(model, x[8,t] == x[8,t-1] + ΔT * u[4,t])
            @NLconstraint(model, x[9,t] == x[9,t-1])
        end
    end
    optimize!(model)
    return value.(x), value.(u), model
end

function level_2_KKT(x_init, u_init, obs_x, θ₀, x0, obs_time_list, obs_state_index_list)
    model = Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    @variable(model, x[1:nx, 1:g.h])
    @variable(model, u[1:nu, 1:g.h])
    @variable(model, λ[1:2, 1:nx, 1:g.h])
    @variable(model, θ[1:4])
    set_start_value.(θ, θ₀)
    set_start_value.(x, x_init)
    set_start_value.(u, u_init)
    @constraint(model, θ[1] + θ[2] == 8 )
    @constraint(model, θ[3] + θ[4] == 8 )
    @constraint(model, θ.>=0)
    @objective(model, Min, sum(sum((x[ii,t] - obs_x[ii,t])^2 for ii in obs_state_index_list ) for t in obs_time_list))# + ctrl_coeff*sum(sum((u[ii,t] - obs_u[ii,t])^2 for ii in obs_control_index_list) for t in obs_time_list) )
    for t in 1:g.h # for each time t within the game horizon
        if t != g.h # dJ1/dx
            @constraint(model,   λ[1,1,t] + 2*θ[2]*x[1,t]             - λ[1,1,t+1] == 0)
            @constraint(model,   λ[1,2,t]               - λ[1,2,t+1] == 0)
            @NLconstraint(model, λ[1,3,t]               - λ[1,3,t+1] + λ[1,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[1,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
            @NLconstraint(model, λ[1,4,t]               - λ[1,4,t+1] - λ[1,1,t+1]*ΔT*cos(x[3,t]) - λ[1,2,t+1]*ΔT*sin(x[3,t]) == 0)
            @constraint(model,   λ[1,5,t] + 2*θ[1]*(x[5,t]-x[9,t])           - λ[1,5,t+1] == 0)
            @constraint(model,   λ[1,6,t]               - λ[1,6,t+1] == 0)
            @NLconstraint(model, λ[1,7,t]               - λ[1,7,t+1] + λ[1,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[1,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
            @NLconstraint(model, λ[1,8,t]               - λ[1,8,t+1] - λ[1,5,t+1]*ΔT*cos(x[7,t]) - λ[1,6,t+1]*ΔT*sin(x[7,t]) == 0)
            @NLconstraint(model, λ[1,9,t] + 2*θ[1]*(x[9,t]-x[5,t])              - λ[1,9,t+1] == 0)
        else
            @constraint(model,   λ[1,1,t] + 2*θ[2]*x[1,t] == 0)
            @constraint(model,   λ[1,2,t] == 0)
            @NLconstraint(model, λ[1,3,t] == 0)
            @NLconstraint(model, λ[1,4,t] == 0)
            @constraint(model,   λ[1,5,t] + 2*θ[1]*(x[5,t]-x[9,t]) == 0)
            @constraint(model,   λ[1,6,t] == 0)
            @NLconstraint(model, λ[1,7,t] == 0)
            @NLconstraint(model, λ[1,8,t] == 0)
            @NLconstraint(model, λ[1,9,t] + 2*θ[1]*(x[9,t]-x[5,t]) == 0)
        end

        if t != g.h # dJ2/dx
            @constraint(model,   λ[2,1,t] + 2*θ[3]*(x[1,t]-x[5,t])                   - λ[2,1,t+1] == 0)
            @constraint(model,   λ[2,2,t]                   - λ[2,2,t+1] == 0)
            @NLconstraint(model, λ[2,3,t]                   - λ[2,3,t+1] + λ[2,1,t+1]*ΔT*x[4,t]*sin(x[3,t]) - λ[2,2,t+1]*ΔT*x[4,t]*cos(x[3,t])  == 0)
            @NLconstraint(model, λ[2,4,t]                   - λ[2,4,t+1] - λ[2,1,t+1]*ΔT*cos(x[3,t]) - λ[2,2,t+1]*ΔT*sin(x[3,t]) == 0)
            @constraint(model,   λ[2,5,t] + 2*θ[3]*(x[5,t]-x[1,t])                    - λ[2,5,t+1] == 0)
            @constraint(model,   λ[2,6,t]                   - λ[2,6,t+1] == 0)
            @NLconstraint(model, λ[2,7,t]                   - λ[2,7,t+1] + λ[2,5,t+1]*ΔT*x[8,t]*sin(x[7,t]) - λ[2,6,t+1]*ΔT*x[8,t]*cos(x[7,t]) == 0)
            @NLconstraint(model, λ[2,8,t] + 2*θ[4]*(x[8,t]-1)                  - λ[2,8,t+1] - λ[2,5,t+1]*ΔT*cos(x[7,t]) - λ[2,6,t+1]*ΔT*sin(x[7,t]) == 0)
            @NLconstraint(model, λ[2,9,t]                   -λ[2,9,t+1] == 0)
        else
            @constraint(model,   λ[2,1,t] + 2*θ[3]*(x[1,t]-x[5,t]) == 0)
            @constraint(model,   λ[2,2,t]  == 0)
            @NLconstraint(model, λ[2,3,t]  == 0)
            @NLconstraint(model, λ[2,4,t]  == 0)
            @constraint(model,   λ[2,5,t] + 2*θ[3]*(x[5,t]-x[1,t]) == 0)
            @constraint(model,   λ[2,6,t]  == 0)
            @NLconstraint(model, λ[2,7,t]  == 0)
            @NLconstraint(model, λ[2,8,t] + 2*θ[4]*(x[8,t]-1) == 0)
            @NLconstraint(model, λ[2,9,t] == 0)
        end

        # dJ1/du and dJ2/du
        @constraint(model, 4*u[1,t] - λ[1,3,t]*ΔT == 0)
        @constraint(model, 4*u[2,t] - λ[1,4,t]*ΔT == 0)
        @constraint(model, 4*u[3,t] - λ[2,7,t]*ΔT == 0)
        @constraint(model, 4*u[4,t] - λ[2,8,t]*ΔT == 0)
        if t == 1
            @NLconstraint(model, x[1,1] == x0[1] + ΔT * x0[4]*cos(x0[3]))
            @NLconstraint(model, x[2,1] == x0[2] + ΔT * x0[4]*sin(x0[3]))
            @NLconstraint(model, x[3,1] == x0[3] + ΔT * u[1,t])
            @NLconstraint(model, x[4,1] == x0[4] + ΔT * u[2,t])
            @NLconstraint(model, x[5,1] == x0[5] + ΔT * x0[8]*cos(x0[7]))
            @NLconstraint(model, x[6,1] == x0[6] + ΔT * x0[8]*sin(x0[7]))
            @NLconstraint(model, x[7,1] == x0[7] + ΔT * u[3,t])
            @NLconstraint(model, x[8,1] == x0[8] + ΔT * u[4,t])
            @NLconstraint(model, x[9,1] == x0[9])
        else
            @NLconstraint(model, x[1,t] == x[1,t-1] + ΔT * x[4,t-1]*cos(x[3,t-1]))
            @NLconstraint(model, x[2,t] == x[2,t-1] + ΔT * x[4,t-1]*sin(x[3,t-1]))
            @NLconstraint(model, x[3,t] == x[3,t-1] + ΔT * u[1,t])
            @NLconstraint(model, x[4,t] == x[4,t-1] + ΔT * u[2,t])
            @NLconstraint(model, x[5,t] == x[5,t-1] + ΔT * x[8,t-1]*cos(x[7,t-1]))
            @NLconstraint(model, x[6,t] == x[6,t-1] + ΔT * x[8,t-1]*sin(x[7,t-1]))
            @NLconstraint(model, x[7,t] == x[7,t-1] + ΔT * u[3,t])
            @NLconstraint(model, x[8,t] == x[8,t-1] + ΔT * u[4,t])
            @NLconstraint(model, x[9,t] == x[9,t-1])
        end
    end
    optimize!(model)
    return value.(x), value.(u), value.(θ), model
end