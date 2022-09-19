









# ------------------------------------ Optimization problem begin ------------------------------------------- #

using JuMP
using Ipopt

model = Model(Ipopt.Optimizer)
@variable(model, x[1:nx, 1:g.h])
@variable(model, u[1:nu, 1:g.h])
@variable(model, λ[1:2, 1:nx, 1:g.h])
@variable(modelm θ[1:3])

set_start_value(x, observation.x)
set_start_value(u, observation.u)

@NLobjective(model, Min, norm(x-observation.x)^2 + norm(u-observation.u)^2)
for ii in length(player_inputs) # number of players is equal to length(player_inputs)
    for t in 1:g.h
        if ii ==1
        @NLconstraint(model, dJdx1_t, 0 - λ[1,1,t] == 0)
        @NLconstraint(model, dJdy1_t, 0 - λ[1,2,t] == 0)
        @NLconstraint(model, dJdv1_t, 0 - λ[1,3,t] == 0)
        @NLconstraint(model, dJdθ1_t, 4*(x[4,t]-1) - λ[1,4,t] == 0)
        @NLconstraint(model, dJdx2_t, 2*θ[1]*(x[5,t]-1) - λ[1,5,t] == 0)
        @NLconstraint(model, dJdy2_t, 0 - λ[1,6,t] == 0)
        @NLconstraint(model, dJdv2_t, 0 - λ[1,7,t] == 0)
        @NLconstraint(model, dJdθ2_t, 0 - λ[1,8,t] == 0)
        
        @NLconstraint(model, dJdu1_t, 2*u[1,t] + λ[1,3,t]*Δt == 0)
        @NLconstraint(model, dJdu2_t, 2*u[2,t] + λ[1,4,t]*Δt == 0)
    else
        @NLconstraint(model, dJdx1_t, 2*θ[3]*(x[1,t]-x[5,t]) - λ[2,1,t] == 0)
        @NLconstraint(model, dJdy1_t, 0 - λ[2,2,t] == 0)
        @NLconstraint(model, dJdv1_t, 0 - λ[2,3,t] == 0)
        @NLconstraint(model, dJdθ1_t, 0 - λ[2,4,t] == 0)
        @NLconstraint(model, dJdx2_t, 2*θ[2]*x[5,t] + 2*θ[3]*(x[5,t]-x[1,t]) - λ[2,5,t] == 0)
        @NLconstraint(model, dJdy2_t, 0 - λ[2,6,t] == 0)
        @NLconstraint(model, dJdv2_t, 0 - λ[2,7,t] == 0)
        @NLconstraint(model, dJdθ2_t, 4*(x[8,t]-1) - λ[2,8,t] == 0)
        
        @NLconstraint(model, dJdu1_t, 2*u[3,t] + λ[2,7,t]*Δt == 0)
        @NLconstraint(model, dJdu2_t, 2*u[4,t] + λ[2,8,t]*Δt == 0)
    end

    if t != g.h
        @NLconstraint(model, x[1,t+1] == x[1,t] + Δt * x[3,t]*cos(x[4,t]))
        @NLconstraint(model, x[2,t+1] == x[2,t] + Δt * x[3,t]*sin(x[4,t]))
        @NLconstraint(model, x[3,t+1] == x[3,t] + Δt * u[1,t])
        @NLconstraint(model, x[4,t+1] == x[4,t] + Δt * u[2,t])
        @NLconstraint(model, x[5,t+1] == x[5,t] + Δt * x[7,t]*cos(x[8,t]))
        @NLconstraint(model, x[6,t+1] == x[6,y] + Δt * x[7,t]*cos(x[8,t]))
        @NLconstraint(model, x[7,t+1] == x[7,t] + Δt * u[3,t])
        @NLconstraint(model, x[8,t+1] == x[7,t] + Δt * u[4,t])
    else
        @NLconstraint(model, x[:,1] == x0)
    end        
    end
end

# ------------------------------------ Optimization problem end ------------------------------------------- #
