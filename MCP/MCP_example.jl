using Complementarity, JuMP

m = MCPModel()

lb = zeros(4)
ub = Inf*ones(4)
items = 1:4
@variable(m, lb[i] <= x[i in items] <= ub[i])

@mapping(m, F1, 3*x[1]^2 + 2*x[1]*x[2] + 2*x[2]^2 + x[3] + 3*x[4] -6)
@mapping(m, F2, 2*x[1]^2 + x[1] + x[2]^2 + 3*x[3] + 2*x[4] -2)
@mapping(m, F3, 3*x[1]^2 + x[1]*x[2] + 2*x[2]^2 + 2*x[3] + 3*x[4] -1)
@mapping(m, F4, x[1]^2 + 3*x[2]^2 + 2*x[3] + 3*x[4] - 3)

@complementarity(m, F1, x[1])
@complementarity(m, F2, x[2])
@complementarity(m, F3, x[3])
@complementarity(m, F4, x[4])

set_start_value(x[1], 1.25)
set_start_value(x[2], 0.)
set_start_value(x[3], 0.)
set_start_value(x[4], 0.5)

status = solveMCP(m, solver=:NLsolve)

z = result_value.(x)



# ----------
using JuMP, Ipopt, Complementarity
m = Model(Ipopt.Optimizer)
@variable(m, x>=0)
@NLobjective(m, Min, x^3)
@complements(m, 0 <= x+2,   x >= 0)
@constraint(m, x>=-10)
optimize!(m)



using JuMP, Ipopt, Complementarity
m = Model(Ipopt.Optimizer)
@variable(m, x[1:2])
@variable(m, lam[1:2] >= 0)
@objective(m, Min, x'*[1 0; 0 1]*x + [1;1]'*x - lam'*(x-[2;2]))
@complements(m, 0 <= x[1]-2, lam[1] >= 0)
@complements(m, 0 <= x[2]-2, lam[2] >= 0)
@constraint(m, x[1]==3)
optimize!(m)


# -------------


