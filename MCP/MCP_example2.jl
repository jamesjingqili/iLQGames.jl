using JuMP, PATHSolver

# f_i(x) = 1/2*x'*M1*x + q1'*x
# ∇f_1(x) = 2*x[1] + x[2] + q1[1]
M1 = [1 0.5; 0.5 1]
q1 = -[2;2]

# ∇f_2(x) = 8*x[1] + x[1] + q2[1]
M2 = [1 0.5; 0.5 4]
q2 = -[3; 1]

model = Model(PATHSolver.Optimizer)
set_optimizer_attribute(model, "output", "no")
@variable(model, x[1:2] >= 0)
@constraint(model, 2*x[1]+x[2]+q1[1] ⟂ x[1])
@constraint(model, 8*x[2]+q2[2] ⟂ x[2])
optimize!(model)
value.(x)
solution_summary(model)



# ----------
using Complementarity, JuMP
m = MCPModel()
@variable(m, x[1:2] >= 0)
@mapping(m, F, x[1]+2)
@mapping(m, F1, x[2] + 3)
@complementarity(m, F, x[1])
@complementarity(m, F1, x[2])
status = solveMCP(m)
@show result_value.(x)
