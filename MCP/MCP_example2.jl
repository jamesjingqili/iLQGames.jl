using JuMP, PATHSolver

M1 = [1 0; 0 1]
q1 = -[2;2]

M2 = [1 0; 0 4]
q2 = -[3; 1]

model = Model(PATHSolver.Optimizer)
set_optimizer_attribute(model, "output", "no")
@variable(model, x[1:2] >= 0)
@constraint(model, M1 * x .+ q1 ⟂ x)
@constraint(model, M2 * x .+ q2 ⟂ x)

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
