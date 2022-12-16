using JuMP, PATHSolver, LinearAlgebra
using Infiltrator
# the cost for the i-th player: x'*Qi*x + qi'*x
Q1 = [2 1; 
	1 4]
Q2 = [3 1;
	1 2]

q1 = -[10;5]
q2 = -[20;20]

K = 10
w = ones(2,K)
x = zeros(2,K)
lam = zeros(2,K)
mu=1

function solve_MCP_problems(x,lam)
	model = Model(PATHSolver.Optimizer)
	set_optimizer_attribute(model, "output", "no")
	@variable(model, w[1:2] >= 0)
	@constraint(model, (4*x[1] + 2*x[2] + q1[1] + lam[2] + 2*mu*(w[2]-x[2]))⟂w[1])
	@constraint(model, (4*x[2] + 2*x[1] + q2[2] + lam[2] + 2*mu*(w[2]-x[2]))⟂w[2])
	@constraint(model, (8*x[3] + 2*x[1] + q3[3] + lam[3] + 2*mu*(w[3]-x[3]))⟂w[3])
	optimize!(model)
	value.(w)
	return value.(w), solution_summary(model)
end
# test MCP solver
new_w, current_solution = solve_MCP_problems([1;1;1],[1,1,1])


for k in 1:K-1
	w[:,k+1],_ = solve_MCP_problems(x[:,k], lam[:,k])
	x[:,k+1] = 1/mu*lam[:,k] - w[:,k+1]
	lam[:,k+1] = lam[:,k] + mu*(w[:,k+1]-x[:,k+1])
end

