using JuMP, PATHSolver, LinearAlgebra
# using Infiltrator
# the cost for the i-th player: x'*Qi*x + qi'*x
Q1 = [2 1 0; 
	  1 4 0;
	  0 0 1]
Q2 = [3 0 0;
	  0 2 1;
	  0 1 4]
Q3 = [3 0 1;
      0 6 0;
      1 0 4]

q1 = -[10;5;10]
q2 = -[20;20;10]
q3 = -[10;5;20]
mu=1


function solve_MCP_problems(x,lam)
	model = Model(PATHSolver.Optimizer)
	set_optimizer_attribute(model, "output", "no")
	@variable(model, w[1:3] >= 0)
	@constraint(model, (4*w[1] +2*w[2] +q1[1] +lam[1] +2*mu*(w[1]-x[1])) ⟂ w[1])
	@constraint(model, (4*w[2] +2*w[3] +q2[2] +lam[2] +2*mu*(w[2]-x[2])) ⟂ w[2])
	@constraint(model, (8*w[3] +2*w[1] +q3[3] +lam[3] +2*mu*(w[3]-x[3])) ⟂ w[3])
	optimize!(model)
	value.(w)
	return value.(w), solution_summary(model)
end

function centralized_solver()
	model = Model(PATHSolver.Optimizer)
	set_optimizer_attribute(model, "output", "no")
	@variable(model, w[1:3] >= 0)
	
	@constraint(model, (4*w[1] + 2*w[2] + q1[1]) ⟂ w[1])
	@constraint(model, (4*w[2] + 2*w[3] + q2[2]) ⟂ w[2])
	@constraint(model, (8*w[3] + 2*w[1] + q3[3]) ⟂ w[3])
	optimize!(model)
	value.(w)
	return value.(w), solution_summary(model)
end


K = 20
w = ones(3,K)
x = ones(3,K)
lam = ones(3,K)
mu=1


# test centralized solver
ww, ss = centralized_solver()

# test MCP solver
new_w, current_solution = solve_MCP_problems([1;1;1],[1,1,1])



for k in 1:K-1
	w[:,k+1],_ = solve_MCP_problems(x[:,k], lam[:,k])
	x[:,k+1] = 1/mu*lam[:,k] + w[:,k+1]
	lam[:,k+1] = lam[:,k] + mu*(w[:,k+1]-x[:,k+1])
end

"""
Consider two problems: 

min_x        x'*Q*x + q'*x

and

min_{x,w}    w'*Q*w + q'*w
subject to   w = x

Lagrangian:  L(x,w) = w'*Q*w + q'*w + y'*(w-x)

∇_w L = Q*w + q + y =0
∇_x L = -y =0





"""

