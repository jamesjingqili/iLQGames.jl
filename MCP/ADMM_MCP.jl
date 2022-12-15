using JuMP, PATHSolver, LinearAlgebra

Q1 = diagm([1,0,0])
Q2 = diagm([0,1,0])
Q3 = diagm([0,0,1])

q1 = [-2;0;0]
q2 = [0;-2;0]
q3 = [0;0;-2]

K = 10
w = zeros(3,K)
x = zeros(3,K)
lam = zeros(3,K)

function solve_MCP_problems(x,w,lam)
	model = Model(PATHSolver.Optimizer)
	set_optimizer_attribute(model, "output", "no")
	@variable(model, x[1:2] >= 0)
	@constraint(model, M * x .+ q ⟂ x)
	optimize!(model)
	value.(x)
end

for k in range(K)

end


