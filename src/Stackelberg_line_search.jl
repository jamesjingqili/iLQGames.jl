
function Stackelberg_KKT_residual(λ::Vector, η::Vector, ψ::Vector, current_op::SystemTrajectory, g::LQGame, x0::SVector)
    # current_op is the trajectory under the strategy evaluated now
    # g is the linear quadratic approximation for the trajectory under the strategy evaluated now
    # extract control and input dimensions
    nx, nu, m, T = n_states(g), n_controls(g), length(uindex(g)[1]), horizon(g)-1
    # m is the input size of agent i, and T is the horizon.
    num_player = n_players(g) # number of player
    @assert length(num_player) != 2
    M_size = nu+nx*num_player+m+nx# + (num_player-1)*nu 
    new_M_size = nu+nx+nx*num_player+m + (num_player-1)*nu
    M_next, N_next, n_next = zeros(M_size, M_size), zeros(M_size, nx), zeros(M_size)
    Mₜ,     Nₜ,      nₜ     = zeros(M_size, M_size), zeros(M_size, nx), zeros(M_size)
    record_old_Mₜ_size = M_size
    K, k = zeros(M_size, nx), zeros(M_size)
    Π² = zeros(m, nu)
    π̂², π̌² = zeros(m, nx), zeros(m, m) # π²'s dependence on x and u1
    Π_next = zeros(nu, nx*num_player)
    π¹_next = zeros(m, nx)
    solution_vector = []
    Aₜ₊₁ = zeros(nx,nx)
    Bₜ₊₁ = zeros(nx,nu)
    K_lambda_next = zeros(2*nx, nx)
    k_lambda_next = zeros(2*nx)
    
    for t in T:-1:1 # work in backwards to construct the KKT constraint matrix
        dyn, cost = dynamics(g)[t], player_costs(g)[t]
        next_cost = player_costs(g)[t+1]
        # convenience shorthands for the relevant quantities
        A, B = dyn.A, dyn.B
        Âₜ₊₁, B̂ₜ = zeros(nx*num_player, nx*num_player), zeros(nx*num_player, nu)
        Rₜ, Qₜ₊₁ = zeros(nu, nu), zeros(nx*num_player, nx)
        rₜ, qₜ₊₁ = zeros(nu), zeros(nx*num_player)
        B̃ₜ₊₁, R̃ₜ₊₁ = zeros(num_player*nx, (num_player-1)*nu), zeros((num_player-1)*nu, nu)
        Πₜ = zeros((num_player-1)*nu, num_player*nx)
        B̃² = [B[:,m+1:nu]; zeros(nx, m)]
        if t == T
            # We first solve for the follower
            for (ii, udxᵢ) in enumerate(uindex(g))
                B̂ₜ[(ii-1)*nx+1:ii*nx, (ii-1)*m+1:ii*m] = B[:,udxᵢ]
                Qₜ₊₁[(ii-1)*nx+1:ii*nx,:], Rₜ[udxᵢ,:] = next_cost[ii].Q, cost[ii].R[udxᵢ,:]
                qₜ₊₁[(ii-1)*nx+1:ii*nx], rₜ[udxᵢ] = next_cost[ii].l, cost[ii].r[udxᵢ]
            end
            N2 = zeros(m+nx+nx, nx+m)
            M2 = zeros(m+nx+nx, m+nx+nx)
            N2[m+1:m+nx, :] = [-A -B[:,1:m]]

            M2[1:m, 1:m] = cost[2].R[m+1:2*m, m+1:2*m]
            M2[1:m, m+1:m+nx] = transpose(B[:,m+1:nu])
            M2[m+1:m+nx, 1:m] = -B[:,m+1:nu]
            M2[m+1:m+nx, m+nx+1:m+nx+nx] = I(nx)
            M2[m+nx+1:m+nx+nx, m+1:m+nx] = -I(nx)
            M2[m+nx+1:m+nx+nx, m+nx+1:m+nx+nx] = cost[2].Q
            inv_M2_N2 = -M2\N2
            π̂², π̌² = inv_M2_N2[1:m, 1:nx], inv_M2_N2[1:m, nx+1:nx+m]
            Π²[:,1:m] = π̌²
            # we then solve for the leader
            Nₜ[nu+1:nu+nx,:] = -A
            Mₜ[1:nu, 1:nu] = Rₜ
            Mₜ[1:nu, nu+1:nu+nx*num_player] = transpose(B̂ₜ)
            Mₜ[nu+1:nu+nx, 1:nu] = -B
            Mₜ[nu+1:nu+nx, M_size-nx+1:M_size] = I(nx)
            Mₜ[nu+nx+1:M_size-m, nu+1:nu+nx*num_player] = -I(nx*num_player)
            Mₜ[nu+nx+1:M_size-m, M_size-nx+1:M_size] = Qₜ₊₁
            Mₜ[M_size-m+1:M_size, nu+1:nu+2*nx] = B̃²'
            Mₜ[M_size-m+1:M_size, nu+2*nx+1:nu+2*nx+m] = -I(m)
            nₜ[1:nu], nₜ[M_size-nx*num_player+1-m:M_size-m] = rₜ, qₜ₊₁
            
            M_next, N_next, n_next = Mₜ, Nₜ, nₜ
            K, k = -Mₜ\Nₜ, -Mₜ\nₜ
            π¹_next = K[1:m,:]
            Π_next[1:m,1:nx] = π̂²
            
            
            Π_next[m+1:nu, nx+1:2*nx] = π¹_next
            solution_vector = [current_op.u[t]; λ[end-2*nx+1:end]; ψ[end-m+1:end]; current_op.x[t+1]; solution_vector]
            Aₜ₊₁ = A
            Bₜ₊₁ = B
            K_lambda_next = -K[nu+1:nu+nx+nx, :]
            k_lambda_next = -k[nu+1:nu+nx+nx, :]
        else
            # when t < T, we first solve for the follower
            for (ii, udxᵢ) in enumerate(uindex(g))
                Âₜ₊₁[(ii-1)*nx+1:ii*nx, (ii-1)*nx+1:ii*nx] = Aₜ₊₁
                B̂ₜ[(ii-1)*nx+1:ii*nx, (ii-1)*m+1:ii*m] = B[:,udxᵢ]
                Qₜ₊₁[(ii-1)*nx+1:ii*nx,:], Rₜ[udxᵢ,:] = next_cost[ii].Q, cost[ii].R[udxᵢ,:]
                qₜ₊₁[(ii-1)*nx+1:ii*nx], rₜ[udxᵢ] = next_cost[ii].l, cost[ii].r[udxᵢ]
                udxᵢ_complement = setdiff(1:1:nu, udxᵢ)
                
                B̃ₜ₊₁[(ii-1)*nx+1:ii*nx, (ii-1)*(num_player-1)*m+1:ii*(num_player-1)*m] = Bₜ₊₁[:,udxᵢ_complement] # 
                R̃ₜ₊₁[(ii-1)*(num_player-1)*m+1:ii*(num_player-1)*m, :] = cost[ii].R[udxᵢ_complement,:] # 
                Πₜ[(ii-1)*(num_player-1)*m+1:ii*(num_player-1)*m, (ii-1)*nx+1:ii*nx] = K[udxᵢ_complement, :] #
            end
            size_M_next = size(M_next, 1)
            
            # tmp_D1 = [cost[2].R[m+1:2*m, m+1:2*m]  transpose(B[:,m+1:nu])  zeros(m,m)  zeros(m,nx);
            #           -B[:,m+1:nu]  zeros(nx, nx+m)  I(nx);
            #           zeros(nx, m)  -I(nx)  π¹_next'  cost[2].Q;
            #           zeros(m, m+nx)  -I(m)  zeros(m, nx)]
            
            # tmp_D2 = [zeros(m+nx, size_M_next); 
            #         zeros(nx, nu+nx) A' zeros(nx, size_M_next-nu-2*nx);
            #         zeros(m, nu+nx)  B[:,1:m]' zeros(m, size_M_next-nu-2*nx) ]

            # M2 = [tmp_D1  tmp_D2; zeros(size_M_next, size(tmp_D1, 2)-nx)  Nₜ  Mₜ]

            # N2 = zeros(m + nx + nx + m + size_M_next, nx + m)
            # N2[m+1:m+nx, :] = [-A -B[:,1:m]]
            # inv_M2_N2 = -inv(M2)*N2
            # π̂², π̌² = inv_M2_N2[1:m, 1:nx], inv_M2_N2[1:m, nx+1:nx+m]
            # Π²[:,1:m] = π̌²
            tmp_Pₜ¹ = inv([cost[2].R[m+1:2*m, m+1:2*m]  transpose(B[:,m+1:nu])  zeros(m,m)  zeros(m,nx);
                      -B[:,m+1:nu]  zeros(nx, nx+m)  I(nx);
                      zeros(nx, m)  -I(nx)  π¹_next'  cost[2].Q-Aₜ₊₁'*K_lambda_next[nx+1:end,:];
                      zeros(m, m+nx)  -I(m)  -Bₜ₊₁[:,1:m]'*K_lambda_next[nx+1:end,:]])
            
            Nₜ² = zeros(size(tmp_Pₜ¹,1), nx+m)
            Nₜ²[m+1:m+nx, :] = [-A -B[:,1:m]]
            inv_tmp_Pₜ¹_Nₜ² = -tmp_Pₜ¹*Nₜ²
            π̂², π̌² = inv_tmp_Pₜ¹_Nₜ²[1:m, 1:nx], inv_tmp_Pₜ¹_Nₜ²[1:m, nx+1:nx+m]
            Π²[:,1:m] = π̌²
            
            

            # we then solve the leader
            Mₜ = zeros(new_M_size+record_old_Mₜ_size, new_M_size+record_old_Mₜ_size) 
            Nₜ = zeros(new_M_size+record_old_Mₜ_size, nx)
            nₜ = zeros(new_M_size+record_old_Mₜ_size)
            Mₜ[end-record_old_Mₜ_size+1:end, end-record_old_Mₜ_size+1:end] = M_next
            Mₜ[end-record_old_Mₜ_size+1:end, nu+nx*num_player+(num_player-1)*nu+1:nu+nx*num_player+(num_player-1)*nu+nx] = N_next
            nₜ[end-record_old_Mₜ_size+1:end] = n_next

            Nₜ[nu+1:nu+nx,:] = -A # Nₜ is defined here!
            D1 = [Rₜ  B̂ₜ'  zeros(nu, nu)  Π²'  zeros(nu, nx);
                -B  zeros(nx, 2*nx+nu+m)  I(nx);
                zeros(2*nx, nu)  -I(2*nx)  Π_next'  zeros(2*nx,m)  Qₜ₊₁;
                zeros(m, nu)  B̃²'  zeros(m, nu)  -I(m)  zeros(m, nx);
                zeros(nu,nu)  zeros(nu, 2*nx)  -I(nu)  zeros(nu, m+nx)]
            D2 = [zeros(nu+nx, size_M_next);
                zeros(2*nx, nu) Âₜ₊₁' zeros(2*nx, size_M_next-nu-2*nx);
                zeros(m, size_M_next);
                zeros(nu, nu) B̃ₜ₊₁' zeros(nu, size_M_next-2*nx-nu)]
            Mₜ = [D1  D2; zeros(size_M_next, size(D1, 2)-nx)  N_next  M_next]

            nₜ[1:nu], nₜ[nu+nx+1:nu+nx+num_player*nx] = rₜ, qₜ₊₁
            

            M_next, N_next, n_next = Mₜ, Nₜ, nₜ
            # K, k = -Mₜ\Nₜ, -Mₜ\nₜ
            Pₜ¹ = inv([Rₜ  B̂ₜ'  zeros(nu, nu)  Π²'  zeros(nu, nx);
                -B  zeros(nx, 2*nx+nu+m)  I(nx);
                zeros(2*nx, nu)  -I(2*nx)  Π_next'  zeros(2*nx,m)   Qₜ₊₁-Âₜ₊₁'*K_lambda_next;
                zeros(m, nu)  B̃²'  zeros(m, nu)  -I(m)  zeros(m, nx);
                zeros(nu,nu)  zeros(nu, 2*nx)  -I(nu)  zeros(nu, m)   -B̃ₜ₊₁'*K_lambda_next ])
            Pₜ²nₜ₊₁ = -Pₜ¹* [zeros(nu+nx, 1); Âₜ₊₁'*k_lambda_next; zeros(m,1); B̃ₜ₊₁'*k_lambda_next ]
            K, k = -Pₜ¹*[zeros(nu,nx);-A;zeros(new_M_size-nu-nx,nx)], -Pₜ¹*[rₜ; zeros(nx,1); qₜ₊₁; zeros(m+nu,1)] - Pₜ²nₜ₊₁
            
            record_old_Mₜ_size += new_M_size

            π¹_next = K[1:m,:] # update π¹_next
            Π_next[1:m,1:nx] = π̂²
            Π_next[m+1:nu, nx+1:2*nx] = π¹_next
            solution_vector = [current_op.u[t]; 
                                λ[(t-1)*2*nx+1:t*2*nx]; 
                                η[(t-1)*nu+1:t*nu]; 
                                ψ[(t-1)*m+1:t*m]; 
                                current_op.x[t+1]; 
                                solution_vector]
            Aₜ₊₁ = A
            Bₜ₊₁ = B
            K_lambda_next = -K[nu+1:nu+nx+nx, :]
            k_lambda_next = -k[nu+1:nu+nx+nx, :]
        end
    end
    loss = norm(Mₜ*solution_vector + Nₜ*x0+nₜ, 2)
    return loss   
end





function Stackelberg_KKT_line_search!(last_KKT_residual, λ::Vector, η::Vector, ψ::Vector, last_λ::Vector, last_η::Vector, last_ψ::Vector,
    current_strategy::SizedVector, last_strategy::SizedVector,
    current_op::SystemTrajectory, last_op::SystemTrajectory,
    cs::ControlSystem, solver::iLQSolver, g::GeneralGame, current_lqg_approx::LQGame, x0::SVector) #2
    # return the current_op as the new trajectory, the current_strategy as the γₖ + α(γ_next - γₖ), and the last_KKT_residual
    # question: given a line-searched operating point, how can we retrive the policy achieving that trajectory?
    α = 1.0
    Δ_strategy = current_strategy-last_strategy
    Δ_λ = λ - last_λ
    Δ_η = η - last_η
    Δ_ψ = ψ - last_ψ
    for iter in 1:solver.max_scale_backtrack
        trajectory!(current_op, cs, last_strategy + α*Δ_strategy, last_op, x0, solver.max_elwise_diff_step)
        lq_approximation!(current_lqg_approx, solver, g, current_op)
        current_loss = Stackelberg_KKT_residual(last_λ+α*Δ_λ, last_η+α*Δ_η, last_ψ+α*Δ_ψ, current_op, current_lqg_approx, x0)
        # @infiltrate
        if current_loss < last_KKT_residual
            @infiltrate
            # current_strategy = last_strategy + α*Δ_strategy
            last_KKT_residual = copy(current_loss)
            # println("KKT residual is ",last_KKT_residual)
            println("Line Search finished with α = ", α, " and KKT residual is ", last_KKT_residual)
            @infiltrate
            return true, current_op, last_KKT_residual, α, Δ_strategy, Δ_λ, Δ_η, Δ_ψ
            # println("α is ", α)
            break
        end
        α = α * 0.5
    end
    # println("Current α is ",α)
    # @warn "Line Search failed."
    println("line search failed but KKT residual is ",last_KKT_residual)
    return true, current_op, last_KKT_residual, α, Δ_strategy, Δ_λ, Δ_η, Δ_ψ
end


