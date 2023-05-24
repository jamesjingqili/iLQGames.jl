function Stackelberg_KKT_residual_old(λ::Vector, η::Vector, ψ::Vector, current_op::SystemTrajectory, g::LQGame, x0::SVector) #1    
    # current_op is the trajectory under the strategy evaluated now
    # g is the linear quadratic approximation for the trajectory under the strategy evaluated now
    # extract control and input dimensions
    nx, nu, m, T = n_states(g), n_controls(g), length(uindex(g)[1]), horizon(g)
    # m is the input size of agent i, and T is the horizon.
    num_player = n_players(g) # number of player
    @assert length(num_player) != 2
    M_size = nu+nx*num_player+m+nx# + (num_player-1)*nu 
    # size of the M matrix for each time instant, will be used to define KKT matrix

    new_M_size = nu+nx+nx*num_player+m + (num_player-1)*nu
    # initialize some intermidiate variables in KKT conditions
    
    M_next, N_next, n_next = zeros(M_size, M_size), zeros(M_size, nx), zeros(M_size)
    Mₜ,     Nₜ,      nₜ     = zeros(M_size, M_size), zeros(M_size, nx), zeros(M_size)
    
    record_old_Mₜ_size = M_size
    K, k = zeros(M_size, nx), zeros(M_size)
    Π² = zeros(m, nu)
    π̂², π̌² = zeros(m, nx), zeros(m, m) # π²'s dependence on x and u1
    
    Π_next = zeros(nu, nx*num_player)
    π¹_next = zeros(m, nx)

    solution_vector = []
    for t in T:-1:1 # work in backwards to construct the KKT constraint matrix
        dyn, cost = dynamics(g)[t], player_costs(g)[t]
        # convenience shorthands for the relevant quantities
        A, B = dyn.A, dyn.B
        Âₜ, B̂ₜ = zeros(nx*num_player, nx*num_player), zeros(nx*num_player, nu)
        Rₜ, Qₜ = zeros(nu, nu), zeros(nx*num_player, nx)
        rₜ, qₜ = zeros(nu), zeros(nx*num_player)
        B̃ₜ, R̃ₜ = zeros(num_player*nx, (num_player-1)*nu), zeros((num_player-1)*nu, nu)
        Πₜ = zeros((num_player-1)*nu, num_player*nx)
        
        B̃² = [B[:,m+1:nu]; zeros(nx, m)]
        
        if t == T
            # We first solve for the follower
            for (ii, udxᵢ) in enumerate(uindex(g))
                B̂ₜ[(ii-1)*nx+1:ii*nx, (ii-1)*m+1:ii*m] = B[:,udxᵢ]
                Qₜ[(ii-1)*nx+1:ii*nx,:], Rₜ[udxᵢ,:] = cost[ii].Q, cost[ii].R[udxᵢ,:]
                qₜ[(ii-1)*nx+1:ii*nx], rₜ[udxᵢ] = cost[ii].l, cost[ii].r[udxᵢ]
            end
            N2 = zeros(m+nx+nx, nx+m)
            M2 = zeros(m+nx+nx, m+nx+nx)
            # @infiltrate
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
            Mₜ[nu+nx+1:M_size-m, M_size-nx+1:M_size] = Qₜ
            Mₜ[M_size-m+1:M_size, nu+1:nu+2*nx] = B̃²'
            Mₜ[M_size-m+1:M_size, nu+2*nx+1:nu+2*nx+m] = -I(m)
            nₜ[1:nu], nₜ[M_size-nx*num_player+1-m:M_size-m] = rₜ, qₜ
            
            
            M_next, N_next, n_next = Mₜ, Nₜ, nₜ
            K, k = -Mₜ\Nₜ, -Mₜ\nₜ
            π¹_next = K[1:m,:]
            Π_next[1:m,1:nx] = π̂²
            
            Π_next[m+1:nu, nx+1:2*nx] = π¹_next
            # strategies[t] = AffineStrategy(SMatrix{nu, nx}(-K[1:nu,:]), SVector{nu}(-k[1:nu]))     
            solution_vector = [current_op.u[t]; λ[end-2*nx+1:end]; ψ[end-m+1:end]; current_op.x[t]; solution_vector]
        else
            # when t < T, we first solve for the follower
            for (ii, udxᵢ) in enumerate(uindex(g))
                Âₜ[(ii-1)*nx+1:ii*nx, (ii-1)*nx+1:ii*nx] = A
                B̂ₜ[(ii-1)*nx+1:ii*nx, (ii-1)*m+1:ii*m] = B[:,udxᵢ]
                Qₜ[(ii-1)*nx+1:ii*nx,:], Rₜ[udxᵢ,:] = cost[ii].Q, cost[ii].R[udxᵢ,:]
                qₜ[(ii-1)*nx+1:ii*nx], rₜ[udxᵢ] = cost[ii].l, cost[ii].r[udxᵢ]
                udxᵢ_complement = setdiff(1:1:nu, udxᵢ)
                
                B̃ₜ[(ii-1)*nx+1:ii*nx, (ii-1)*(num_player-1)*m+1:ii*(num_player-1)*m] = B[:,udxᵢ_complement] # 
                R̃ₜ[(ii-1)*(num_player-1)*m+1:ii*(num_player-1)*m, :] = cost[ii].R[udxᵢ_complement,:] # 
                Πₜ[(ii-1)*(num_player-1)*m+1:ii*(num_player-1)*m, (ii-1)*nx+1:ii*nx] = K[udxᵢ_complement, :] #
            end
            size_M_next = size(M_next, 1)
            
            tmp_D1 = [cost[2].R[m+1:2*m, m+1:2*m]  transpose(B[:,m+1:nu])  zeros(m,m)  zeros(m,nx);
                      -B[:,m+1:nu]  zeros(nx, nx+m)  I(nx);
                      zeros(nx, m)  -I(nx)  π¹_next'  cost[2].Q;
                      zeros(m, m+nx)  -I(m)  zeros(m, nx)]
            
            tmp_D2 = [zeros(m+nx, size_M_next); 
                    zeros(nx, nu+nx) A' zeros(nx, size_M_next-nu-2*nx);
                    zeros(m, nu+nx)  B[:,1:m]' zeros(m, size_M_next-nu-2*nx) ]

            M2 = [tmp_D1  tmp_D2; zeros(size_M_next, size(tmp_D1, 2)-nx)  Nₜ  Mₜ]

            N2 = zeros(m + nx + nx + m + size_M_next, nx + m)
            N2[m+1:m+nx, :] = [-A -B[:,1:m]]
            inv_M2_N2 = -inv(M2)*N2
            π̂², π̌² = inv_M2_N2[1:m, 1:nx], inv_M2_N2[1:m, nx+1:nx+m]
            Π²[:,1:m] = π̌²


            # we then solve the leader
            Mₜ, Nₜ, nₜ = zeros(new_M_size+record_old_Mₜ_size, new_M_size+record_old_Mₜ_size), zeros(new_M_size+record_old_Mₜ_size, nx), zeros(new_M_size+record_old_Mₜ_size)

            Mₜ[end-record_old_Mₜ_size+1:end, end-record_old_Mₜ_size+1:end] = M_next
            Mₜ[end-record_old_Mₜ_size+1:end, nu+nx*num_player+(num_player-1)*nu+1:nu+nx*num_player+(num_player-1)*nu+nx] = N_next
            nₜ[end-record_old_Mₜ_size+1:end] = n_next

            Nₜ[nu+1:nu+nx,:] = -A # Nₜ is defined here!
            D1 = [Rₜ  B̂ₜ'  zeros(nu, nu)  Π²'  zeros(nu, nx);
                -B  zeros(nx, 2*nx+nu+m)  I(nx);
                zeros(2*nx, nu)  -I(2*nx)  Π_next'  zeros(2*nx,m)  Qₜ;
                zeros(m, nu)  B̃²'  zeros(m, nu)  -I(m)  zeros(m, nx);
                zeros(nu,nu)  zeros(nu, 2*nx)  -I(nu)  zeros(nu, m+nx)]
            D2 = [zeros(nu+nx, size_M_next);
                zeros(2*nx, nu) Âₜ' zeros(2*nx, size_M_next-nu-2*nx);
                zeros(m, size_M_next);
                zeros(nu, nu) B̃ₜ' zeros(nu, size_M_next-2*nx-nu)]
            Mₜ = [D1  D2; zeros(size_M_next, size(D1, 2)-nx)  N_next  M_next]

            nₜ[1:nu], nₜ[nu+nx+1:nu+nx+num_player*nx] = rₜ, qₜ
            
            M_next, N_next, n_next = Mₜ, Nₜ, nₜ
            K, k = -Mₜ\Nₜ, -Mₜ\nₜ
            record_old_Mₜ_size += new_M_size

            π¹_next = K[1:m,:] # update π¹_next
            Π_next[1:m,1:nx] = π̂²
            Π_next[m+1:nu, nx+1:2*nx] = π¹_next
            # strategies[t] = AffineStrategy(SMatrix{nu, nx}(-K[1:nu,:]), SVector{nu}(-k[1:nu]))
            solution_vector = [current_op.u[t]; λ[(t-1)*2*nx+1:t*2*nx]; η[(t-1)*nu+1:t*nu]; ψ[(t-1)*m+1:t*m]; current_op.x[t]; solution_vector]
        end
    end
    
    
    loss = norm(Mₜ*solution_vector + Nₜ*x0+nₜ, 2)
    return loss   
end
