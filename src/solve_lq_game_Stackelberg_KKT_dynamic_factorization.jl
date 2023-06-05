using Infiltrator
# We only have two players 
function solve_lq_game_Stackelberg_KKT_dynamic_factorization!(strategies, g::LQGame, x0)
    # extract control and input dimensions
    nx, nu, m, T = n_states(g), n_controls(g), length(uindex(g)[1]), horizon(g)-1
    # m is the input size of agent i, and T is the horizon.
    num_player = n_players(g) # number of player
    @assert length(num_player) != 2
    M_size = nu+nx*num_player+m+nx# + (num_player-1)*nu 
    # size of the M matrix for each time instant, will be used to define KKT matrix

    new_M_size = nu+nx+nx*num_player+m + (num_player-1)*nu
    # initialize some intermidiate variables in KKT conditions
    
    # M_next, N_next, n_next = zeros(M_size, M_size), zeros(M_size, nx), zeros(M_size)
    Mₜ,     Nₜ,      nₜ     = zeros(M_size, M_size), zeros(M_size, nx), zeros(M_size)
    λ = zeros(T*nx*num_player)
    η = zeros((T)*nu)
    ψ = zeros(T*m)
    
    record_old_Mₜ_size = M_size
    K, k = zeros(M_size, nx), zeros(M_size)
    Π² = zeros(m, nu)
    π̂², π̌² = zeros(m, nx), zeros(m, m) # π²'s dependence on x and u1
    
    Π_next = zeros(nu, nx*num_player)
    π¹_next = zeros(m, nx)

    Aₜ₊₁ = zeros(nx, nx)
    Bₜ₊₁ = zeros(nx, nu)
    K_lambda_next = zeros(2*nx, nx)
    k_lambda_next = zeros(2*nx)
    K_multipliers = zeros((T-1)*(2*nx+nu+m) + (2*nx+m), nx)
    k_multipliers = zeros((T-1)*(2*nx+nu+m) + (2*nx+m), 1)
    for t in T:-1:1 # work in backwards to construct the KKT constraint matrix
        dyn, cost = dynamics(g)[t], player_costs(g)[t]
        next_cost = player_costs(g)[t+1]
        # convenience shorthands for the relevant quantities
        A, B = dyn.A, dyn.B
        Âₜ₊₁, B̂ₜ = zeros(nx*num_player, nx*num_player), zeros(nx*num_player, nu)
        Rₜ, Qₜ₊₁ = zeros(nu, nu), zeros(nx*num_player, nx)
        rₜ, qₜ₊₁ = zeros(nu), zeros(nx*num_player)
        B̃ₜ₊₁ = zeros(num_player*nx, (num_player-1)*nu) 
        # R̃ₜ = zeros((num_player-1)*nu, nu) 
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
            M2[1:m, m+1:m+nx] = transpose(B[:,m+1:nu]) # B̂ₜ
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
            
            
            # M_next, N_next, n_next = Mₜ, Nₜ, nₜ
            K, k = -Mₜ\Nₜ, -Mₜ\nₜ
            π¹_next = K[1:m,:]
            Π_next[1:m,1:nx] = π̂² # πₜ₊₁
            Π_next[m+1:nu, nx+1:2*nx] = π¹_next
            strategies[t] = AffineStrategy(SMatrix{nu, nx}(-K[1:nu,:]), SVector{nu}(-k[1:nu])) 
            Aₜ₊₁ = A
            Bₜ₊₁ = B
            K_lambda_next = -K[nu+1:nu+nx+nx, :]
            k_lambda_next = -k[nu+1:nu+nx+nx, :]
            K_multipliers[(t-1)*(2*nx+nu+m)+1:end, :] = K[nu+1:nu+2*nx+m, :]
            k_multipliers[(t-1)*(2*nx+nu+m)+1:end] = k[nu+1:nu+2*nx+m, :]
            # @infiltrate
        else
            # when t < T, we first solve for the follower
            for (ii, udxᵢ) in enumerate(uindex(g))
                Âₜ₊₁[(ii-1)*nx+1:ii*nx, (ii-1)*nx+1:ii*nx] = Aₜ₊₁
                B̂ₜ[(ii-1)*nx+1:ii*nx, (ii-1)*m+1:ii*m] = B[:,udxᵢ]
                Qₜ₊₁[(ii-1)*nx+1:ii*nx,:], Rₜ[udxᵢ,:] = next_cost[ii].Q, cost[ii].R[udxᵢ,:]
                qₜ₊₁[(ii-1)*nx+1:ii*nx], rₜ[udxᵢ] = next_cost[ii].l, cost[ii].r[udxᵢ]
                udxᵢ_complement = setdiff(1:1:nu, udxᵢ)
                
                B̃ₜ₊₁[(ii-1)*nx+1:ii*nx, (ii-1)*(num_player-1)*m+1:ii*(num_player-1)*m] = Bₜ₊₁[:,udxᵢ_complement] # 
                # R̃ₜ[(ii-1)*(num_player-1)*m+1:ii*(num_player-1)*m, :] = cost[ii].R[udxᵢ_complement,:] # not used actually 
            end
            # size_M_next = size(M_next, 1)
            
            # below is the construction of the KKT matrix for follower
            # we don't need to compute the offset term here
            tmp_Pₜ¹ = inv([cost[2].R[m+1:2*m, m+1:2*m]  transpose(B[:,m+1:nu])  zeros(m,m)  zeros(m,nx);
                      -B[:,m+1:nu]  zeros(nx, nx+m)  I(nx);
                      zeros(nx, m)  -I(nx)  π¹_next'  cost[2].Q-Aₜ₊₁'*K_lambda_next[nx+1:end,:];
                      zeros(m, m+nx)  -I(m)  -Bₜ₊₁[:,1:m]'*K_lambda_next[nx+1:end,:]])
            
            Nₜ² = zeros(size(tmp_Pₜ¹,1), nx+m)
            Nₜ²[m+1:m+nx, :] = [-A -B[:,1:m]]
            inv_tmp_Pₜ¹_Nₜ² = -tmp_Pₜ¹*Nₜ²
            π̂², π̌² = inv_tmp_Pₜ¹_Nₜ²[1:m, 1:nx], inv_tmp_Pₜ¹_Nₜ²[1:m, nx+1:nx+m]
            Π²[:,1:m] = π̌²
            # @infiltrate
            # we then solve the leader
            
            # Nₜ = zeros(new_M_size, nx)
            # Nₜ[nu+1:nu+nx,:] = -A # Nₜ is defined here!
            # @infiltrate
            Pₜ¹ = inv([Rₜ  B̂ₜ'  zeros(nu, nu)  Π²'  zeros(nu, nx);
                -B  zeros(nx, 2*nx+nu+m)  I(nx);
                zeros(2*nx, nu)  -I(2*nx)  Π_next'  zeros(2*nx,m)   Qₜ₊₁-Âₜ₊₁'*K_lambda_next;
                zeros(m, nu)  B̃²'  zeros(m, nu)  -I(m)  zeros(m, nx);
                zeros(nu,nu)  zeros(nu, 2*nx)  -I(nu)  zeros(nu, m)   -B̃ₜ₊₁'*K_lambda_next ])
            Pₜ²nₜ₊₁ = -Pₜ¹* [zeros(nu+nx, 1); Âₜ₊₁'*k_lambda_next; zeros(m,1); B̃ₜ₊₁'*k_lambda_next ]
           
            K, k = -Pₜ¹*[zeros(nu,nx);-A;zeros(new_M_size-nu-nx,nx)], -Pₜ¹*[rₜ; zeros(nx,1); qₜ₊₁; zeros(m+nu,1)] - Pₜ²nₜ₊₁
            π¹_next = K[1:m,:] # update π¹_next
            Π_next[1:m,1:nx] = π̂²
            Π_next[m+1:nu, nx+1:2*nx] = π¹_next
            # @infiltrate
            strategies[t] = AffineStrategy(SMatrix{nu, nx}(-K[1:nu,:]), SVector{nu}(-k[1:nu]))
            Aₜ₊₁ = A
            Bₜ₊₁ = B
            K_lambda_next = -K[nu+1:nu+nx+nx, :]
            k_lambda_next = -k[nu+1:nu+nx+nx, :]
            K_multipliers[(t-1)*(2*nx+nu+m)+1:t*(2*nx+nu+m), :] = K[nu+1:nu+2*nx+nu+m, :]
            k_multipliers[(t-1)*(2*nx+nu+m)+1:t*(2*nx+nu+m)] = k[nu+1:nu+2*nx+nu+m, :]
            # @infiltrate
        end
    end
    # solution = K*x0+k
    x=x0
    for t in 1:1:T
        if t == T
            λ[(t-1)*nx*num_player+1:t*nx*num_player] =  K_multipliers[(t-1)*(2*nx+nu+m)+1:end-m, :]*x + k_multipliers[(t-1)*(2*nx+nu+m)+1:end-m]
            ψ[(t-1)*m+1:t*m] =                          K_multipliers[end-m+1:end, :]*x + k_multipliers[end-m+1:end]
        else
            λ[(t-1)*nx*num_player+1:t*nx*num_player] =  K_multipliers[ (t-1)*(2*nx+nu+m)+1:(t-1)*(2*nx+nu+m)+2*nx, : ]*x + k_multipliers[(t-1)*(2*nx+nu+m)+1:(t-1)*(2*nx+nu+m)+2*nx, : ]
            η[(t-1)*nu+1:t*nu] = K_multipliers[(t-1)*(2*nx+nu+m)+2*nx+1:(t-1)*(2*nx+nu+m)+2*nx+nu, :]*x + k_multipliers[ (t-1)*(2*nx+nu+m)+2*nx+1:(t-1)*(2*nx+nu+m)+2*nx+nu, : ]
            # @infiltrate
            ψ[(t-1)*m+1:t*m] = K_multipliers[(t-1)*(2*nx+nu+m)+2*nx+nu+1:t*(2*nx+nu+m), : ]*x + k_multipliers[(t-1)*(2*nx+nu+m)+2*nx+nu+1:t*(2*nx+nu+m)]
            x = dynamics(g)[t].A*x - dynamics(g)[t].B*(strategies[t].P * x + strategies[t].α) # update x
        end
    end
    # @infiltrate
    # TODO: why the λ is twice of the matlab version? offset term is not consistent! 
    return λ, η, ψ
end