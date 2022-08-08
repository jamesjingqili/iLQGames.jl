function solve_lq_game_OLNE!(strategies, g::LQGame)
    # extract control and input dimensions
    nx, nu, T, N = n_states(g), n_controls(g), horizon(g), n_players(g)
    # N is the number of players
    Mₜ = [player_costs(g)[T][ii].Q for ii in 1:N]
    mₜ = [player_costs(g)[T][ii].l for ii in 1:N]
    for kk in T:-1:1
        dyn = dynamics(g)[kk]
        cost = player_costs(g)[kk]
        # convenience shorthands for the relevant quantities
        A, B = dyn.A, dyn. B
        Λₜ, ηₜ = I(nx), zeros(nx)
        M_next, m_next = Mₜ, mₜ
        P = zeros(nu, nx)
        α = zeros(nu)

        for (ii, udxᵢ) in enumerate(uindex(g))
            inv_Rₜ = inv(cost[ii].R[udxᵢ,udxᵢ])
            Λₜ +=  B[:, udxᵢ]*inv_Rₜ*B[:, udxᵢ]'*M_next[ii]
            ηₜ -=  B[:, udxᵢ]*inv_Rₜ*(B[:, udxᵢ]'*m_next[ii] + cost[ii].r[udxᵢ])
        end
        for (ii, udxᵢ) in enumerate(uindex(g))
            pinv_Λₜ = inv(Λₜ)
            mₜ[ii] = cost[ii].l + A'*(m_next[ii] + M_next[ii]*pinv_Λₜ*ηₜ)
            Mₜ[ii] = cost[ii].Q + A'*M_next[ii]*pinv_Λₜ*A
            
            inv_Rₜ = inv(cost[ii].R[udxᵢ,udxᵢ])
            P[udxᵢ,:] = - inv_Rₜ*B[:,udxᵢ]'*(M_next[ii]*pinv_Λₜ*A)
            α[udxᵢ] = - inv_Rₜ*B[:,udxᵢ]'*(M_next[ii]*pinv_Λₜ*ηₜ+m_next[ii]) - inv_Rₜ*cost[ii].r[udxᵢ]
        end
        strategies[kk] = AffineStrategy(SMatrix{nu,nx}(-P), SVector{nu}(-α)) 
    end
end