using Infiltrator
"""
$(TYPEDSIGNATURES)

Solve a time-varying, finite horizon LQ-game to find closed-loop Stackelberg feedback
strategies for both players.

Assumes that dynamics are given by `xₖ₊₁ = Aₖ*xₖ + ∑ᵢBₖⁱ uₖⁱ`.

Important! Assumes that the first player is the Stackelberg leader. 
The control cost is quadratic only in each player's control input.


"""
function solve_lq_game_Stackelberg!(strategies, g::LQGame)
    # extract control and input dimensions
    nx = n_states(g)
    nu = n_controls(g)

    L1 = last(player_costs(g))[1].Q
    L2 = last(player_costs(g))[2].Q

    # working backwards in time to solve the dynamic program
    for kk in horizon(g):-1:1
        dyn = dynamics(g)[kk]
        cost = player_costs(g)[kk]
        # convenience shorthands for the relevant quantities
        A = dyn.A
        B = dyn.B
        B1 = B[:, uindex(g)[1]]
        B2 = B[:, uindex(g)[2]]
        R1 = cost[1].R[uindex(g)[1], uindex(g)[1]]
        R2 = cost[2].R[uindex(g)[2], uindex(g)[2]]
        r1 = cost[1].r[uindex(g)[1]]
        r2 = cost[2].r[uindex(g)[2]]
        l1 = cost[1].l
        l2 = cost[2].l

        M = -B1 + B2 * inv(R2 + B2' * L2 * B2) * B2' * L2 * B1
        N = A - B2 * inv(R2 + B2' * L2 * B2) * B2' * L2 * A
        S1 = inv(R1 + M' * L1 * M) * (-M' * L1 * N)
        S2 = inv(R2 + B2' * L2 * B2) * B2' * L2 * (A - B1 * S1)
        # @infiltrate
        F = A - B1 * S1 - B2 * S2
        L1 = S1' * R1 * S1 + F' * L1 * F + cost[1].Q
        L2 = S2' * R2 * S2 + F' * L2 * F + cost[2].Q

        l1 = -r1 * S1 + l1 * F + cost[1].q 
        l2 = -r2 * S2 + l2 * F + cost[2].q
        P = SMatrix{nu, nx}(vcat(S1[:, 1:nx], S2[:, 1:nx]))
        α = SVector{nu}(zeros(nu))
        strategies[kk] = AffineStrategy(P, α)
    end
    # @infiltrate
    # # extract control and input dimensions
    # nx = n_states(g)
    # nu = n_controls(g)
    # T = horizon(g)

    # # initializting the optimal cost to go representation for DP
    # # quadratic cost to go
    # # zeros_matrix = @SMatrix zeros(nx, nx)
    # # identity_matrix = SMatrix{nx,nx}(Matrix{Float64}(I, nx, nx))
    
    # zeros_matrix = zeros(nx, nx)
    # identity_matrix = Matrix{Float64}(I, nx, nx)
    
    # L1 = last(player_costs(g))[1].Q
    # L2 = last(player_costs(g))[2].Q
    
    # # Setup the S and Y matrix of the S * P = Y matrix equation
    # # As `nx` and `nu` are known at compile time and all operations below can be
    # # inlined, allocations can be eliminted by the compiler.
    
    # # working backwards in time to solve the dynamic program
    # for kk in horizon(g):-1:1
    #     dyn = dynamics(g)[kk]
    #     cost = player_costs(g)[kk]
        
    #     R1 = cost[1].R[uindex(g)[1], uindex(g)[1]]
    #     R2 = cost[2].R[uindex(g)[2], uindex(g)[2]]
    #     A = dyn.A
    #     B = dyn.B
    #     B1 = B[:, uindex(g)[1]]
    #     B2 = B[:, uindex(g)[2]]
        
    #     inv_term = inv(R2 + B2'*L2*B2)
    #     middle_term = B2 * inv_term * B2' * L2
    #     M = -B1 + middle_term * B1
    #     N = A   - middle_term * A
    #     S1 = inv(R1 + M' * L1 * M) * (-M' * L1 * N)
    #     S2 = inv_term * B2'*L2*(A - B1 * S1)
        
    #     F = A - B1 * S1 - B2 * S2 # TODO: should we lift the dimension of A and B?

    #     L1 = F'*L1*F + S1'*R1*S1 + cost[1].Q
    #     L2 = F'*L2*F + S2'*R2*S2 + cost[2].Q
        
    #     # TODO: construct P and α from S1 and S2
    #     P = SMatrix{nu, nx}(vcat(S1[:, 1:nx], S2[:, 1:nx]))
    #     α = SVector{nu}(ones(nu))
    #     strategies[kk] = AffineStrategy(P, α) # u = -P * x - α    
    #     @infiltrate 
    # end
    # @infiltrate
end
