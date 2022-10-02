module Differentiable_Solvers

"----------------------- differentiable forward game solver -----------------------"

import StaticArrays
import ForwardDiff
using NamedTupleTools: @namedtuple
using iLQGames:
    LQGame,
    LTVSystem,
    AffineStrategy,
    QuadraticPlayerCost,
    SystemTrajectory,
    uindex,
    dynamics,
    player_costs,
    control_input,
    next_x,
    n_players,
    n_states,
    n_controls,
    time_disc2cont,
    linearize_discrete,
    regularize,
    LinearizationStyle
using Infiltrator
using LinearAlgebra
using StaticArrays
# untyped_solve_lq_game_FBNE(g::LQGame) = untyped_solve_lq_game_FBNE(uindex(g), dynamics(g), player_costs(g))

"A type relaxed version of solve_lq_game! without side effects"
function solve_lq_game_FBNE(g::LQGame)
    costs = player_costs(g)
    h = length(costs)
    uids = uindex(g)

    # initializting the optimal cost to go representation for DP
    # quadratic cost to go
    cost_to_go = map(c -> (ζ = c.l, Z = c.Q), last(costs))
    nx = first(size(dynamics(g)[1].A))
    full_xrange = StaticArrays.SVector{nx}(1:nx)
    
    # working backwards in time to solve the dynamic program
    map(h:-1:1) do kk 
        dyn = dynamics(g)[kk]
        cost = costs[kk]
        # convenience shorthands for the relevant quantities
        A = dyn.A
        B = dyn.B

        S, Y = mapreduce((a, b) -> vcat.(a, b), uids, cost, cost_to_go) do uidᵢ, cᵢ, Cᵢ
            BᵢZᵢ = B[:, uidᵢ]' * Cᵢ.Z
            (
                cᵢ.R[uidᵢ, :] + BᵢZᵢ * B,                     # rows of S
                [(BᵢZᵢ * A) (B[:, uidᵢ]' * Cᵢ.ζ + cᵢ.r[uidᵢ])],
            ) # rows of Y
        end

        # solve for the gains `P` and feed forward terms `α` simulatiously
        P_and_α = S \ Y
        P = P_and_α[:, full_xrange]
        α = P_and_α[:, end]

        # compute F and β as intermediate resu,lt for estimating the cost to go
        F = A - B * P
        β = -B * α

        # update Z and ζ (cost to go representation for the next step backwards
        # in time)
        cost_to_go = map(cost, cost_to_go) do cᵢ, Cᵢ
            PRᵢ = P' * cᵢ.R
            (
                ζ = (F' * (Cᵢ.ζ + Cᵢ.Z * β) + cᵢ.l + PRᵢ * α - P' * cᵢ.r),
                Z = (F' * Cᵢ.Z * F + cᵢ.Q + PRᵢ * P),
            )
        end

        AffineStrategy(P, α)
    end |> reverse
end


# function solve_lq_game_FBNE(g::LQGame)
#     nx = n_states(g)
#     nu = n_controls(g)
#     @infiltrate
#     Z = [ForwardDiff.Dual.(pc.Q) for pc in last(player_costs(g))]
#     ζ = [ForwardDiff.Dual.(pc.l) for pc in last(player_costs(g))]
#     strategies = []
    
#     T = length(player_costs(g))
#     for kk in T:-1:1
#         S = zeros(ForwardDiff.Dual, nu, nu)
#         YP = zeros(ForwardDiff.Dual, nu, nx)
#         Yα = zeros(ForwardDiff.Dual, nu)
#         dyn = dynamics(g)[kk]
#         cost = player_costs(g)[kk]
#         A = dyn.A
#         B = dyn.B
#         # @infiltrate
#         for (ii, udxᵢ) in enumerate(uindex(g))
#             BᵢZᵢ = B[:, udxᵢ]' * Z[ii]
#             S[udxᵢ, :] = cost[ii].R[udxᵢ, :] + BᵢZᵢ*B
#             YP[udxᵢ, :] =  BᵢZᵢ*A
#             Yα[udxᵢ] =  B[:, udxᵢ]'*ζ[ii] + cost[ii].r[udxᵢ]
#         end
#         # Sinv = inv(S)
#         # @infiltrate
#         P = S\YP
#         α = S\Yα
#         F = A - B * P
#         β = -B * α
#         for ii in 1:n_players(g)
#             cᵢ= cost[ii]
#             PRᵢ = P' * cᵢ.R
#             ζ[ii] = F' * (ζ[ii] + Z[ii] * β) + cᵢ.l + PRᵢ * α - P' * cᵢ.r
#             Z[ii] = F' * Z[ii] * F + cᵢ.Q + PRᵢ * P
#         end
#         push!(strategies, AffineStrategy(SMatrix{nu,nx}(P), SVector{nu}(α)))
#     end
#     return reverse(strategies)
# end

function solve_lq_game_OLNE(g::LQGame)
    nx, nu, N = n_states(g), n_controls(g), n_players(g)
    T = length(player_costs(g))
    # N is the number of players
    Mₜ = [player_costs(g)[T][ii].Q for ii in 1:N]
    mₜ = [player_costs(g)[T][ii].l for ii in 1:N]
    strategies = []
    
    for kk in T:-1:1
        tmp_M = [zeros(nx, nx) for ii in 1:N]
        tmp_m = [zeros(nx) for ii in 1:N]
        Mₜ, mₜ = [tmp_M Mₜ], [tmp_m mₜ]
        dyn = dynamics(g)[kk]
        cost = player_costs(g)[kk]
        A, B = dyn.A, dyn.B
        Λₜ, ηₜ = I(nx), zeros(nx)
        P = zeros(ForwardDiff.Dual, nu, nx)
        α = zeros(ForwardDiff.Dual, nu)
        for (ii, udxᵢ) in enumerate(uindex(g))
            inv_Rₜ = inv(cost[ii].R[udxᵢ,udxᵢ])
            Λₜ +=  B[:, udxᵢ]*inv_Rₜ*B[:, udxᵢ]'*Mₜ[ii,2]
            ηₜ -=  B[:, udxᵢ]*inv_Rₜ*(B[:, udxᵢ]'*mₜ[ii,2] + cost[ii].r[udxᵢ])
        end
        for (ii, udxᵢ) in enumerate(uindex(g))
            
            inv_Λₜ = inv(Λₜ)
            inv_Rₜ = inv(cost[ii].R[udxᵢ,udxᵢ])
            P[udxᵢ,:] = - inv_Rₜ*B[:,udxᵢ]'*(Mₜ[ii,2]*inv_Λₜ*A)
            α[udxᵢ] = - inv_Rₜ*B[:,udxᵢ]'*(Mₜ[ii,2]*inv_Λₜ*ηₜ+mₜ[ii,2]) - inv_Rₜ*cost[ii].r[udxᵢ]
            mₜ[ii,1] = cost[ii].l + A'*(mₜ[ii,2] + Mₜ[ii,2]*inv_Λₜ*ηₜ)
            Mₜ[ii,1] = cost[ii].Q + A'*Mₜ[ii,2]*inv_Λₜ*A
        end
        push!(strategies, AffineStrategy(SMatrix{nu,nx}(-P), SVector{nu}(-α)))
    end
    return reverse(strategies)
end


"A type relaxed version of trajectory! without side effects"
function trajectory(x0, g, γ, op = zero(SystemTrajectory, g))
    xs, us = StaticArrays.SVector{n_states(g)}[], StaticArrays.SVector{n_controls(g)}[]
    reduce(zip(γ, op.x, op.u); init = x0) do xₖ, (γₖ, x̃ₖ, ũₖ)
        Δxₖ = xₖ - x̃ₖ
        uₖ = control_input(γₖ, Δxₖ, ũₖ)
        push!(xs, xₖ)
        push!(us, uₖ)
        # @infiltrate
        next_x(dynamics(g), xₖ, uₖ, 0.0)
    end
    vectype = StaticArrays.SizedVector{length(γ)}
    SystemTrajectory{0.1}(vectype(xs), vectype(us), 0.0) # loses information
end



function integrate(cs, x0, u, t0,
                   ΔT, n_intsteps=2)
    Δt = ΔT/n_intsteps
    x = x0
    for t in range(t0, stop=t0+ΔT, length=n_intsteps+1)[1:end-1]
        k1 = Δt * dx(cs, x, u, t);
        k2 = Δt * dx(cs, x + 0.5 * k1, u, t + 0.5 * Δt);
        k3 = Δt * dx(cs, x + 0.5 * k2, u, t + 0.5 * Δt);
        k4 = Δt * dx(cs, x + k3      , u, t + Δt);
        x += (k1 + 2.0 * (k2 + k3) + k4) / 6.0;
    end
    return x
end

"A type relaxed version of lq_approximation! without side effects and gradient
optimzation"
function lq_approximation(g, op, solver)
    lqs = map(eachindex(op.x), op.x, op.u) do k, x, u
        # discrete linearization along the operating point
        t = time_disc2cont(op, k)
        # @infiltrate
        
        ldyn = linearize_discrete(dynamics(g), x, u, t)

        # quadratiation of the cost along the operating point
        qcost =
            map(player_costs(g)) do pc
                x_cost = x -> pc(g, x, u, t)
                u_cost = u -> pc(g, x, u, t)
                l = ForwardDiff.gradient(x_cost, x)
                Q = ForwardDiff.hessian(x_cost, x)
                r = ForwardDiff.gradient(u_cost, u)
                R = ForwardDiff.hessian(u_cost, u)
                c = QuadraticPlayerCost(l, Q, r, R)
                regularize(solver, c)
            end |> StaticArrays.SizedVector{n_players(g),QuadraticPlayerCost}
        @namedtuple(ldyn, qcost)
    end
    dyn = LTVSystem([lq.ldyn for lq in lqs])
    pcost = [lq.qcost for lq in lqs]
    LQGame(uindex(g), dyn, pcost)
end

end # module