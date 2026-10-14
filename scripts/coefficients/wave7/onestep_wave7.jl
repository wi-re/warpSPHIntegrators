# Wave-7 one-step cross-check: run each ODE method for exactly one step of a
# 2D nonlinear NON-autonomous problem and print the propagated solution. The
# Python probe (crosscheck_wave7.py) runs the assembled Butcher tableaus on the
# identical problem; the two must agree to ~1e-14.
#
# Run from the checkout root with julia --project=scripts/coefficients/julia scripts/coefficients/wave7/onestep_wave7.jl
using OrdinaryDiffEq
# `using OrdinaryDiffEq` only re-exports Vern6-9; pull the rest from their home packages.
using OrdinaryDiffEqVerner: RKV76IIa
using OrdinaryDiffEqHighOrderRK: TanYam7, TsitPap8, DP8
using OrdinaryDiffEqFeagin: Feagin10, Feagin12, Feagin14

# f(u, p, t) = [ u1^2 cos(t) + u2 sin(t) + 0.1 t ,
#                -u1 u2 + exp(-t) u1 - 0.5 u2^3 ]
# (SciMLBase requires the parameters argument p; it is unused)
f(u, p, t) = [u[1]^2 * cos(t) + u[2] * sin(t) + 0.1 * t,
              -u[1] * u[2] + exp(-t) * u[1] - 0.5 * u[2]^3]

y0 = [0.5, -0.3]
dt = 0.1
prob = ODEProblem(f, y0, (0.0, dt), ())

methods = [
    ("TanYam7",  TanYam7()),
    ("TsitPap8", TsitPap8()),
    ("DP8",      DP8()),
    ("Feagin10", Feagin10()),
    ("Feagin12", Feagin12()),
    ("Feagin14", Feagin14()),
    ("Vern6",    Vern6()),
    ("Vern7",    Vern7()),
    ("Vern8",    Vern8()),
    ("Vern9",    Vern9()),
    ("RKV76IIa", RKV76IIa()),
]

out = open(joinpath(@__DIR__, "onestep_wave7_out.txt"), "w")
for (name, alg) in methods
    try
        sol = solve(prob, alg; dt=dt, adaptive=false)
        y1 = sol.u[end]
        @assert sol.t[end] == dt
        @assert length(sol.t) == 2  # exactly one step
        println(out, name, "\t", y1[1], "\t", y1[2])
    catch e
        println(out, name, "\tERROR\t", sprint(showerror, e))
    end
end
close(out)
println("DONE -> onestep_wave7_out.txt")
