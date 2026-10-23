# Wave 1 SSPRK reference trajectories (docs/development/plan-reconciliation-2026-10-23.md wave 1, 2026-10-12).
#
# Integrates a fixed nonlinear time-dependent 3D ODE with each of the 7 new
# SSPRK methods at fixed dt and writes the final state. The python companion
# (.tmp/wave1_tableaus.py) integrates the same problem with the transcribed
# Butcher tableaus and compares; agreement to ~1e-12 is the end-to-end proof
# that the transcribed (a, b, c) reproduces ODE v7.8.1's hand-coded
# Shu-Osher recurrences (coefficients from ssprk_caches.jl, recurrences from
# ssprk_perform_step.jl, read line-by-line 2026-10-12).
#
# Loading note: the OrdinaryDiffEq metapackage does NOT reexport its subpackages
# ("no blanket @reexport", src/OrdinaryDiffEq.jl), and the SSPRK subpackage is a
# [sources] path dependency (lib/OrdinaryDiffEqSSPRK) that Pkg did not record in
# the default environment's manifest ("not found in current path"). Its eight
# source files are self-contained and every one of its dependencies
# (OrdinaryDiffEqCore, DiffEqBase, SciMLBase, RecursiveArrayTools, FastBroadcast,
# MuladdMacro, PrecompileTools, Preferences, Reexport) is already in the default
# environment via OrdinaryDiffEq 7.8.1, so include the module source directly
# instead of touching the environment.
using OrdinaryDiffEq
include(joinpath(dirname(dirname(pathof(OrdinaryDiffEq))), "lib/OrdinaryDiffEqSSPRK/src/OrdinaryDiffEqSSPRK.jl"))

f(du, u, p, t) = begin
    du[1] = u[2] - u[1]*u[3] + sin(t)
    du[2] = -u[1] - u[2]^2 + cos(2t)
    du[3] = u[1]*u[2] - u[3] + t
end

u0 = [0.3, -0.7, 1.1]
prob = ODEProblem(f, u0, (0.0, 0.5))
dt = 0.02
algs = [
    ("SSPRK53", OrdinaryDiffEqSSPRK.SSPRK53()),
    ("SSPRK63", OrdinaryDiffEqSSPRK.SSPRK63()),
    ("SSPRK73", OrdinaryDiffEqSSPRK.SSPRK73()),
    ("SSPRK83", OrdinaryDiffEqSSPRK.SSPRK83()),
    ("SSPRK54", OrdinaryDiffEqSSPRK.SSPRK54()),
    ("KYKSSPRK42", OrdinaryDiffEqSSPRK.KYKSSPRK42()),
    ("KYK2014DGSSPRK_3S2", OrdinaryDiffEqSSPRK.KYK2014DGSSPRK_3S2()),
]

out = open(joinpath(@__DIR__, "ode_reference.txt"), "w")
for (name, alg) in algs
    sol = solve(prob, alg; dt=dt, adaptive=false, save_everystep=false)
    println(out, name)
    for x in sol.u[end]
        println(out, string(x))
    end
end
close(out)
println("wrote wave1/ode_reference.txt")
