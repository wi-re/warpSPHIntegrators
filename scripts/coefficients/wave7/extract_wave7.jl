# Wave-7: dump the ODE v7.8.1 tableaus we are about to register.
#
# Includes the three pure-data tableau files from the installed OrdinaryDiffEq
# package and instantiates each cache/tableau the way ODE does for a Float64
# problem (T = T2 = Float64). Every field is repr()'d so the floats round-trip
# exactly into the Python probe.
#
# The data files need exactly two symbols in scope: OrdinaryDiffEqConstantCache
# (referenced by the HighOrderRK and Feagin cache files) and @fold (Verner
# tableaus). The wave6 project does not list OrdinaryDiffEqCore as a direct
# dependency (`using OrdinaryDiffEqCore` fails with "not found in current
# path"), so we bridge through the loaded OrdinaryDiffEq namespace and define
# the 2-line fold macro locally (verbatim from
# OrdinaryDiffEqCore/src/misc_utils.jl:273).
#
# Run from the checkout root with julia --project=scripts/coefficients/julia scripts/coefficients/wave7/extract_wave7.jl
using OrdinaryDiffEq
const _ODEC = OrdinaryDiffEq.OrdinaryDiffEqCore
const OrdinaryDiffEqConstantCache = _ODEC.OrdinaryDiffEqConstantCache
const CompiledFloats = _ODEC.CompiledFloats

macro fold(arg)
    return esc(:(Base.@assume_effects :foldable $arg))
end

L = joinpath(dirname(dirname(pathof(OrdinaryDiffEq))), "lib")

include(joinpath(L, "OrdinaryDiffEqHighOrderRK/src/high_order_rk_tableaus.jl"))
include(joinpath(L, "OrdinaryDiffEqFeagin/src/feagin_tableaus.jl"))
include(joinpath(L, "OrdinaryDiffEqVerner/src/verner_tableaus.jl"))

out = open(joinpath(@__DIR__, "extract_wave7_out.txt"), "w")

function dumpobj(io, name, obj)
    println(io, "=== $name (", nameof(typeof(obj)), ") ===")
    for f in fieldnames(typeof(obj))
        println(io, "  ", f, " = ", repr(getfield(obj, f)))
    end
    println(io)
end

try
    dumpobj(out, "TanYam7", TanYam7ConstantCache(Float64, Float64))
    dumpobj(out, "TsitPap8", TsitPap8ConstantCache(Float64, Float64))
    dumpobj(out, "DP8", DP8ConstantCache(Float64, Float64))
    dumpobj(out, "Feagin10", Feagin10ConstantCache(Float64, Float64))
    dumpobj(out, "Feagin12", Feagin12ConstantCache(Float64, Float64))
    dumpobj(out, "Feagin14", Feagin14ConstantCache(Float64, Float64))
    dumpobj(out, "Vern6", Vern6Tableau(Float64, Float64))
    dumpobj(out, "Vern7", Vern7Tableau(Float64, Float64))
    dumpobj(out, "Vern8", Vern8Tableau(Float64, Float64))
    dumpobj(out, "Vern9", Vern9Tableau(Float64, Float64))
    dumpobj(out, "RKV76IIa", RKV76IIaTableau(Float64, Float64))
finally
    close(out)
end
println("DONE -> ", joinpath(@__DIR__, "extract_wave7_out.txt"))
