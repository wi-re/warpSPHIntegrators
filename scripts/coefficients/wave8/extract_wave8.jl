# Wave 8: dump the ODE v7.8.1 SDIRK tableaus we are about to register.
#
# Every pure-SDIRK method in OrdinaryDiffEqSDIRK has a materialized
# ESDIRKIMEXTableau builder (Ai, bi, Ae, be, c, ce, btilde, ebtilde, +
# metadata). For a PURE problem (no explicit split) the effective method is
# the implicit side: a = Ai, b = bi, c = c, error pair from btilde -- the
# KenCarp3 Ai/bi/c below match the library's already-registered
# ESDIRK3(2)4L[2]SA (SUNDIALS ARKODE) to ~1e-16, confirming the convention.
# Dump all fields via repr() so the floats round-trip exactly into the
# Python probe.
#
# SDIRK22 has no usable builder in this pinned source and is excluded.
# The compact-field dump is exploratory evidence, not a registered method.
#
# Run from the checkout root with julia --project=scripts/coefficients/julia scripts/coefficients/wave8/extract_wave8.jl
using OrdinaryDiffEq
_sdirk_keys = [k for k in keys(Base.loaded_modules) if string(k.name) == "OrdinaryDiffEqSDIRK"]
_sdirk_keys == [] && error("OrdinaryDiffEqSDIRK not in the loaded modules")
const S = Base.loaded_modules[_sdirk_keys[1]]
const F = Float64

out = open(joinpath(@__DIR__, "extract_wave8_out.txt"), "w")

function dumpobj(io, name, obj)
    println(io, "=== $name (", nameof(typeof(obj)), ") ===")
    for f in fieldnames(typeof(obj))
        println(io, "  ", f, " = ", repr(getfield(obj, f)))
    end
    println(io)
end

function main()
    # (name, builder) -- the alg-argument builders get a default-constructed alg.
    builders = [
        ("Kvaerno3",   (T, T2) -> S.Kvaerno3ESDIRKIMEXTableau(T, T2)),
        ("Kvaerno4",   (T, T2) -> S.Kvaerno4ESDIRKIMEXTableau(T, T2)),
        ("Kvaerno5",   (T, T2) -> S.Kvaerno5ESDIRKIMEXTableau(T, T2)),
        ("KenCarp3",   (T, T2) -> S.KenCarp3ESDIRKIMEXTableau(T, T2)),
        ("KenCarp4",   (T, T2) -> S.KenCarp4ESDIRKIMEXTableau(T, T2)),
        ("KenCarp47",  (T, T2) -> S.KenCarp47ESDIRKIMEXTableau(T, T2)),
        ("KenCarp5",   (T, T2) -> S.KenCarp5ESDIRKIMEXTableau(T, T2)),
        ("KenCarp58",  (T, T2) -> S.KenCarp58ESDIRKIMEXTableau(T, T2)),
        ("ESDIRK325L2SA", (T, T2) -> S.ESDIRK325L2SAESDIRKIMEXTableau(T, T2)),
        ("ESDIRK436L2SA2", (T, T2) -> S.ESDIRK436L2SA2ESDIRKIMEXTableau(T, T2)),
        ("ESDIRK437L2SA", (T, T2) -> S.ESDIRK437L2SAESDIRKIMEXTableau(T, T2)),
        ("ESDIRK547L2SA2", (T, T2) -> S.ESDIRK547L2SA2ESDIRKIMEXTableau(T, T2)),
        ("ESDIRK54I8L2SA", (T, T2) -> S.ESDIRK54I8L2SAESDIRKIMEXTableau(T, T2)),
        ("ESDIRK659L2SA", (T, T2) -> S.ESDIRK659L2SAESDIRKIMEXTableau(T, T2)),
        ("SSPSDIRK2",  (T, T2) -> S.SSPSDIRK2ESDIRKIMEXTableau(T, T2)),
        ("SDIRK2",     (T, T2) -> S.SDIRK2ESDIRKIMEXTableau(T, T2)),
        ("ARS222",     (T, T2) -> S.ARS222Tableau(T, T2)),
        ("ARS232",     (T, T2) -> S.ARS232Tableau(T, T2)),
        ("ARS343",     (T, T2) -> S.ARS343Tableau(T, T2)),
        ("CFNLIRK3",   (T, T2) -> S.CFNLIRK3ESDIRKIMEXTableau(T, T2)),
        ("ARS443",     (T, T2) -> S.ARS443Tableau(T, T2)),
        ("BHR553",     (T, T2) -> S.BHR553Tableau(T, T2)),
        ("Cash4",      (T, T2) -> S.Cash4ESDIRKIMEXTableau(S.Cash4(), T, T2)),
        ("Hairer4",    (T, T2) -> S.Hairer4ESDIRKIMEXTableau(S.Hairer4(), T, T2)),
        ("Hairer42",   (T, T2) -> S.Hairer4ESDIRKIMEXTableau(S.Hairer42(), T, T2)),
        ("SFSDIRK4",   (T, T2) -> S.SFSDIRK4ESDIRKIMEXTableau(T, T2)),
        ("SFSDIRK5",   (T, T2) -> S.SFSDIRK5ESDIRKIMEXTableau(T, T2)),
        ("SFSDIRK6",   (T, T2) -> S.SFSDIRK6ESDIRKIMEXTableau(T, T2)),
        ("SFSDIRK7",   (T, T2) -> S.SFSDIRK7ESDIRKIMEXTableau(T, T2)),
        ("SFSDIRK8",   (T, T2) -> S.SFSDIRK8ESDIRKIMEXTableau(T, T2)),
    ]
    for (name, build) in builders
        try
            dumpobj(out, name, build(F, F))
        catch e
            println(out, "=== $name FAILED: ", sprint(showerror, e), " ===")
            println(out)
        end
    end
    # Compact SDIRK22 fields, for the hand transcription.
    dumpobj(out, "SDIRK22compact", S.SDIRK22Tableau(F))
end

try
    main()
finally
    close(out)
end
println("DONE -> ", joinpath(@__DIR__, "extract_wave8_out.txt"))
