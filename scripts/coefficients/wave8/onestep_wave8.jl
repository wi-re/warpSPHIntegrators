# Wave-8 one-step cross-check: exact Butcher step from the parsed ODE v7.8.1
# dump vs ODE's own solver, one fixed step, 2D nonlinear problem.
#
# Two problem variants per method:
#   aut:  f independent of t  (c-independent; validates a, b, b_embed)
#   nonaut: f depends on t    (validates the abscissae c as well)
#
# Three columns per variant:
#   diff_stored  -- my step with the STORED ODE c vs ODE's solve
#   diff_rowsum  -- my step with c = row sums (the consistent abscissae) vs ODE
#   drow_rowsum  -- |step(c_stored) - step(c_rowsum)|  (isolates the stored-c bug)
#
# Expected: clean methods MATCH at ~1e-13 (12-digit families ~1e-11) on both
# variants; Kvaerno3/4/5 and ESDIRK547L2SA2 (and possibly SSPSDIRK2) MISMATCH
# on the nonaut variant exactly by drow_rowsum (ODE's stored-c execution bug).
#
# Run from the checkout root with julia --project=scripts/coefficients/julia scripts/coefficients/wave8/onestep_wave8.jl
using OrdinaryDiffEq
using LinearAlgebra

_sdirk_keys = [k for k in keys(Base.loaded_modules) if string(k.name) == "OrdinaryDiffEqSDIRK"]
sdirk_loaded = !isempty(_sdirk_keys)
if !sdirk_loaded
    # Force-load the module (alg_order etc. reference it lazily).
    using OrdinaryDiffEqSDIRK
    _sdirk_keys = [k for k in keys(Base.loaded_modules) if string(k.name) == "OrdinaryDiffEqSDIRK"]
end
const S = Base.loaded_modules[_sdirk_keys[1]]
# DiffEqBase (for SplitODEProblem) is a transitive dep not loadable in the
# minimal probe project; grab it from the loaded modules like S above.
const D = Base.loaded_modules[[k for k in keys(Base.loaded_modules) if string(k.name) == "DiffEqBase"][1]]

# ---------------------------------------------------------------- parsing ---
text = read(joinpath(@__DIR__, "extract_wave8_out.txt"), String)

function parse_dump(txt)
    schemes = Dict{String, Dict{String, Any}}()
    cur = nothing
    for line in split(txt, "\n")
        m = match(r"^=== (\S+) \(\w+\) ===$", strip(line))
        if m !== nothing
            cur = m.captures[1]
            schemes[cur] = Dict{String, Any}()
            continue
        end
        cur === nothing && continue
        line = strip(line)
        occursin(" = ", line) || continue
        key, val = split(line, " = ", limit = 2)
        schemes[cur][strip(key)] = strip(val)
    end
    return schemes
end

function pmat(v::AbstractString)
    v = string(v)
    startswith(v, "Matrix{") && return nothing
    startswith(v, "Vector{") && return nothing
    v = strip(v)
    v == "Float64[]" && return nothing
    startswith(v, "[") || return nothing
    rows = [ [parse(Float64, x) for x in split(r)] for r in split(v[2:end-1], ";") ]
    s = length(rows)
    M = zeros(Float64, s, s)
    for i in 1:s
        M[i, :] = rows[i]
    end
    return M
end

function pvec(v::AbstractString)
    v = string(v)
    startswith(v, "Matrix{") && return nothing
    startswith(v, "Vector{") && return nothing
    v = strip(v)
    v == "Float64[]" && return nothing
    startswith(v, "[") || return nothing
    return [parse(Float64, x) for x in split(v[2:end-1], ",")]
end

# ----------------------------------------------------------------- problem ---
# 2D nonlinear, non-autonomous (t used only in the nonaut variant):
f(u, p, t) = [u[1]^2 * cos(t) + u[2] * sin(t) + 0.1 * t,
              -u[1] * u[2] + exp(-t) * u[1] - 0.5 * u[2]^3]
f_aut(u, p, t) = [u[1]^2 + u[2] + 0.1 * u[1] * u[2],
                  -u[1] * u[2] + 0.5 * u[1] - 0.5 * u[2]^3]

y0 = [0.5, -0.3]
dt = 0.1

function one_step(f, A, b, c, u0, dt, t0)
    s = length(b)
    n = length(u0)
    ks = Vector{typeof(u0)}(undef, s)
    for i in 1:s
        ti = t0 + c[i] * dt
        rhs = u0 .+ dt * sum((A[i, j] * ks[j] for j in 1:i-1); init=zero(u0))
        aii = A[i, i]
        if aii == 0
            ks[i] = f(rhs, nothing, ti)
        else
            # DIRK stage: solve F(u) = u - rhs - dt*a_ii*f(u, ti) = 0 by Newton
            # with a finite-difference Jacobian (n=2, smooth problem).
            F(u) = u .- rhs .- dt * aii * f(u, nothing, ti)
            u = rhs
            for _ in 1:60
                Fu = F(u)
                if norm(Fu) < 1e-15 * (1 + norm(u))
                    break
                end
                h = 1e-7
                Jm = zeros(n, n)
                for d in 1:n
                    up = copy(u)
                    up[d] += h
                    Jm[:, d] = (F(up) .- Fu) ./ h
                end
                u = u .- (Jm \ Fu)
            end
            ks[i] = f(u, nothing, ti)
        end
    end
    return u0 .+ dt * sum(b[i] * ks[i] for i in 1:s)
end

# --------------------------------------------------------------------- plan ---
# (name, alg constructor). All from OrdinaryDiffEqSDIRK (module S).
PLAN = [
    ("Kvaerno3",        () -> S.Kvaerno3()),
    ("Kvaerno4",        () -> S.Kvaerno4()),
    ("Kvaerno5",        () -> S.Kvaerno5()),
    ("KenCarp3",        () -> S.KenCarp3()),
    ("KenCarp4",        () -> S.KenCarp4()),
    ("KenCarp47",       () -> S.KenCarp47()),
    ("KenCarp5",        () -> S.KenCarp5()),
    ("KenCarp58",       () -> S.KenCarp58()),
    ("ESDIRK325L2SA",   () -> S.ESDIRK325L2SA()),
    ("ESDIRK436L2SA2",  () -> S.ESDIRK436L2SA2()),
    ("ESDIRK437L2SA",   () -> S.ESDIRK437L2SA()),
    ("ESDIRK547L2SA2",  () -> S.ESDIRK547L2SA2()),
    ("ESDIRK54I8L2SA",  () -> S.ESDIRK54I8L2SA()),
    ("ESDIRK659L2SA",   () -> S.ESDIRK659L2SA()),
    ("SSPSDIRK2",       () -> S.SSPSDIRK2()),
    ("ARS222",          () -> S.ARS222()),
    ("ARS232",          () -> S.ARS232()),
    ("ARS343",          () -> S.ARS343()),
    ("ARS443",          () -> S.ARS443()),
    ("CFNLIRK3",        () -> S.CFNLIRK3()),
    ("BHR553",          () -> S.BHR553()),
    ("Cash4",           () -> S.Cash4()),
    ("Hairer4",         () -> S.Hairer4()),
    ("Hairer42",        () -> S.Hairer42()),
    ("SFSDIRK4",        () -> S.SFSDIRK4()),
    ("SFSDIRK5",        () -> S.SFSDIRK5()),
    ("SFSDIRK6",        () -> S.SFSDIRK6()),
    ("SFSDIRK7",        () -> S.SFSDIRK7()),
    ("SFSDIRK8",        () -> S.SFSDIRK8()),
]

dump = parse_dump(text)

out = open(joinpath(@__DIR__, "onestep_wave8_out.txt"), "w")
println(out, rpad("method", 16), rpad("aut_diff_stored", 16),
        rpad("aut_diff_rowsum", 16), rpad("na_diff_stored", 16),
        rpad("na_diff_rowsum", 16), rpad("na_cbug", 16))
println(out, "-" ^ 96)

for (name, makealg) in PLAN
    d = dump[name]
    A = pmat(d["Ai"])
    b = pvec(d["bi"])
    c_stored = pvec(d["c"])
    c_rs = [sum(A[i, :]) for i in 1:size(A, 1)]
    if name == "SSPSDIRK2"
        # published Ketcheson 2009 form; row sums already give [1/4, 3/4]
        c_rs = [0.25, 0.75]
    end
    line = rpad(name, 16)
    for (tag, f) in (("aut", f_aut), ("na", f))
        prob = if name == "CFNLIRK3"
            # issplit(alg) = true: ODE needs a SplitODEProblem. Zeroing the
            # explicit part makes the effective step exactly the implicit-side
            # DIRK step (Ae/be act on zero explicit stage derivatives).
            D.SplitODEProblem(f, (u, p, t) -> zero(u), y0, (0.0, dt), ())
        else
            ODEProblem(f, y0, (0.0, dt), ())
        end
        try
            sol = solve(prob, makealg(); dt = dt, adaptive = false, nlsolve_kwargs = (rtol = 1e-14, atol = 1e-16))
            @assert sol.t[end] == dt
            @assert length(sol.t) == 2  # exactly one step
            ode_u = sol.u[end]
            d_stored = maximum(abs, ode_u .- one_step(f, A, b, c_stored, y0, dt, 0.0))
            d_rowsum = maximum(abs, ode_u .- one_step(f, A, b, c_rs, y0, dt, 0.0))
            drow = if tag == "na"
                maximum(abs, one_step(f, A, b, c_stored, y0, dt, 0.0) .- one_step(f, A, b, c_rs, y0, dt, 0.0))
            else
                0.0
            end
            line = line * rpad(string(d_stored), 16) * rpad(string(d_rowsum), 16)
            if tag == "na"
                line = line * rpad(string(drow), 16)
            else
                # keep column alignment: aut has no cbug column
                line = line * rpad("-", 16)
            end
        catch e
            line = line * rpad("ERROR: " * sprint(showerror, e), 16) * rpad("-", 16) * rpad("-", 16)
        end
    end
    println(out, line)
end

# SDIRK22 constructor-bug record: its constructor literally returns Trapezoid()
# in ODE v7.8.1 (algorithms.jl:314-330).
println(out)
println(out, "SDIRK22 constructor type: ", typeof(S.SDIRK22()))
println(out, "SDIRK22compact fields: ")
for (k, v) in pairs(dump["SDIRK22compact"])
    println(out, "  ", k, " = ", v)
end
close(out)
println("DONE -> ", joinpath(@__DIR__, "onestep_wave8_out.txt"))
