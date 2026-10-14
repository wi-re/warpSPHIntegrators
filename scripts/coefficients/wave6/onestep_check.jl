# One-step cross-check: my tableau reconstruction (from extract_out.txt) vs
# ODE's own alg, one fixed step on a nonlinear 3D problem.
using OrdinaryDiffEq
using OrdinaryDiffEqLowOrderRK
using OrdinaryDiffEqTsit5
using Printf

text = read(joinpath(@__DIR__, "extract_out.txt"), String)
function parse_extract(txt)
    vals = Dict{String, Dict{String, Float64}}()
    current = nothing
    for line in split(txt, "\n")
        line = strip(line)
        if startswith(line, "### ")
            current = strip(line[5:end])
            vals[current] = Dict{String, Float64}()
        elseif current !== nothing && occursin(" = ", line)
            f, v = split(line, " = ", limit = 2)
            vals[current][strip(f)] = parse(Float64, strip(v))
        end
    end
    return vals
end
vals = parse_extract(text)

f(u, p, t) = [-u[2] - 0.5u[1]*u[3],
              u[1] + 0.3u[2]*u[3] - 0.2u[2]^2,
              0.1u[1]*u[2] + 0.05cos(3t)*u[3]]

function one_step(a, b, c, u0, dt, t0)
    s = length(b)
    ks = Vector{typeof(u0)}(undef, s)
    for i in 1:s
        ui = u0 .+ dt * sum((a[i, j] * ks[j] for j in 1:i-1); init=zero(u0))
        ks[i] = f(ui, nothing, t0 + c[i] * dt)
    end
    return u0 .+ dt * sum(b[i] * ks[i] for i in 1:s)
end

function zeroA(s)
    return zeros(Float64, s, s)
end
function putA!(a, row, cols)
    for (j, v) in cols
        a[row, j] = v
    end
end

# method name => (alg constructor, cache name, builder)
function build_Tsit5(t)
    a = zeroA(7)
    putA!(a, 2, [(1, t["a21"])])
    putA!(a, 3, [(1, t["a31"]), (2, t["a32"])])
    putA!(a, 4, [(1, t["a41"]), (2, t["a42"]), (3, t["a43"])])
    putA!(a, 5, [(1, t["a51"]), (2, t["a52"]), (3, t["a53"]), (4, t["a54"])])
    putA!(a, 6, [(1, t["a61"]), (2, t["a62"]), (3, t["a63"]), (4, t["a64"]), (5, t["a65"])])
    putA!(a, 7, [(1, t["a71"]), (2, t["a72"]), (3, t["a73"]), (4, t["a74"]), (5, t["a75"]), (6, t["a76"])])
    c = [0.0, t["c1"], t["c2"], t["c3"], t["c4"], t["c5"], t["c6"]]
    b = copy(a[7, :])
    return a, b, c
end

function build_OwrenZen3(t)
    a = zeroA(4)
    putA!(a, 2, [(1, t["a21"])])
    putA!(a, 3, [(1, t["a31"]), (2, t["a32"])])
    putA!(a, 4, [(1, t["a41"]), (2, t["a42"]), (3, t["a43"])])
    c = [0.0, t["c1"], t["c2"], 1.0]
    b = [t["a41"], t["a42"], t["a43"], 0.0]
    a[4, :] = b
    return a, b, c
end

function build_OwrenZen4(t)
    a = zeroA(6)
    putA!(a, 2, [(1, t["a21"])])
    putA!(a, 3, [(1, t["a31"]), (2, t["a32"])])
    putA!(a, 4, [(1, t["a41"]), (2, t["a42"]), (3, t["a43"])])
    putA!(a, 5, [(1, t["a51"]), (2, t["a52"]), (3, t["a53"]), (4, t["a54"])])
    c = [0.0, t["c1"], t["c2"], t["c3"], t["c4"], 1.0]
    b = [t["a61"], 0.0, t["a63"], t["a64"], t["a65"], 0.0]
    a[6, :] = b
    return a, b, c
end

function build_OwrenZen5(t)
    a = zeroA(8)
    putA!(a, 2, [(1, t["a21"])])
    putA!(a, 3, [(1, t["a31"]), (2, t["a32"])])
    putA!(a, 4, [(1, t["a41"]), (2, t["a42"]), (3, 1.0)])
    putA!(a, 5, [(1, t["a51"]), (2, t["a52"]), (3, t["a53"]), (4, t["a54"])])
    putA!(a, 6, [(1, t["a61"]), (2, t["a62"]), (3, t["a63"]), (4, t["a64"]), (5, t["a65"])])
    putA!(a, 7, [(1, t["a71"]), (2, t["a72"]), (3, t["a73"]), (4, t["a74"]), (5, t["a75"]), (6, t["a76"])])
    c = [0.0, t["c1"], t["c2"], t["c3"], t["c4"], t["c5"], t["c6"], 1.0]
    b = [t["a81"], 0.0, t["a83"], t["a84"], t["a85"], t["a86"], t["a87"], 0.0]
    a[8, :] = b
    return a, b, c
end

function build_BS5(t)
    a = zeroA(8)
    putA!(a, 2, [(1, t["a21"])])
    putA!(a, 3, [(1, t["a31"]), (2, t["a32"])])
    putA!(a, 4, [(1, t["a41"]), (2, t["a42"]), (3, t["a43"])])
    putA!(a, 5, [(1, t["a51"]), (2, t["a52"]), (3, t["a53"]), (4, t["a54"])])
    putA!(a, 6, [(1, t["a61"]), (2, t["a62"]), (3, t["a63"]), (4, t["a64"]), (5, t["a65"])])
    putA!(a, 7, [(1, t["a71"]), (2, t["a72"]), (3, t["a73"]), (4, t["a74"]), (5, t["a75"]), (6, t["a76"])])
    c = [0.0, t["c1"], t["c2"], t["c3"], t["c4"], t["c5"], 1.0, 1.0]
    b = [t["a81"], 0.0, t["a83"], t["a84"], t["a85"], t["a86"], t["a87"], 0.0]
    a[8, :] = b
    return a, b, c
end

function build_Stepanov5(t)
    a = zeroA(7)
    putA!(a, 2, [(1, t["a21"])])
    putA!(a, 3, [(1, t["a31"]), (2, t["a32"])])
    putA!(a, 4, [(1, t["a41"]), (2, t["a42"]), (3, t["a43"])])
    putA!(a, 5, [(1, t["a51"]), (2, t["a52"]), (3, t["a53"]), (4, t["a54"])])
    putA!(a, 6, [(1, t["a61"]), (2, t["a62"]), (3, t["a63"]), (4, t["a64"]), (5, t["a65"])])
    c = [0.0, t["c2"], t["c3"], t["c4"], t["c5"], t["c6"], 1.0]
    b = [t["b1"], 0.0, t["b3"], t["b4"], t["b5"], t["b6"], 0.0]
    a[7, :] = b
    return a, b, c
end

function build_SIR54(t)
    a = zeroA(8)
    putA!(a, 2, [(1, t["a21"])])
    putA!(a, 3, [(1, t["a31"]), (2, t["a32"])])
    putA!(a, 4, [(1, t["a41"]), (2, t["a42"]), (3, t["a43"])])
    putA!(a, 5, [(1, t["a51"]), (2, t["a52"]), (3, t["a53"]), (4, t["a54"])])
    putA!(a, 6, [(1, t["a61"]), (2, t["a62"]), (3, t["a63"]), (4, t["a64"]), (5, t["a65"])])
    putA!(a, 7, [(1, t["a71"]), (2, t["a72"]), (3, t["a73"]), (4, t["a74"]), (5, t["a75"]), (6, t["a76"])])
    c = [0.0, t["c2"], t["c3"], t["c4"], t["c5"], t["c6"], t["c7"], 1.0]
    b = [t["b1"], t["b2"], t["b3"], t["b4"], t["b5"], t["b6"], 0.0, 0.0]
    a[8, :] = b
    return a, b, c
end

function build_Alshina2(t)
    a = zeroA(2)
    putA!(a, 2, [(1, t["a21"])])
    c = [0.0, t["c2"]]
    b = [t["b1"], t["b2"]]
    return a, b, c
end

function build_Alshina3(t)
    a = zeroA(3)
    putA!(a, 2, [(1, t["a21"])])
    putA!(a, 3, [(2, t["a32"])])
    c = [0.0, t["c2"], t["c3"]]
    b = [t["b1"], t["b2"], t["b3"]]
    return a, b, c
end

function build_Alshina6(t)
    a = zeroA(7)
    putA!(a, 2, [(1, t["a21"])])
    putA!(a, 3, [(1, t["a31"]), (2, t["a32"])])
    putA!(a, 4, [(1, t["a41"]), (2, t["a42"]), (3, t["a43"])])
    putA!(a, 5, [(1, t["a51"]), (2, t["a52"]), (3, t["a53"]), (4, t["a54"])])
    putA!(a, 6, [(1, t["a61"]), (2, t["a62"]), (3, t["a63"]), (4, t["a64"]), (5, t["a65"])])
    putA!(a, 7, [(1, t["a71"]), (2, t["a72"]), (3, t["a73"]), (4, t["a74"]), (5, t["a75"]), (6, t["a76"])])
    c = [0.0, t["c2"], t["c3"], t["c4"], t["c5"], t["c6"], t["c7"]]
    b = [t["b1"], 0.0, 0.0, 0.0, t["b5"], t["b6"], t["b7"]]
    return a, b, c
end

function build_MSK56(t, msrk6::Bool)
    a = zeroA(9)
    putA!(a, 2, [(1, t["a21"])])
    if msrk6
        putA!(a, 3, [(2, t["a32"])])
    else
        putA!(a, 3, [(1, t["a31"]), (2, t["a32"])])
    end
    putA!(a, 4, [(1, t["a41"]), (3, t["a43"])])
    putA!(a, 5, [(1, t["a51"]), (3, t["a53"]), (4, t["a54"])])
    putA!(a, 6, [(1, t["a61"]), (3, t["a63"]), (4, t["a64"]), (5, t["a65"])])
    putA!(a, 7, [(1, t["a71"]), (3, t["a73"]), (4, t["a74"]), (5, t["a75"]), (6, t["a76"])])
    putA!(a, 8, [(1, t["a81"]), (3, t["a83"]), (4, t["a84"]), (5, t["a85"]), (6, t["a86"]), (7, t["a87"])])
    c = [0.0, t["c2"], t["c3"], t["c4"], t["c5"], t["c6"], t["c7"], t["c8"], 1.0]
    b = [t["b1"], 0.0, 0.0, t["b4"], t["b5"], t["b6"], t["b7"], t["b8"], 0.0]
    a[9, :] = b
    return a, b, c
end

function build_RKO65(t)
    a = zeroA(6)
    putA!(a, 2, [(1, t["α21"])])
    putA!(a, 3, [(1, t["α31"]), (2, t["α32"])])
    putA!(a, 4, [(1, t["α41"]), (2, t["α42"]), (3, t["α43"])])
    putA!(a, 5, [(1, t["α51"]), (2, t["α52"]), (3, t["α53"]), (4, t["α54"])])
    putA!(a, 6, [(2, t["α62"]), (3, t["α63"]), (4, t["α64"]), (5, t["α65"])])
    c = [t["c1"], t["c2"], t["c3"], t["c4"], t["c5"], t["c6"]]
    b = [0.0, t["β2"], t["β3"], t["β4"], t["β5"], t["β6"]]
    return a, b, c
end

function build_RKM(t)
    a = zeroA(6)
    putA!(a, 2, [(1, t["α2"])])
    putA!(a, 3, [(2, t["α3"])])
    putA!(a, 4, [(3, t["α4"])])
    putA!(a, 5, [(4, t["α5"])])
    putA!(a, 6, [(5, t["α6"])])
    c = [0.0, t["c2"], t["c3"], t["c4"], t["c5"], t["c6"]]
    b = [t["β1"], t["β2"], t["β3"], t["β4"], 0.0, t["β6"]]
    return a, b, c
end

function build_Ralston4(t)
    a = zeroA(4)
    putA!(a, 2, [(1, t["a21"])])
    putA!(a, 3, [(1, t["a31"]), (2, t["a32"])])
    putA!(a, 4, [(1, t["a41"]), (2, t["a42"]), (3, t["a43"])])
    c = [0.0, t["c2"], t["c3"], 1.0]
    b = [t["b1"], t["b2"], t["b3"], t["b4"]]
    return a, b, c
end

function build_PSRK(t, s)
    a = zeroA(s)
    for i in 2:s
        for j in 1:i-1
            a[i, j] = t[string("a", i, j)]
        end
    end
    c = [0.0]
    for i in 2:s
        push!(c, t[string("c", i)])
    end
    b = [t[string("b", i)] for i in 1:s]
    return a, b, c
end

u0 = [0.3, -0.2, 0.1]
dt = 0.1
t0 = 0.0

methods = [
    ("Tsit5", () -> Tsit5(), "Tsit5ConstantCacheActual", build_Tsit5),
    ("OwrenZen3", () -> OwrenZen3(), "OwrenZen3ConstantCache", build_OwrenZen3),
    ("OwrenZen4", () -> OwrenZen4(), "OwrenZen4ConstantCache", build_OwrenZen4),
    ("OwrenZen5", () -> OwrenZen5(), "OwrenZen5ConstantCache", build_OwrenZen5),
    ("BS5", () -> BS5(), "BS5ConstantCache", build_BS5),
    ("Stepanov5", () -> Stepanov5(), "Stepanov5ConstantCache", build_Stepanov5),
    ("SIR54", () -> SIR54(), "SIR54ConstantCache", build_SIR54),
    ("Alshina2", () -> Alshina2(), "Alshina2ConstantCache", build_Alshina2),
    ("Alshina3", () -> Alshina3(), "Alshina3ConstantCache", build_Alshina3),
    ("Alshina6", () -> Alshina6(), "Alshina6ConstantCache", build_Alshina6),
    ("MSRK5", () -> MSRK5(), "MSRK5ConstantCache", t -> build_MSK56(t, false)),
    ("MSRK6", () -> MSRK6(), "MSRK6ConstantCache", t -> build_MSK56(t, true)),
    ("RKO65", () -> RKO65(), "RKO65ConstantCache", build_RKO65),
    ("RKM", () -> RKM(), "RKMConstantCache", build_RKM),
    ("Ralston4", () -> Ralston4(), "Ralston4ConstantCache", build_Ralston4),
    ("PSRK4p7q6", () -> PSRK4p7q6(), "PSRK4p7q6ConstantCache", t -> build_PSRK(t, 6)),
    ("PSRK3p6q5", () -> PSRK3p6q5(), "PSRK3p6q5ConstantCache", t -> build_PSRK(t, 5)),
    ("PSRK3p5q4", () -> PSRK3p5q4(), "PSRK3p5q4ConstantCache", t -> build_PSRK(t, 4)),
]

println(rpad("method", 14), " ", rpad("max|diff|", 14), "  ", "verdict")
println("-" ^ 44)
for (name, makealg, cachename, build) in methods
    a, b, c = build(vals[cachename])
    mine = one_step(a, b, c, u0, dt, t0)
    prob = ODEProblem(f, u0, (t0, t0 + dt))
    sol = solve(prob, makealg(); dt = dt, adaptive = false)
    ode_u = sol.u[end]
    d = maximum(abs, ode_u .- mine)
    verdict = d < 1e-12 ? "MATCH" : (d < 1e-9 ? "close" : "MISMATCH")
    println(rpad(name, 14), " ", rpad(string(d), 14), "  ", verdict)
end
println("ONE-STEP CHECK DONE")
