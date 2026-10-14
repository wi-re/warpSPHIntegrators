# Independent coefficient/one-step oracle from the pinned monorepo's FIRK source.
# No FIRK installation or network required: only Julia stdlibs. Gauss quadrature
# is supplied by Golub-Welsch; the pinned constructor is evaluated unchanged.
# julia --startup-file=no oracle.jl /path/to/OrdinaryDiffEq/OnlvZ > oracle.json
using LinearAlgebra, SHA, Pkg

root = ARGS[1]
tree = bytes2hex(Pkg.GitTools.tree_hash(root))
@assert tree == "bae9b8e5f1d00e51c554648e5094e215164a19f8" tree
file = joinpath(root, "lib/OrdinaryDiffEqFIRK/src/firk_tableaus.jl")
source = read(file, String)
section(start, stop) = split(split(source, start; limit=2)[2], stop; limit=2)[1]
include_string(Main, "struct RadauIIA5Tableau" * section("struct RadauIIA5Tableau", "struct RadauIIA9Tableau"), file)
include_string(Main, "struct GaussLegendreTableau" * section("struct GaussLegendreTableau", "import FastGaussQuadrature"), file)
include_string(Main, "function generateGaussLegendreTableau" * section("function generateGaussLegendreTableau", "const GaussLegendreTableauCache"), file)
function gausslegendre(::Type{T}, n) where T
    values, vectors = eigen(SymTridiagonal(zeros(T,n), T[k/sqrt(4k*k-1) for k in 1:n-1]))
    return values, 2 .* vectors[1,:].^2
end
gausslegendre(n) = gausslegendre(Float64,n)

g = generateGaussLegendreTableau(Float64,Float64,3)
r = RadauIIA5Tableau(Float64,Float64)
T = [r.T11 r.T12 r.T13; r.T21 r.T22 r.T23; r.T31 1.0 0.0]
TI = [r.TI11 r.TI12 r.TI13; r.TI21 r.TI22 r.TI23; r.TI31 r.TI32 r.TI33]
inverse_A = T * [r.γ 0 0; 0 r.α -r.β; 0 r.β r.α] * TI
A = inv(inverse_A)
b = vec(A[3,:]); c = [r.c1,r.c2,1.0]; e = [r.e1,r.e2,r.e3]
radau_companion = vcat(-1.0, b - A' * e)

function step(A,b,c,companion,nonaut)
    y = [.7,-.4]; t=.3; h=.1; n=2; s=length(c)
    f(t,y) = [-y[1]+y[2]^2+(nonaut ? .1*sin(t) : 0), -.4*y[2]+.2*y[1]^2+(nonaut ? .1*cos(t) : 0)]
    jf(y) = [-1.0 2*y[2]; .4*y[1] -.4]
    Y = repeat(y,1,s)
    for iteration in 1:12
        K = hcat([f(t+c[j]*h,Y[:,j]) for j in 1:s]...)
        residual = Y .- y .- h*K*A'
        maximum(abs,residual)<1e-14 && break
        J = zeros(n*s,n*s)
        for i in 1:s, j in 1:s
            J[n*(i-1)+1:n*i,n*(j-1)+1:n*j] .= -h*A[i,j]*jf(Y[:,j]) + (i==j ? Matrix{Float64}(I,n,n) : zeros(n,n))
        end
        Y -= reshape(J\vec(residual),n,s)
    end
    K = hcat([f(t+c[j]*h,Y[:,j]) for j in 1:s]...)
    @assert maximum(abs,Y .- y .- h*K*A') < 1e-13
    main = y+h*K*b
    embedded = y+h*(companion[1]*f(t,y)+K*companion[2:end])
    return Dict("state"=>main,"error"=>main-embedded)
end

function json(io,value)
    if value isa AbstractDict
        print(io,"{");for (i,k) in enumerate(sort(collect(keys(value))))
            i>1 && print(io,",");json(io,k);print(io,":");json(io,value[k])
        end;print(io,"}")
    elseif value isa AbstractMatrix
        json(io,[collect(row) for row in eachrow(value)])
    elseif value isa AbstractVector
        print(io,"[");for (i,v) in enumerate(value);i>1 && print(io,",");json(io,v);end;print(io,"]")
    elseif value isa AbstractString
        print(io,repr(value))
    else
        print(io,value)
    end
end
schemes = Dict()
for (name,A,b,c,companion) in (("gaussLegendre3",g.A,g.b,g.c,vcat(0.,g.b-g.e)),
                             ("radauIia3",A,b,c,radau_companion))
    schemes[name] = Dict("a"=>A,"b"=>b,"c"=>c,"companion_b"=>companion,
        "autonomous"=>step(A,b,c,companion,false),"nonautonomous"=>step(A,b,c,companion,true))
end
json(stdout,Dict("julia"=>string(VERSION),"monorepo_tree"=>tree,
    "firk_source_sha256"=>bytes2hex(sha256(source)),"firk_version"=>"2.8.1",
    "source_file"=>"lib/OrdinaryDiffEqFIRK/src/firk_tableaus.jl","schemes"=>schemes))
println()
