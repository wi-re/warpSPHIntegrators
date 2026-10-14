# Pinned FIRK constructors, with high-precision coefficient arithmetic.
# Julia stdlib quadrature replaces FastGaussQuadrature; no SciML install needed.
using LinearAlgebra, SHA, Pkg
root=ARGS[1]
tree=bytes2hex(Pkg.GitTools.tree_hash(root))
@assert tree=="bae9b8e5f1d00e51c554648e5094e215164a19f8"
file=joinpath(root,"lib/OrdinaryDiffEqFIRK/src/firk_tableaus.jl")
source=read(file,String)
section(start,stop)=split(split(source,start;limit=2)[2],stop;limit=2)[1]
include_string(Main,"struct GaussLegendreTableau"*section("struct GaussLegendreTableau","import FastGaussQuadrature"),file)
include_string(Main,"function generateGaussLegendreTableau"*section("function generateGaussLegendreTableau","const GaussLegendreTableauCache"),file)
radau_head="function generateRadauTableau"*section("function generateRadauTableau","    local eigval, eigvec")
include_string(Main,radau_head*"    return a,c\nend",file)
function gausslegendre(::Type{T},n) where T
    x,v=eigen(SymTridiagonal(zeros(n),[k/sqrt(4k*k-1) for k in 1:n-1]))
    return T.(x),T.(2 .* v[1,:].^2)
end
gausslegendre(n)=gausslegendre(Float64,n)
prior=read(joinpath(@__DIR__,"oracle.jl"),String)
include_string(Main,"function step("*split(split(prior,"function step(";limit=2)[2],"\nfunction json(";limit=2)[1])
function gaussradau(::Type{T},n) where T
    diagonal=zeros(n);diagonal[end]=-n/(2n-1)
    x,v=eigen(SymTridiagonal(diagonal,[k/sqrt(4k*k-1) for k in 1:n-1]))
    x[1]=-1.
    return T.(x),T.(2 .* v[1,:].^2)
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
schemes=Dict()
for s in 2:9
    g=generateGaussLegendreTableau(BigFloat,BigFloat,s)
    ga,gb,gc,ge=Float64.(g.A),Float64.(g.b),Float64.(g.c),vcat(0.,Float64.(g.b-g.e))
    schemes["gauss$s"]=Dict("a"=>ga,"b"=>gb,"c"=>gc,"companion_b"=>ge,
        "autonomous"=>step(ga,gb,gc,ge,false),"nonautonomous"=>step(ga,gb,gc,ge,true))
    a,c=generateRadauTableau(BigFloat,BigFloat,s)
    # Derived unsmoothed companion: null-stage weight -1 and Lagrange
    # extrapolation of f(0) using the collocation stages.
    l0=[prod(-c[k]/(c[j]-c[k]) for k in 1:s if k!=j) for j in 1:s]
    b=vec(a[end,:])
    ra,rb,rc,re=Float64.(a),Float64.(b),Float64.(c),vcat(-1.,Float64.(b+l0))
    schemes["radau$s"]=Dict("a"=>ra,"b"=>rb,"c"=>rc,"companion_b"=>re,
        "autonomous"=>step(ra,rb,rc,re,false),"nonautonomous"=>step(ra,rb,rc,re,true))
end
json(stdout,Dict("julia"=>string(VERSION),"monorepo_tree"=>tree,
    "firk_source_sha256"=>bytes2hex(sha256(source)),"firk_version"=>"2.8.1",
    "coefficient_arithmetic"=>"BigFloat; stdlib Golub-Welsch Float64 nodes converted to BigFloat",
    "radau_qualification"=>"Constructor evaluated through a,c construction before eigenbasis transform; derived unsmoothed companion, not source filter/controller",
    "schemes"=>schemes));println()
