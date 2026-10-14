# Unchanged pinned stabilized RK constant-cache step bodies, stdlib only.
using LinearAlgebra, SHA, Pkg
root=ARGS[1];tree=bytes2hex(Pkg.GitTools.tree_hash(root))
@assert tree=="bae9b8e5f1d00e51c554648e5094e215164a19f8"
json_source=read(joinpath(@__DIR__,"../wave9/higher_stages.jl"),String)
include_string(Main,"function json(io,value)"*split(split(json_source,"function json(io,value)";limit=2)[2],"\nschemes=Dict()";limit=2)[1])
macro muladd(ex);esc(ex);end
module OrdinaryDiffEqCore
increment_nf!(stats,n)=(stats.nf+=n)
end
for name in (:RKL1ConstantCache,:RKL2ConstantCache,:RKG1ConstantCache,:RKG2ConstantCache)
    @eval mutable struct $name
        min_stage::Int;max_stage::Int;mdeg::Int
    end
end
mutable struct Stats
    nf::Int
end
mutable struct Integrator
    t::Float64;dt::Float64;uprev::Vector{Float64};u::Vector{Float64};f::Function;p::Any
    fsalfirst::Vector{Float64};fsallast::Vector{Float64};k::Vector{Vector{Float64}}
    alg::Any;stats::Stats;opts::Any;eigen_est::Float64
end
unwrap_alg(i,flag)=i.alg
file="lib/OrdinaryDiffEqStabilizedRK/src/rkc_perform_step.jl"
source=read(joinpath(root,file),String)
include_string(Main,split(split(source,"\nrkg1_b";limit=2)[2],"\n\n";limit=2)[1] |> s->"rkg1_b"*s)
for name in ("RKL1","RKL2","RKG1","RKG2")
    start="@muladd function perform_step!(integrator, cache::$(name)ConstantCache"
    stop="\n@muladd function perform_step!(integrator, cache::$(name)Cache"
    body=start*split(split(source,start;limit=2)[2],stop;limit=2)[1]
    include_string(Main,body,file)
end
records=Dict()
for name in ("RKL1","RKL2","RKG1","RKG2"), s in (3,5,9,21), forced in (false,true)
    times=Float64[]
    f=(y,p,t)->begin
        push!(times,t)
        [-2*y[1]+.1*y[2]^2+(forced ? .2*sin(t) : 0),-.5*y[2]+.2*y[1]^2+(forced ? .1*cos(t) : 0)]
    end
    t=.3;h=.1;y=[.7,-.4];f0=f(y,nothing,t)
    cache=getfield(Main,Symbol(name*"ConstantCache"))(s,s,0)
    i=Integrator(t,h,copy(y),copy(y),f,nothing,f0,zeros(2),[zeros(2),zeros(2)],
        (eigen_est=i->nothing,),Stats(1),(adaptive=false,),0.)
    perform_step!(i,cache,false)
    records[name*"_s$s"*(forced ? "_forced" : "_autonomous")]=Dict(
        "t"=>t,"h"=>h,"initial"=>y,"state"=>i.u,"rhs_times"=>times,"rhs_calls"=>i.stats.nf,
        "stages"=>cache.mdeg,"euler_difference"=>i.u-(y+h*f0))
end
json(stdout,Dict("monorepo_tree"=>tree,"source_file"=>file,
    "source_sha256"=>bytes2hex(sha256(source)),"records"=>records,
    "qualification"=>"Unchanged constant-cache steps, forced fixed stage counts and supplied zero spectral estimate; adaptive controller and internal spectral-radius estimator not reproduced"));println()
