using LinearAlgebra,SHA,Pkg
root=ARGS[1];tree=bytes2hex(Pkg.GitTools.tree_hash(root))
@assert tree=="bae9b8e5f1d00e51c554648e5094e215164a19f8"
json_source=read(joinpath(@__DIR__,"../wave9/higher_stages.jl"),String)
include_string(Main,"function json(io,value)"*split(split(json_source,"function json(io,value)";limit=2)[2],"\nschemes=Dict()";limit=2)[1])
abstract type OrdinaryDiffEqConstantCache end
const CompiledFloats=Union{Float32,Float64}
tableau_file="lib/OrdinaryDiffEqRKN/src/rkn_tableaus.jl"
source=read(joinpath(root,tableau_file),String)
include_string(Main,split(source,"\nstruct IRKN3ConstantCache";limit=2)[1],tableau_file)
include_string(Main,"struct DPRKN6Tableau"*split(source,"struct DPRKN6Tableau";limit=2)[2],tableau_file)
macro muladd(ex);esc(ex);end
struct ArrayPartition
    x::Tuple
end
ArrayPartition(a,b)=ArrayPartition((a,b))
struct SplitForce
    f1::Function;f2::Function
end
struct NystromVIConstantCache
    tab::Any
end
struct NystromVDConstantCache
    tab::Any
end
mutable struct Stats
    nf::Int;nf2::Int
end
module OrdinaryDiffEqCore
increment_nf!(stats,n)=(stats.nf+=n)
end
mutable struct Integrator
    t::Float64;dt::Float64;f::SplitForce;p::Any;uprev::ArrayPartition;u::ArrayPartition
    fsalfirst::ArrayPartition;fsallast::ArrayPartition;k::Vector{Any};stats::Stats;opts::Any
end
_store_vi_k!(integrator,tab,ks)=nothing
step_file="lib/OrdinaryDiffEqRKN/src/rkn_perform_step.jl"
steps=read(joinpath(root,step_file),String)
for kind in ("VI","VD")
    start="@muladd function perform_step!(\n        integrator, cache::Nystrom$(kind)ConstantCache"
    stop="\n@muladd function perform_step!(\n        integrator, cache::Nystrom$(kind)Cache"
    include_string(Main,start*split(split(steps,start;limit=2)[2],stop;limit=2)[1],step_file)
end
names=["DPRKN4","DPRKN5","DPRKN6","DPRKN6FM","DPRKN8","DPRKN12","ERKN4","ERKN5","ERKN7",
    "Nystrom4VelocityIndependent","Nystrom5VelocityIndependent","RKN4","Nystrom4","FineRKN4","FineRKN5"]
tableaus=Dict();records=Dict()
for name in names
    tab=getfield(Main,Symbol(name*"Tableau"))(Float64,Float64)
    vd=hasfield(typeof(tab),:abar)
    data=Dict{String,Any}(string(key)=>getfield(tab,key) for key in (:a,:b,:bp,:btilde,:bptilde,:c))
    data["velocity_dependent"]=vd;data["pos_only_error"]=hasfield(typeof(tab),:pos_only_error) && tab.pos_only_error
    if vd data["abar"]=tab.abar end
    tableaus[name]=data
    for forced in (false,true)
        calls=Any[]
        acceleration=(v,q,p,t)->begin
            result=-2*q+.1*q.^2+(forced ? [.2*sin(t),.1*cos(t)] : zeros(2))
            if vd result+=-.3*v+.05*q.*v+.1*v.^2 end
            push!(calls,Dict("t"=>t,"q"=>copy(q),"v"=>copy(v),"a"=>copy(result)))
            result
        end
        f=SplitForce(acceleration,(v,q,p,t)->v);t=.3;h=.1;q=[.7,-.4];v=[-.2,.3]
        first=ArrayPartition((acceleration(v,q,nothing,t),copy(v)))
        cache=vd ? NystromVDConstantCache(tab) : NystromVIConstantCache(tab)
        i=Integrator(t,h,f,nothing,ArrayPartition((copy(v),copy(q))),ArrayPartition((copy(v),copy(q))),
            first,first,Any[first,first,first],Stats(1,1),(adaptive=false,))
        perform_step!(i,cache,false)
        n=length(tab.b)
        error_q=isempty(tab.btilde) ? Float64[] : h*h*sum(tab.btilde[j]*calls[j]["a"] for j in 1:n)
        error_v=isempty(tab.bptilde) ? Float64[] : h*sum(tab.bptilde[j]*calls[j]["a"] for j in 1:n)
        records[name*(forced ? "_forced" : "_autonomous")]=Dict("initial_q"=>q,"initial_v"=>v,"t"=>t,"h"=>h,
            "q"=>i.u.x[2],"v"=>i.u.x[1],"error_q"=>error_q,"error_v"=>error_v,"calls"=>calls)
    end
end
json(stdout,Dict("monorepo_tree"=>tree,"tableaus"=>tableaus,"records"=>records,
    "source_sha256"=>Dict(tableau_file=>bytes2hex(sha256(source)),step_file=>bytes2hex(sha256(steps))),
    "qualification"=>"Pinned tableau constructors and unchanged constant-cache step bodies; independent scaffolding, no FSAL/controller/dense-output equivalence; difference weights applied independently to recorded source stages"));println()
