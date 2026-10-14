using SHA,Pkg
root=ARGS[1];tree=bytes2hex(Pkg.GitTools.tree_hash(root))
@assert tree=="bae9b8e5f1d00e51c554648e5094e215164a19f8"
helper=read(joinpath(@__DIR__,"../wave9/higher_stages.jl"),String)
include_string(Main,"function json(io,value)"*split(split(helper,"function json(io,value)";limit=2)[2],"\nschemes=Dict()";limit=2)[1])
macro muladd(ex);esc(ex);end
struct FunctionMap{S} end
struct FunctionMapConstantCache end
FunctionMap_scale_by_time(::FunctionMap{S}) where {S}=S
module SciMLBase
const DISCRETE_OUTOFPLACE_DEFAULT=:unused_default
end
module DiffEqBase
struct EvalFunc
 f::Any
end
end
module OrdinaryDiffEqCore
increment_nf!(stats,n)=(stats.nf+=n)
end
mutable struct Stats
 nf::Int
end
mutable struct Integrator
 uprev::Vector{Float64};u::Vector{Float64};dt::Float64;t::Float64;f::Function;p::Any;alg::Any;stats::Stats
end
unwrap_alg(i,args...)=i.alg
file="lib/OrdinaryDiffEqFunctionMap/src/fixed_timestep_perform_step.jl";source=read(joinpath(root,file),String)
start="function perform_step!(integrator, cache::FunctionMapConstantCache, repeat_step = false)"
body=start*split(split(source,start;limit=2)[2],"\nfunction initialize!";limit=2)[1]
body=replace(replace(body,r"@\.\.\s+broadcast\s*=\s*false\s+"=>""),r"@\.\.\s+"=>"")
include_string(Main,body,file)
records=Dict()
for scaled in (false,true), nonlinear in (false,true), h in (.05,.2)
 calls=Any[];initial=[.7,-.4];t=.3
 f=(y,p,time)->begin
  result=[.9*y[1]+.1*y[2]+.2*sin(time),-.2*y[1]+.8*y[2]+.1*cos(time)]
  if nonlinear result+=[.05*y[1]*y[2],.1*y[1]^2] end
  push!(calls,Dict("t"=>time,"input"=>copy(y),"value"=>copy(result)));result
 end
 i=Integrator(initial,copy(initial),h,t,f,nothing,FunctionMap{scaled}(),Stats(0))
 perform_step!(i,FunctionMapConstantCache(),false)
 records["scaled$(scaled)_nonlinear$(nonlinear)_h$(h)"]=Dict("scaled"=>scaled,"nonlinear"=>nonlinear,
  "h"=>h,"t"=>t,"initial"=>initial,"state"=>i.u,"calls"=>calls,"nf"=>i.stats.nf)
end
json(stdout,Dict("monorepo_tree"=>tree,"source_sha256"=>Dict(file=>bytes2hex(sha256(source))),"records"=>records,
 "qualification"=>"Unchanged FunctionMap constant-cache step, broadcast/muladd annotations removed, default-map sentinel/EvalFunc/stats scaffolding. Both direct and scale_by_time branches recorded. Package scaled behavior is an explicit map wrapper, not a public overloaded RHS flag. No source limiter/default-identity machinery claim."));println()
