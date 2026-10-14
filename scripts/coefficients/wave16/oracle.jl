using LinearAlgebra,SHA,Pkg
root=ARGS[1];tree=bytes2hex(Pkg.GitTools.tree_hash(root))
@assert tree=="bae9b8e5f1d00e51c554648e5094e215164a19f8"
helper=read(joinpath(@__DIR__,"../wave9/higher_stages.jl"),String)
include_string(Main,"function json(io,value)"*split(split(helper,"function json(io,value)";limit=2)[2],"\nschemes=Dict()";limit=2)[1])
json(io::IO,::Nothing)=print(io,"null")
abstract type OrdinaryDiffEqConstantCache end
macro muladd(ex);esc(ex);end
macro var".."(args...);esc(last(args));end
function name_of(sig)
 if sig isa Symbol return sig end
 if sig isa Expr && sig.head in (:where,:(::),:curly) return name_of(sig.args[1]) end
 if sig isa Expr && sig.head==:call return name_of(sig.args[1]) end
 return nothing
end
cache_file="lib/OrdinaryDiffEqLowStorageRK/src/low_storage_rk_caches.jl"
step_file="lib/OrdinaryDiffEqLowStorageRK/src/low_storage_rk_perform_step.jl"
utils_file="lib/OrdinaryDiffEqLowStorageRK/src/alg_utils.jl"
cache_source=read(joinpath(root,cache_file),String);step_source=read(joinpath(root,step_file),String)
utils_source=read(joinpath(root,utils_file),String)
for expr in Meta.parseall(cache_source).args
 expr isa LineNumberNode && continue
 if expr isa Expr && expr.head==:struct && occursin("ConstantCache",string(expr.args[2]))
  eval(expr)
 elseif expr isa Expr && expr.head==:function
  name=name_of(expr.args[1])
  if name!==nothing && endswith(string(name),"ConstantCache") eval(expr) end
 end
end
for original in Meta.parseall(step_source).args
 expr=original
 if expr isa Expr && expr.head==:macrocall && expr.args[1]==Symbol("@muladd") expr=expr.args[end] end
 if expr isa Expr && expr.head==:function && name_of(expr.args[1]) in (:initialize!,:perform_step!) && occursin("ConstantCache",string(expr.args[1]))
  eval(original)
 end
end
const ERROR=Ref{Any}(nothing)
calculate_residuals(value,args...)=(ERROR[]=copy(value);value)
module OrdinaryDiffEqCore
increment_nf!(stats,n)=(stats.nf+=n)
set_EEst!(i,value)=(i.EEst=value)
end
mutable struct Stats
 nf::Int
end
mutable struct Integrator
 t::Float64;dt::Float64;uprev::Vector{Float64};u::Vector{Float64};f::Function;p::Any
 fsalfirst::Vector{Float64};fsallast::Vector{Float64};k::Vector{Any};kshortsize::Int
 stats::Stats;opts::Any;derivative_discontinuity::Bool;EEst::Float64
end
function execute(name,phase;basis=false,forced=false)
 cache=getfield(Main,Symbol(name*"ConstantCache"))(Float64,Float64)
 if hasfield(typeof(cache),:step) cache.step=phase end
 initial=basis ? [1.;zeros(127)] : [.7,-.4]
 t=basis ? 0. : .3;h=basis ? 1. : .1;calls=Any[]
 f=(y,p,time)->begin
  if basis
   value=zeros(128);value[length(calls)+2]=1.
  else
   value=[-2*y[1]+.1*y[2]^2+(forced ? .2*sin(time) : 0),-.5*y[2]+.2*y[1]^2+(forced ? .1*cos(time) : 0)]
  end
  push!(calls,Dict("t"=>time,"state"=>copy(y),"value"=>copy(value)));value
 end
 ERROR[]=nothing
 i=Integrator(t,h,copy(initial),copy(initial),f,nothing,zero(initial),zero(initial),Any[],0,
  Stats(0),(adaptive=true,abstol=1.,reltol=0.,internalnorm=(x,t)->norm(x)),false,0.)
 initialize!(i,cache);perform_step!(i,cache,false)
 if basis
  count=length(calls)
  A=[calls[j]["state"][k+1] for j in 1:count,k in 1:count]
  return Dict("a"=>A,"b"=>i.u[2:count+1],"c"=>[call["t"] for call in calls],
   "error_weights"=>ERROR[]===nothing ? nothing : ERROR[][2:count+1],
   "initial_stage_scales"=>[call["state"][1] for call in calls],"initial_output_scale"=>i.u[1],
   "initial_error_scale"=>ERROR[]===nothing ? nothing : ERROR[][1],
   "cache"=>string(typeof(cache)),"rhs_calls"=>count,"reported_nf"=>i.stats.nf)
 end
 return Dict("initial"=>initial,"t"=>t,"h"=>h,"state"=>i.u,"error"=>ERROR[],"calls"=>calls,
  "rhs_calls"=>length(calls),"reported_nf"=>i.stats.nf)
end
function trajectory(name,count,mode)
 cache=getfield(Main,Symbol(name*"ConstantCache"))(Float64,Float64)
 u=[mode=="decay" ? 1. : 0.];h=1/count
 for j in 0:count-1
  f=(y,p,t)->mode=="decay" ? -y : fill(t,length(y))
  i=Integrator(j*h,h,copy(u),copy(u),f,nothing,zero(u),zero(u),Any[],0,
   Stats(0),(adaptive=false,abstol=1.,reltol=0.,internalnorm=(x,t)->norm(x)),false,0.)
  initialize!(i,cache);perform_step!(i,cache,false);u=i.u
 end
 return u[1]
end
methods=Dict();records=Dict();failures=Dict();trajectories=Dict()
for m in eachmatch(r"alg_order\(alg::(\w+)\) = (\d+)",utils_source)
 name=m.captures[1];order=parse(Int,m.captures[2])
 trajectories[name]=Dict(mode=>Dict(string(n)=>trajectory(name,n,mode) for n in (20,40,80)) for mode in ("decay","time_only"))
 for phase in (name=="SHLDDRK_2N" ? (1,2) : (1,))
  key=name*(name=="SHLDDRK_2N" ? "_phase$(phase)" : "")
  try
   methods[key]=merge(execute(name,phase;basis=true),Dict("order"=>order,"source_method"=>name,"phase"=>phase))
   for forced in (false,true)
    records[key*(forced ? "_forced" : "_autonomous")]=execute(name,phase;forced=forced)
   end
  catch e failures[key]=sprint(showerror,e) end
 end
end
json(stdout,Dict("monorepo_tree"=>tree,"methods"=>methods,"records"=>records,"failures"=>failures,"trajectories"=>trajectories,
 "source_sha256"=>Dict(cache_file=>bytes2hex(sha256(cache_source)),step_file=>bytes2hex(sha256(step_source)),utils_file=>bytes2hex(sha256(utils_source))),
 "qualification"=>"Unchanged constant-cache constructors/initialization/step bodies, no-op muladd and broadcast annotations, adaptive normalized-error scaffolding captured before scaling. Independent unit RHS bases recover full affine stage coefficients; duplicate initial/end RHS rows are preserved. Actual autonomous/forced nonlinear cases execute separately. Source alternating SHLDDRK_2N phases are separate records; no single fixed-map equivalence or low-storage backend claim."));println()
