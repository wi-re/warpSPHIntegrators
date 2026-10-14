# Actual pinned IRKN kernels, with source cache layouts and minimal step harness.
using LinearAlgebra,SHA,Pkg
root=ARGS[1];tree=bytes2hex(Pkg.GitTools.tree_hash(root))
@assert tree=="bae9b8e5f1d00e51c554648e5094e215164a19f8"
helper=read(joinpath(@__DIR__,"../wave9/higher_stages.jl"),String)
include_string(Main,"function json(io,value)"*split(split(helper,"function json(io,value)";limit=2)[2],"\nschemes=Dict()";limit=2)[1])
json(io::IO,::Nothing)=print(io,"null")
abstract type OrdinaryDiffEqConstantCache end
abstract type OrdinaryDiffEqMutableCache end
abstract type NystromConstantCache <: OrdinaryDiffEqConstantCache end
abstract type NystromMutableCache <: OrdinaryDiffEqMutableCache end
const CompiledFloats=Union{Float32,Float64}
macro muladd(ex);esc(ex);end
macro cache(ex);esc(ex);end
macro tight_loop_macros(ex);esc(ex);end
macro var".."(args...)
 ex=last(args)
 return esc(:(@. $ex))
end
struct ArrayPartition
 x::Tuple
end
Base.zero(x::ArrayPartition)=ArrayPartition(map(zero,x.x))
Base.copy(x::ArrayPartition)=ArrayPartition(map(copy,x.x))
struct SplitForce
 f1::Function;f2::Function
end
mutable struct Stats
 nf::Int;nf2::Int
end
module OrdinaryDiffEqCore
increment_nf!(stats,n)=(stats.nf+=n)
end
mutable struct Integrator
 t::Float64;dt::Float64;f::SplitForce;p::Any;uprev::ArrayPartition;u::ArrayPartition
 uprev2::ArrayPartition;fsalfirst::ArrayPartition;fsallast::ArrayPartition
 k::Vector{Any};stats::Stats;cache::Any;iter::Int;derivative_discontinuity::Bool;tprev::Float64
end
function name_of(sig)
 sig isa Symbol && return sig
 sig isa Expr && sig.head in (:where,:(::),:curly) && return name_of(sig.args[1])
 sig isa Expr && sig.head==:call && return name_of(sig.args[1])
 return nothing
end
files=["lib/OrdinaryDiffEqRKN/src/"*f for f in ("rkn_tableaus.jl","rkn_caches.jl","rkn_perform_step.jl","alg_utils.jl","algorithms.jl")]
sources=Dict(f=>read(joinpath(root,f),String) for f in files)
for expr in Meta.parseall(sources[files[1]]).args
 if expr isa Expr && expr.head==:struct && occursin(r"IRKN[34]ConstantCache",string(expr.args[2]))
  eval(expr)
 elseif expr isa Expr && expr.head==:function && name_of(expr.args[1]) in (:IRKN3ConstantCache,:IRKN4ConstantCache)
  eval(expr)
 end
end
for original in Meta.parseall(sources[files[2]]).args
 expr=original
 expr isa Expr && expr.head==:macrocall && expr.args[1]==Symbol("@cache") && (expr=expr.args[end])
 if expr isa Expr && expr.head==:struct && occursin(r"^(IRKN[34]Cache|Nystrom4VelocityIndependentCache|Nystrom4VelocityIndependentConstantCache)(\{|\s|$)",string(expr.args[2]))
  eval(original)
 end
end
for original in Meta.parseall(sources[files[3]]).args
 expr=original
 expr isa Expr && expr.head==:macrocall && expr.args[1]==Symbol("@muladd") && (expr=expr.args[end])
 if expr isa Expr && expr.head==:function && name_of(expr.args[1])==:perform_step! && occursin(r"cache::(IRKN[34](Constant)?Cache|Nystrom4VelocityIndependent(Constant)?Cache)",string(expr.args[1]))
  eval(original)
 end
end
function setup(name,h,problem;constant=false)
 q=[1.];v=[.7];calls=Ref(0);calls2=Ref(0)
 value=ArrayPartition((copy(v),copy(q)));k1=zero(value);k2=zero(value);k3=zero(value);k=zero(value);tmp=zero(value);tmp2=zero(value)
 tab=getfield(Main,Symbol(name*"ConstantCache"))(Float64,Float64)
 # These allocations and alias relationships mirror source alg_cache(Val(true)).
 if name=="IRKN3"
  start=Nystrom4VelocityIndependentCache(value,value,k1,k2.x[2],k3.x[2],k,tmp)
  cache=IRKN3Cache(value,value,value,k1,k2,k,tmp,k3,start,tab)
 else
  start=Nystrom4VelocityIndependentCache(value,value,k1,k2.x[2],k3.x[2],k,tmp)
  cache=IRKN4Cache(value,value,value,k1,k2,k3,k,tmp,tmp2,start,tab)
 end
 if constant;cache=tab;end
 f=SplitForce((out,v,q,p,t)->begin
  calls[]+=1
  out .= problem=="autonomous" ? -q : problem=="time_only" ? fill(t,length(q)) : -q .+ .1*t
 end,(out,v,q,p,t)->begin calls2[]+=1;out .= v;end)
 i=Integrator(0.,h,f,nothing,copy(value),copy(value),copy(value),k1,k,Any[k1,k],Stats(0,0),cache,1,false,0.)
 f.f1(k1.x[1],v,q,nothing,0.);f.f2(k1.x[2],v,q,nothing,0.)
 return i,calls,calls2
end
function run(name,count,problem;reseed=false)
 h=1/count;i,calls,calls2=setup(name,h,problem)
 for step in 1:count
  i.iter=step;i.t=(step-1)*h;i.tprev=max(0.,i.t-h);i.u=copy(i.uprev)
  perform_step!(i,i.cache,false)
  if reseed && step==1
   tab=i.cache.tab;v0,q0=i.uprev.x;a0=zero(q0)
   i.f.f1(a0,v0,q0,nothing,0.)
   copyto!(i.cache.tmp2.x[2],a0)
   copyto!(i.cache.tmp2.x[1],i.cache.k.x[1])
   qm=q0+h*tab.c1*v0+h*h*tab.a21*a0
   i.f.f1(i.cache.k₂.x[1],v0,qm,nothing,tab.c1*h)
   if name=="IRKN4"
    qm=q0+h*tab.c2*v0+h*h*tab.a32*i.cache.k₂.x[1]
    i.f.f1(i.cache.k₃.x[1],v0,qm,nothing,tab.c2*h)
   end
  end
  old=copy(i.uprev);i.uprev=copy(i.u);i.uprev2=old
  for j in 1:2;copyto!(i.fsalfirst.x[j],i.cache.k.x[j]);end
 end
 q=i.u.x[2][1];v=i.u.x[1][1]
 qe=problem=="autonomous" ? cos(1)+.7*sin(1) : problem=="time_only" ? 1+.7+1/6 : cos(1)+.6*sin(1)+.1
 ve=problem=="autonomous" ? -sin(1)+.7*cos(1) : problem=="time_only" ? .7+.5 : -sin(1)+.6*cos(1)+.1
 return Dict("steps"=>count,"q"=>q,"v"=>v,"exact_q"=>qe,"exact_v"=>ve,"position_error"=>abs(q-qe),"velocity_error"=>abs(v-ve),"acceleration_calls"=>calls[],"velocity_calls"=>calls2[],"rhs_reported"=>i.stats.nf,"rhs2_reported"=>i.stats.nf2)
end
function fitted_order(rows,key)
 for i in length(rows):-1:2
  if min(rows[i-1][key],rows[i][key])>1e-11
   return log2(rows[i-1][key]/rows[i][key])
  end
 end
 return nothing
end
records=Dict();constant_errors=Dict()
for name in ("IRKN3","IRKN4")
 for problem in ("autonomous","time_only","manufactured"), reseed in (false,true)
  rows=[run(name,n,problem;reseed) for n in (20,40,80,160)]
  records[name*"_"*problem*(reseed ? "_reseeded_control" : "_source")]=Dict("runs"=>rows,"position_order"=>fitted_order(rows,"position_error"),"velocity_order"=>fitted_order(rows,"velocity_error"))
 end
 i,_,_=setup(name,.1,"autonomous";constant=true)
 constant_errors[name]=try
  perform_step!(i,i.cache,false);"unexpected success"
 catch error
  sprint(showerror,error)
 end
end
@assert occursin("no field `k₂`",constant_errors["IRKN3"])
@assert startswith(constant_errors["IRKN4"],"MethodError")
for name in ("IRKN3","IRKN4"), problem in ("time_only","manufactured")
 r=records[name*"_"*problem*"_source"]
 @assert 1.9<r["position_order"]<2.1 && 1.9<r["velocity_order"]<2.1
end
for name in ("IRKN3","IRKN4")
 @assert maximum(r["velocity_error"] for r in records[name*"_time_only_reseeded_control"]["runs"])<1e-13
end
json(stdout,Dict("monorepo_tree"=>tree,"package_version"=>"2.2.0","initial_q"=>[1.],"initial_v"=>[.7],"end"=>1.,
 "fit_policy"=>"Last adjacent pair with both component errors above 1e-11; null if none. Raw errors remain available; pre-asymptotic fits are not formal order proofs.",
 "problems"=>Dict("autonomous"=>"q''=-q; q=cos(t)+.7*sin(t)","time_only"=>"q''=t; q=1+.7*t+t^3/6","manufactured"=>"q''=-q+.1*t; q=cos(t)+.6*sin(t)+.1*t"),
 "constant_cache_errors"=>constant_errors,
 "control"=>"Reseeded controls replace only first-step cached history with physical old/end accelerations and correctly timed old substage values; subsequent source kernels are unchanged. They are diagnostics, not an implemented package method.",
 "records"=>records,"source_sha256"=>Dict(f=>bytes2hex(sha256(sources[f])) for f in files),
 "qualification"=>"Actual coefficient constructors/cache structs and step bodies; cache allocation/aliasing and accepted-state/FSAL advancement mirrored in a stdlib harness. Broadcast/muladd/thread annotations replaced by ordinary Julia broadcast/arithmetic. Constant-cache failures and mutable full-startup refinement are source-path findings, not a claim about every implementation of the mathematical IRKN algorithms."));println()
