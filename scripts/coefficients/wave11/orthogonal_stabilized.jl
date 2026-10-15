# Actual fixed-stage ROCK/SERK/RKMC kernels, original and translated coordinates.
using LinearAlgebra,SHA,Pkg
root=ARGS[1];tree=bytes2hex(Pkg.GitTools.tree_hash(root))
@assert tree=="bae9b8e5f1d00e51c554648e5094e215164a19f8"
helper=read(joinpath(@__DIR__,"../wave9/higher_stages.jl"),String)
include_string(Main,"function json(io,value)"*split(split(helper,"function json(io,value)";limit=2)[2],"\nschemes=Dict()";limit=2)[1])
json(io::IO,value::Tuple)=json(io,collect(value))
abstract type OrdinaryDiffEqConstantCache end
abstract type OrdinaryDiffEqMutableCache end
macro muladd(ex);esc(ex);end
struct SERK2
 eigen_est::Function
end
struct ESERK4 end
struct ESERK5 end
struct BoundAlgorithm
 eigen_est::Function
end
mutable struct Stats
 nf::Int
end
module OrdinaryDiffEqCore
increment_nf!(s,n)=(s.nf+=n)
set_EEst!(i,value)=(i.EEst=value)
end
const last_error=Ref{Any}(nothing)
function calculate_residuals(error,args...)
 last_error[]=copy(error);return error
end
value(x)=x
mutable struct Oracle
 t::Float64;dt::Float64;uprev::Vector{Float64};u::Vector{Float64};f::Function;p::Nothing
 fsalfirst::Vector{Float64};fsallast::Vector{Float64};opts::Any;stats::Stats;k::Vector{Any}
 EEst::Float64;eigen_est::Float64;alg::Any
end
unwrap_alg(i,b)=i.alg
maybe_maxeig!(i,cache,alg)=nothing
maxeig!(i,cache)=nothing
function name_of(sig)
 sig isa Symbol && return sig
 sig isa Expr && sig.head in (:where,:(::),:curly) && return name_of(sig.args[1])
 sig isa Expr && sig.head==:call && return name_of(sig.args[1])
 return nothing
end
base="lib/OrdinaryDiffEqStabilizedRK/"
files=[base*"src/"*f for f in ("rkc_caches.jl","rkc_perform_step.jl","rkc_utils.jl",
 "rkc_tableaus_rock2.jl","rkc_tableaus_rock4.jl","rkc_tableaus_serk2.jl","algorithms.jl","alg_utils.jl")]
push!(files,base*"Project.toml")
sources=Dict(f=>read(joinpath(root,f),String) for f in files)
for expr in Meta.parseall(sources[files[1]]).args
 if expr isa Expr && expr.head==:struct && occursin(r"^(ROCK2|ROCK4|SERK2|RKMC2)ConstantCache",string(expr.args[2]))
  eval(expr)
 end
end
for file in files[4:6];include_string(Main,sources[file]);end
for original in Meta.parseall(sources[files[3]]).args
 expr=original
 expr isa Expr && expr.head==:macrocall && (expr=expr.args[end])
 if expr isa Expr && expr.head==:function && name_of(expr.args[1]) in (:choosedeg!,:choosedeg_SERK!)
  eval(original)
 end
end
for original in Meta.parseall(sources[files[2]]).args
 expr=original
 expr isa Expr && expr.head==:macrocall && expr.args[1]==Symbol("@muladd") && (expr=expr.args[end])
 if expr isa Expr && expr.head==:function && name_of(expr.args[1])==:perform_step! &&
    occursin(r"cache::(ROCK2|ROCK4|SERK2|RKMC2)ConstantCache",string(expr.args[1]))
  eval(original)
 end
end
function setup_cache(name,stages,y)
 if name=="ROCK2" || name=="ROCK4"
  cache=getfield(Main,Symbol(name*"ConstantCache"))(Float64,Float64,copy(y))
  cache.min_stage=stages;cache.max_stage=stages
 elseif name=="SERK2"
  cache=SERK2ConstantCache(Float64,copy(y))
 else
  cache=RKMC2ConstantCache(copy(y),0,stages,stages,0.,0.)
 end
 return cache
end
function run(name,stages,problem,h,translated;linear_z=nothing)
 y=[1.,.4];anchor=translated ? copy(y) : zero(y);u0=y-anchor;calls=[]
 f=(u,p,t)->begin
  q=u+anchor
  rate=linear_z!==nothing ? (linear_z/h)*q : problem=="zero" ? zero(q) : [-2*q[1]+.1*q[1]*q[2]+(problem=="forced" ? .2*sin(t) : 0),
   -3*q[2]+.2*q[1]^2+(problem=="forced" ? .1*cos(t) : 0)]
  push!(calls,Dict("t"=>t,"state"=>q,"rate"=>rate));return rate
 end
 cache=setup_cache(name,stages,y)
 alg=name=="SERK2" ? SERK2(i->nothing) : BoundAlgorithm(i->nothing)
 spectral=name=="SERK2" ? .8*(stages-.5)^2/abs(h) : 0.
 opts=(adaptive=true,abstol=1.,reltol=0.,internalnorm=(x,t=0.)->norm(x))
 i=Oracle(.3,h,copy(u0),copy(u0),f,nothing,f(u0,nothing,.3),zeros(2),opts,Stats(0),Any[zeros(2),zeros(2)],0.,spectral,alg)
 last_error[]=nothing;perform_step!(i,cache,false)
 record=Dict("method"=>name,"requested_stages"=>stages,"degree"=>cache.mdeg,"problem"=>problem,
  "h"=>h,"t"=>i.t,"initial"=>y,"output"=>i.u+anchor,"error"=>last_error[],"calls"=>calls,
  "w0"=>name=="RKMC2" ? cache.w0 : 0.,"w1"=>name=="RKMC2" ? cache.w1 : 0.,
  "translated"=>translated)
 linear_z!==nothing && (record["linear_z"]=linear_z)
 return record
end
tables=Dict()
for name in ("ROCK2","ROCK4","SERK2")
 cache=setup_cache(name,10,[1.,.4])
 keys=name=="ROCK2" ? (:ms,:fp1,:fp2,:recf) : name=="ROCK4" ? (:ms,:fpa,:fpb,:fpbe,:recf) : (:ms,:Bᵢ)
 tables[name]=Dict(string(key)=>getfield(cache,key) for key in keys)
end
roots=[]
for stages in 3:1000
 y=zeros(2);cache=setup_cache("RKMC2",stages,y);f=(u,p,t)->zero(u)
 opts=(adaptive=false,abstol=1.,reltol=0.,internalnorm=(x,t=0.)->norm(x))
 i=Oracle(.3,.125,copy(y),copy(y),f,nothing,zero(y),zero(y),opts,Stats(0),Any[zero(y),zero(y)],0.,0.,BoundAlgorithm(i->nothing))
 perform_step!(i,cache,false)
 @assert isfinite(cache.w0) && isfinite(cache.w1)
 push!(roots,[cache.w0,cache.w1])
end
tables["RKMC2"]=Dict("ms"=>collect(3:1000),"roots"=>roots)
records=[]
for name in ("ROCK2","ROCK4","SERK2","RKMC2")
 stages=name=="ROCK2" ? [tables[name]["ms"][i]+2 for i in (1,3,10,20,46)] :
  name=="ROCK4" ? [tables[name]["ms"][i]+4 for i in (1,3,10,20,47,49)] : name=="SERK2" ? [10,30,100,250] : [3,10,30,100,1000]
 for s in stages,problem in ("autonomous","forced","zero"),translated in (false,true)
  push!(records,run(name,s,problem,.125,translated))
 end
end
counterexamples=[run(name,s,"linear",.125,true;linear_z=z) for (name,s,z) in (
 ("ROCK4",133,-8.254930664062499),("ROCK4",152,-7974.05622644043),
 ("ROCK4",152,-8.401918017786704),
 ("SERK2",150,-17760.8),("SERK2",250,-49600.8))]
for record in counterexamples
 @assert abs(record["output"][1])>1.001
end
json(stdout,Dict("monorepo_tree"=>tree,"package_version"=>match(r"version = \"([^\"]+)\"",sources[files[end]]).captures[1],"tableaus"=>tables,"records"=>records,
 "stability_counterexamples"=>counterexamples,
 "source_sha256"=>Dict(f=>bytes2hex(sha256(sources[f])) for f in files),
 "qualification"=>"Actual coefficient constructors, cache structs, degree selectors and constant-cache kernels with fixed supplied stage counts. Spectral estimation is bypassed, not validated. Translated controls run the unchanged kernel on increments with RHS f(t,anchor+increment), then add anchor; this exposes affine rounding/zero-mode defects without changing the exact mathematical map. No source controller, mutable-cache or native-storage claim."));println()
