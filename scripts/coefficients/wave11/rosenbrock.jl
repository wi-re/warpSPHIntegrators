# Actual pinned constant-cache Rosenbrock kernels with exact dense J and f_t.
using LinearAlgebra,SHA,Pkg
root=ARGS[1];tree=bytes2hex(Pkg.GitTools.tree_hash(root))
@assert tree=="bae9b8e5f1d00e51c554648e5094e215164a19f8"
helper=read(joinpath(@__DIR__,"../wave9/higher_stages.jl"),String)
include_string(Main,"function json(io,value)"*split(split(helper,"function json(io,value)";limit=2)[2],"\nschemes=Dict()";limit=2)[1])
json(io::IO,::Nothing)=print(io,"null")
macro muladd(ex);esc(ex);end
macro var".."(args...);esc(:(@. $(last(args))));end
abstract type OrdinaryDiffEqConstantCache end
struct Rosenbrock23ConstantCache <: OrdinaryDiffEqConstantCache
 c₃₂::Float64;d::Float64;tf::Nothing;uf::Nothing
end
struct Rosenbrock32ConstantCache <: OrdinaryDiffEqConstantCache
 c₃₂::Float64;d::Float64;tf::Nothing;uf::Nothing
end
struct RosenbrockCombinedConstantCache{T} <: OrdinaryDiffEqConstantCache
 tab::T;tf::Nothing;uf::Nothing
end
struct Force
 mass_matrix::UniformScaling{Bool};forced::Bool;time_only::Bool
end
Force(m,forced)=Force(m,forced,false)
(f::Force)(y,p,t)=f.time_only ? fill(t,length(y)) : [-2*y[1]+.1*y[1]*y[2]+(f.forced ? .2*sin(t) : 0),
 -3*y[2]+.2*y[1]^2+(f.forced ? .1*cos(t) : 0)]
mutable struct Stats
 nf::Int;nsolve::Int
end
module OrdinaryDiffEqCore
increment_nf!(s,n)=(s.nf+=n)
set_EEst!(i,v)=(i.EEst=v)
end
mutable struct Oracle
 t::Float64;dt::Float64;uprev::Vector{Float64};u::Vector{Float64};f::Force;p::Nothing
 fsalfirst::Vector{Float64};fsallast::Vector{Float64};opts::Any;stats::Stats;k::Vector{Any};EEst::Float64
end
const last_error=Ref{Any}(nothing)
function calculate_residuals(error,args...)
 last_error[]=copy(error);return error
end
_vec(x)=x
_restructure_state(template,x)=x
issuccess_W(W)=true
function calc_rosenbrock_differentiation(i,cache,hgamma,repeat)
 y=i.uprev
 J=i.f.time_only ? zeros(2,2) : [-2+.1*y[2] .1*y[1];.4*y[1] -3]
 ft=i.f.time_only ? ones(2) : i.f.forced ? [.2*cos(i.t),-.1*sin(i.t)] : zeros(2)
 return ft,J-I/hgamma
end
files=["lib/OrdinaryDiffEqRosenbrock/src/"*f for f in ("rosenbrock_tableaus.jl","rosenbrock_perform_step.jl","alg_utils.jl","algorithms.jl")]
push!(files,"lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl")
push!(files,"lib/OrdinaryDiffEqDifferentiation/src/derivative_utils.jl")
sources=Dict(f=>read(joinpath(root,f),String) for f in files)
function name_of(sig)
 sig isa Symbol && return sig
 sig isa Expr && sig.head in (:where,:(::),:curly) && return name_of(sig.args[1])
 sig isa Expr && sig.head==:call && return name_of(sig.args[1])
 return nothing
end
for original in Meta.parseall(sources[files[1]]).args
 if original isa Expr && ((original.head==:struct && occursin(r"Rosenbrock(23|32)Tableau",string(original.args[2]))) ||
   (original.head==:function && name_of(original.args[1]) in (:Rosenbrock23Tableau,:Rosenbrock32Tableau)))
  eval(original)
 end
end
for original in Meta.parseall(sources[files[end-1]]).args
 expr=original
 expr isa Expr && expr.head==:macrocall && string(expr.args[1])=="Core.var\"@doc\"" && (expr=expr.args[end])
 if expr isa Expr && ((expr.head==:struct && occursin("RodasTableau{",string(expr.args[2]))) ||
   (expr.head==:function && name_of(expr.args[1])==:RosenbrockW6S4OSRodasTableau))
  eval(original)
 end
end
kernels=[]
for original in Meta.parseall(sources[files[2]]).args
 expr=original
 expr isa Expr && expr.head==:macrocall && expr.args[1]==Symbol("@muladd") && (expr=expr.args[end])
 if expr isa Expr && expr.head==:function && name_of(expr.args[1])==:perform_step! &&
   occursin(r"cache::(Rosenbrock23ConstantCache|Rosenbrock32ConstantCache|RosenbrockCombinedConstantCache)",string(expr.args[1]))
  eval(original);push!(kernels,original)
 end
end
tables=Dict();records=Dict()
for name in ("Rosenbrock23","Rosenbrock32","RosenbrockW6S4OS")
 if name=="RosenbrockW6S4OS"
  tab=RosenbrockW6S4OSRodasTableau(Float64,Float64)
  cache=RosenbrockCombinedConstantCache(tab,nothing,nothing)
  tables[name]=Dict("A"=>[collect(r) for r in eachrow(tab.A)],"C"=>[collect(r) for r in eachrow(tab.C)],
   "gamma"=>tab.gamma,"c"=>tab.c,"d"=>tab.d,"b"=>tab.b,"btilde"=>tab.btilde)
 else
  tab=getfield(Main,Symbol(name*"Tableau"))(Float64)
  cache=getfield(Main,Symbol(name*"ConstantCache"))(tab.c₃₂,tab.d,nothing,nothing)
  tables[name]=Dict("c32"=>tab.c₃₂,"gamma"=>tab.d)
 end
 rows=[]
 for forced in (false,true),h in (.2,.05)
  y=[1.,.4];f=Force(I,forced);opts=(adaptive=true,abstol=1.,reltol=0.,internalnorm=norm,calck=false)
  i=Oracle(.3,h,copy(y),copy(y),f,nothing,f(y,nothing,.3),zeros(2),opts,Stats(0,0),Any[zeros(2),zeros(2)],0.)
  last_error[]=nothing;perform_step!(i,cache,false)
  push!(rows,Dict("forced"=>forced,"t"=>i.t,"h"=>h,"initial"=>y,"output"=>i.u,"error"=>last_error[]))
 end
 records[name]=rows
end
function time_force_records()
 rows=Dict()
 for name in ("Rosenbrock23","Rosenbrock32")
  tab=getfield(Main,Symbol(name*"Tableau"))(Float64)
  cache=getfield(Main,Symbol(name*"ConstantCache"))(tab.c₃₂,tab.d,nothing,nothing)
  values=[]
  for h in (.2,.1,.05)
   y=zeros(2);f=Force(I,false,true);opts=(adaptive=true,abstol=1.,reltol=0.,internalnorm=norm,calck=false)
   i=Oracle(0.,h,copy(y),copy(y),f,nothing,f(y,nothing,0.),zeros(2),opts,Stats(0,0),Any[zeros(2),zeros(2)],0.)
   last_error[]=nothing;perform_step!(i,cache,false)
   push!(values,Dict("h"=>h,"output"=>i.u,"defect"=>last_error[],"exact"=>fill(h*h/2,2)))
  end
  rows[name]=values
 end
 return rows
end
source_time_force=time_force_records()
function corrected_time(ex)
 ex==:(dt*dT) && return :(dtγ*dT)
 ex isa Expr && return Expr(ex.head,map(corrected_time,ex.args)...)
 return ex
end
for kernel in kernels
 if occursin(r"cache::Rosenbrock(23|32)ConstantCache",string(kernel))
  eval(corrected_time(kernel))
 end
end
corrected_records=Dict()
for name in ("Rosenbrock23","Rosenbrock32")
 tab=getfield(Main,Symbol(name*"Tableau"))(Float64)
 cache=getfield(Main,Symbol(name*"ConstantCache"))(tab.c₃₂,tab.d,nothing,nothing)
 rows=[]
 for forced in (false,true),h in (.2,.05)
  y=[1.,.4];f=Force(I,forced);opts=(adaptive=true,abstol=1.,reltol=0.,internalnorm=norm,calck=false)
  i=Oracle(.3,h,copy(y),copy(y),f,nothing,f(y,nothing,.3),zeros(2),opts,Stats(0,0),Any[zeros(2),zeros(2)],0.)
  last_error[]=nothing;perform_step!(i,cache,false)
  push!(rows,Dict("forced"=>forced,"t"=>i.t,"h"=>h,"initial"=>y,"output"=>i.u,"error"=>last_error[]))
 end
 corrected_records[name]=rows
end
corrected_time_force=time_force_records()
@assert maximum(abs.(source_time_force["Rosenbrock32"][1]["output"]-source_time_force["Rosenbrock32"][1]["exact"]))>1e-3
for rows in values(corrected_time_force),row in rows
 @assert maximum(abs.(row["output"]-row["exact"]))<1e-15
 @assert maximum(abs.(row["defect"]))<1e-15
end
json(stdout,Dict("monorepo_tree"=>tree,"package_version"=>"2.7.0","tableaus"=>tables,"records"=>records,
 "tableau_package_version"=>"2.4.2","corrected_time_records"=>corrected_records,
 "time_force_source"=>source_time_force,"time_force_corrected"=>corrected_time_force,
 "time_transformation"=>"Rosenbrock23/32: replace only third-stage dt*f_t by dt*gamma*f_t. Exact autonomous augmentation t'=1 gives this term; original source loses nonautonomous main/estimate order. Original and corrected source maps are recorded separately.",
 "source_sha256"=>Dict(f=>bytes2hex(sha256(sources[f])) for f in files),
 "qualification"=>"Actual pinned constant-cache step bodies and coefficient constructors. Exact analytic J/f_t, identity mass, direct dense stage solves, ordinary broadcast in place of source macro annotations. Source FSAL initialization is supplied, adaptive residual normalization stubbed to expose the raw defect; no source controller/cache or mass-matrix equivalence."));println()
