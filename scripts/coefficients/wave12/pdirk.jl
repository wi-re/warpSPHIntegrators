# Pinned parallel DIRK coefficients and actual serial/threaded-branch step body.
# Thread loops execute serially here; nonlinear stages use a tiny analytic-Jacobian
# Newton oracle. This validates numerical maps, not concurrent Julia scheduling.
using LinearAlgebra,SHA,Pkg
root=ARGS[1];tree=bytes2hex(Pkg.GitTools.tree_hash(root))
@assert tree=="bae9b8e5f1d00e51c554648e5094e215164a19f8"
helper=read(joinpath(@__DIR__,"../wave9/higher_stages.jl"),String)
include_string(Main,"function json(io,value)"*split(split(helper,"function json(io,value)";limit=2)[2],"\nschemes=Dict()";limit=2)[1])
macro muladd(ex);esc(ex);end
macro threaded(policy,ex);esc(ex);end
abstract type OrdinaryDiffEqConstantCache end
mutable struct NonlinearStage
 z::Vector{Float64};tmp::Vector{Float64};γ::Float64;c::Float64
end
markfirststage!(n)=nothing
nlsolvefail(n)=false
struct PDIRK44ConstantCache{N,T} <: OrdinaryDiffEqConstantCache
 nlsolver::N;tab::T
end
mutable struct Oracle
 t::Float64;dt::Float64;uprev::Vector{Float64};u::Vector{Float64};threading::Bool;forced::Bool
end
unwrap_alg(i,b)=i
isthreaded(b)=b
function rhs(y,t,forced)
 [-2*y[1]+.1*y[1]*y[2]+(forced ? .2*sin(t) : 0),
  -3*y[2]+.2*y[1]^2+(forced ? .1*cos(t) : 0)]
end
function nlsolve!(stage,i,cache,repeat)
 z=copy(stage.z);t=i.t+i.dt*stage.c
 for _ in 1:20
  y=stage.tmp+stage.γ*z
  residual=z-i.dt*rhs(y,t,i.forced)
  norm(residual,Inf)<1e-15 && return z
  jac=[-2+.1*y[2] .1*y[1];.4*y[1] -3]
  z-=(I-i.dt*stage.γ*jac)\residual
 end
 error("Oracle Newton failed")
end
base="lib/OrdinaryDiffEqPDIRK/"
files=[base*f for f in ("src/pdirk_caches.jl","src/pdirk_perform_step.jl","src/algorithms.jl","src/alg_utils.jl","Project.toml")]
sources=Dict(f=>read(joinpath(root,f),String) for f in files)
cachetext=sources[files[1]]
include_string(Main,"struct PDIRK44Tableau"*split(split(cachetext,"struct PDIRK44Tableau";limit=2)[2],"\nfunction alg_cache";limit=2)[1])
steptext=sources[files[2]]
include_string(Main,split(steptext,"\nfunction initialize!(integrator, cache::PDIRK44Cache)";limit=2)[1])
tab=PDIRK44Tableau(Float64,Float64)
A=[tab.γs[1] 0 0 0;0 tab.γs[2] 0 0;tab.α1[1] tab.α2[1] tab.γs[1] 0;tab.α1[2] tab.α2[2] 0 tab.γs[2]]
b=[tab.b1,tab.b3,tab.b2,tab.b4];c=collect(tab.cs)
@assert maximum(abs.(sum(A,dims=2)[:]-c))<1e-15
rows=[]
for forced in (false,true),threaded in (false,true),h in (.2,.05)
 y=[1.,.4];n=[NonlinearStage(zeros(2),copy(y),1.,0.) for _ in 1:2]
 i=Oracle(.3,h,copy(y),copy(y),threaded,forced);cache=PDIRK44ConstantCache(n,tab)
 perform_step!(i,cache,false)
 push!(rows,Dict("forced"=>forced,"threaded_branch"=>threaded,"t"=>i.t,"h"=>h,"initial"=>y,"output"=>i.u))
end
json(stdout,Dict("monorepo_tree"=>tree,"package_version"=>"2.4.1","A"=>[collect(row) for row in eachrow(A)],"b"=>b,"c"=>c,
 "stage_groups"=>[[1,2],[3,4]],"records"=>rows,
 "source_sha256"=>Dict(f=>bytes2hex(sha256(sources[f])) for f in files),
 "qualification"=>"Actual constant-cache step body and coefficient constructor; analytic-Jacobian nonlinear stage oracle. Both source branches executed; thread loops are serial in this harness. PDIRK means parallel DIRK on ordinary first-order ODEs, not partitioned second-order mechanics."));println()
