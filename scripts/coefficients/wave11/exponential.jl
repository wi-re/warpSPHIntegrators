# Actual pinned ExpRK kernels with exact dense phi actions in place of Krylov.
using LinearAlgebra,SHA,Pkg
BLAS.set_num_threads(1)
macro muladd(expr);esc(expr);end
root=ARGS[1];tree=bytes2hex(Pkg.GitTools.tree_hash(root))
@assert tree=="bae9b8e5f1d00e51c554648e5094e215164a19f8"
helper=read(joinpath(@__DIR__,"../wave9/higher_stages.jl"),String)
include_string(Main,"function json(io,value)"*split(split(helper,"function json(io,value)";limit=2)[2],"\nschemes=Dict()";limit=2)[1])
json(io::IO,value::Tuple)=json(io,collect(value))
json(io::IO,::Nothing)=print(io,"null")
const SEMILINEAR=("LawsonEuler","NorsettEuler","ETDRK2","ETDRK3","ETDRK4","HochOst4")
const LINEARIZED=("Exp4","EPIRK4s3A","EPIRK4s3B","EPIRK5s3","EPIRK5P1","EPIRK5P2","EXPRB53s3","Exprb32","Exprb43")
const METHODS=(SEMILINEAR...,LINEARIZED...)
abstract type ExponentialAlgorithm end
for name in METHODS
 @eval struct $(Symbol(name)) <: ExponentialAlgorithm
  krylov::Bool;m::Int;iop::Int;adaptive_krylov::Bool
 end
 @eval struct $(Symbol(name*"ConstantCache"))
  ops::Any
 end
end
mutable struct Stats
 nf::Int;nf2::Int
end
module OrdinaryDiffEqCore
increment_nf!(s,n)=(s.nf+=n)
set_EEst!(i,value)=(i.EEst=value)
end
const last_error=Ref{Any}(nothing)
function calculate_residuals(error,args...)
 last_error[]=copy(error);return error
end
struct LinearPart
 f::Any
end
struct SplitFunction
 f1::LinearPart;f2::Function
end
(f::SplitFunction)(u,p,t)=f.f1.f*u+f.f2(u,p,t)
const L=[-2. .3;.1 -3.]
const AMPLITUDE=[1.,.4]
function nonlinear(q,t,problem)
 value=[.1*q[1]*q[2],.2*q[1]^2]
 if problem=="forced"
  value+=[.2*sin(t),.1*cos(t)]
 elseif problem=="manufactured"
  target=AMPLITUDE*exp(-t)
  value+=-target-L*target-[.1*target[1]*target[2],.2*target[1]^2]
 end
 return value
end
struct FullForce
 problem::String;augmented::Bool;calls::Vector{Any}
end
function (f::FullForce)(u,p,t)
 q=f.augmented ? u[2:end] : u;clock=f.augmented ? u[1] : t
 value=L*q+nonlinear(q,clock,f.problem)
 push!(f.calls,Dict("t"=>t,"physical_time"=>clock,"state"=>copy(u),"rate"=>f.augmented ? [1.;value] : copy(value)))
 return f.augmented ? [1.;value] : value
end
mutable struct Oracle
 t::Float64;dt::Float64;uprev::Vector{Float64};u::Vector{Float64};f::Any;p::Nothing
 fsalfirst::Vector{Float64};fsallast::Vector{Float64};opts::Any;stats::Stats;k::Vector{Any};EEst::Float64
 alg::Any;problem::String;augmented::Bool
end
unwrap_alg(i,b)=i.alg
function calc_J(i,cache)
 q=i.augmented ? i.uprev[2:end] : i.uprev;t=i.augmented ? i.uprev[1] : i.t
 J=L+[.1*q[2] .1*q[1];.4*q[1] 0.]
 if !i.augmented;return J;end
 target=AMPLITUDE*exp(-t)
 ft=i.problem=="autonomous" ? zeros(2) : i.problem=="forced" ? [.2*cos(t),-.1*sin(t)] :
  target+L*target+2*[.1*target[1]*target[2],.2*target[1]^2]
 return [zeros(1,3);ft J]
end
function phi(A,k)
 n=size(A,1);block=zeros(eltype(A),n*(k+1),n*(k+1));block[1:n,1:n]=A
 for j in 1:k;block[(j-1)*n+1:j*n,j*n+1:(j+1)*n]=Matrix{eltype(A)}(I,n,n);end
 E=exp(block)
 return [E[1:n,j*n+1:(j+1)*n] for j in 0:k]
end
struct DenseBasis
 A::Matrix{Float64};v::Vector{Float64}
end
arnoldi(A,v;kwargs...)=DenseBasis(A,v)
phiv(t,K::DenseBasis,k;kwargs...)=phiv(t,K.A,K.v,k;kwargs...)
phiv(t,A,v,k;kwargs...)=hcat([P*v for P in phi(t*A,k)]...)
expv(t,A,v;kwargs...)=exp(t*A)*v
function phiv_timestep(t::Real,A,B;kwargs...)
 P=phi(t*A,size(B,2)-1)
 return sum(t^(j-1)*P[j]*B[:,j] for j in axes(B,2))
end
phiv_timestep(ts::AbstractVector,A,B;kwargs...)=hcat([phiv_timestep(t,A,B;kwargs...) for t in ts]...)
base="lib/OrdinaryDiffEqExponentialRK/"
files=[base*f for f in ("src/exponential_rk_perform_step.jl","src/exponential_rk_caches.jl","src/algorithms.jl","src/alg_utils.jl","Project.toml")]
sources=Dict(f=>read(joinpath(root,f),String) for f in files)
function name_of(sig)
 sig isa Symbol && return sig
 sig isa Expr && sig.head in (:where,:(::),:curly) && return name_of(sig.args[1])
 sig isa Expr && sig.head==:call && return name_of(sig.args[1])
 return nothing
end
for original in Meta.parseall(sources[files[2]]).args
 expr=original
 expr isa Expr && expr.head==:macrocall && (expr=expr.args[end])
 if expr isa Expr && expr.head in (:function,:(=)) && name_of(expr.args[1])==:expRK_operators
  eval(original)
 end
end
kernels=Dict()
for original in Meta.parseall(sources[files[1]]).args
 expr=original
 expr isa Expr && expr.head==:macrocall && (expr=expr.args[end])
 if expr isa Expr && expr.head in (:function,:(=)) && name_of(expr.args[1])==:_compute_nl
  eval(original)
 elseif expr isa Expr && expr.head==:function && name_of(expr.args[1])==:perform_step!
  signature=string(expr.args[1])
  for name in METHODS
   if occursin("cache::"*name*"ConstantCache",signature)
    eval(original);kernels[name]=original
   end
  end
 end
end
function setup(name,h,problem;krylov=true,augmented=false,t=.3)
 calls=[];q=copy(AMPLITUDE);y=augmented ? [t;q] : q
 if name in SEMILINEAR
  force=SplitFunction(LinearPart(L),(u,p,clock)->begin
   result=nonlinear(u,clock,problem);push!(calls,Dict("t"=>clock,"physical_time"=>clock,"state"=>copy(u),"rate"=>L*u+result));result
  end)
 else
  force=FullForce(problem,augmented,calls)
 end
 alg=getfield(Main,Symbol(name))(krylov,length(y),0,false)
 operators=name in SEMILINEAR && !krylov ? expRK_operators(alg,h,L) : nothing
 cache=getfield(Main,Symbol(name*"ConstantCache"))(operators)
 opts=(adaptive=true,abstol=1.,reltol=1e-14,internalnorm=(x,t=0.)->norm(x),internalopnorm=opnorm)
 i=Oracle(t,h,copy(y),copy(y),force,nothing,force(y,nothing,t),zero(y),opts,Stats(0,0),Any[zero(y),zero(y)],0.,alg,problem,augmented)
 return i,cache,calls
end
records=[]
for name in METHODS,problem in ("autonomous","forced"),h in (.2,.05)
 for krylov in (name in SEMILINEAR ? (true,false) : (true,)),augmented in (name in LINEARIZED ? (false,true) : (false,))
  i,cache,calls=setup(name,h,problem;krylov,augmented)
  last_error[]=nothing;perform_step!(i,cache,false)
  push!(records,Dict("method"=>name,"problem"=>problem,"h"=>h,"t"=>i.t,"initial"=>i.uprev,
   "output"=>i.u,"error"=>last_error[],"krylov_branch"=>krylov,"augmented_time"=>augmented,"calls"=>calls))
 end
end
refinements=[]
for name in METHODS,krylov in (name in SEMILINEAR ? (true,false) : (true,)),augmented in (name in LINEARIZED ? (false,true) : (false,))
 runs=[]
 for n in (4,8,16,32)
  h=1/n;i,cache,calls=setup(name,h,"manufactured";krylov,augmented,t=0.)
  for step in 1:n
   i.t=(step-1)*h;i.uprev=copy(i.u)
   if step>1;i.fsalfirst=copy(i.fsallast);end
   perform_step!(i,cache,false)
  end
  q=augmented ? i.u[2:end] : i.u
  push!(runs,Dict("steps"=>n,"error"=>norm(q-AMPLITUDE*exp(-1),Inf),"output"=>i.u))
 end
 usable=[j for j in 2:length(runs) if min(runs[j-1]["error"],runs[j]["error"])>1e-12]
 order=isempty(usable) ? nothing : log2(runs[usable[end]-1]["error"]/runs[usable[end]]["error"])
 push!(refinements,Dict("method"=>name,"krylov_branch"=>krylov,"augmented_time"=>augmented,"runs"=>runs,"order"=>order))
end
json(stdout,Dict("monorepo_tree"=>tree,"package_version"=>match(r"version = \"([^\"]+)\"",sources[files[end]]).captures[1],
 "linear_matrix"=>[collect(row) for row in eachrow(L)],"records"=>records,"refinements"=>refinements,
 "source_sha256"=>Dict(f=>bytes2hex(sha256(sources[f])) for f in files),
 "qualification"=>"Actual constant-cache kernels and cached operator constructors; minimal algorithm/cache dispatch scaffolding. Exact dense augmented-block exponentials replace Krylov/phi implementations. Exact analytic frozen Jacobians; supplied FSAL and captured raw defects. Both operator branches for semilinear methods; plain and autonomous time-augmented controls for linearized methods. No source Krylov error/controller/cache or full SciML runtime equivalence."));println()
