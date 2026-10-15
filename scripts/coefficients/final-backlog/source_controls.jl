# Pinned pure Nordsieck kernels and actual IRKC constant-cache step body.
using LinearAlgebra,SHA,Pkg
BLAS.set_num_threads(1)
root=ARGS[1];@assert bytes2hex(Pkg.GitTools.tree_hash(root))=="bae9b8e5f1d00e51c554648e5094e215164a19f8"
helper=read(joinpath(@__DIR__,"../wave9/higher_stages.jl"),String)
include_string(Main,"function json(io,value)"*split(split(helper,"function json(io,value)";limit=2)[2],"\nschemes=Dict()";limit=2)[1])
macro muladd(expr);esc(expr);end
macro var".."(args...);esc(last(args));end
abstract type OrdinaryDiffEqConstantCache end
struct JVODECache end
struct JVODEConstantCache end
function name_of(sig)
 sig isa Symbol && return sig
 sig isa Expr && sig.head in (:where,:(::),:curly) && return name_of(sig.args[1])
 sig isa Expr && sig.head==:call && return name_of(sig.args[1])
 nothing
end
path="lib/OrdinaryDiffEqNordsieck/src/nordsieck_utils.jl";text=read(joinpath(root,path),String)
for expr in Meta.parseall(text).args
 if expr isa Expr && expr.head==:function && name_of(expr.args[1]) in (:∫₋₁⁰dx,:calc_coeff!,:perform_predict!,:nordsieck_rescale!,:is_nordsieck_change_order)
  eval(expr)
 end
end
mutable struct NCache <: OrdinaryDiffEqConstantCache
 order;z;l;m;dts;c_LTE;c_LTE₋₁;c_LTE₊₁;c_conv;c_𝒟
end
records=[]
for p in 1:12
 jets=[[1/(j+1)] for j in 0:p]
 cache=NCache(p,deepcopy(jets),zeros(p+1),zeros(p+1),ones(p+2),0.,0.,0.,0.,0.)
 calc_coeff!(cache);perform_predict!(cache);pred=deepcopy(cache.z);perform_predict!(cache,true)
 @assert maximum(abs.(reduce(vcat,cache.z)-reduce(vcat,jets)))<1e-9
 cache.z=deepcopy(jets);cache.dts[1]=.5;nordsieck_rescale!(cache)
 push!(records,Dict("order"=>p,"l"=>cache.l,"jets"=>jets,"predicted"=>pred,"rescaled"=>cache.z))
end
mutable struct NL
 tmp;z;γ;c;failed
end
mutable struct ICache
 minm;du₁;du₂;nlsolver
end
mutable struct Stats
 nf;nf2
end
mutable struct Oracle
 t;dt;uprev;u;f;p;fsalfirst;fsallast;opts;stats;k;eigen_est;force_stepfail
end
unwrap_alg(i,b)=(eigen_est=i->nothing,)
increment_nf!(s,n)=(s.nf+=n)
markfirststage!(s)=nothing
isnewton(s)=false
function nlsolve!(s,i,cache,repeated)
 z=copy(s.z);I=Matrix{Float64}(LinearAlgebra.I,length(z),length(z))
 for iteration in 1:15
  q=s.tmp+s.γ*z;res=z-i.dt*i.f.f1(q,nothing,i.t+s.c*i.dt)
  norm(res,Inf)<1e-13 && return z
  eps=1e-6
  J=hcat([(i.f.f1(q+eps*I[:,j],nothing,i.t+s.c*i.dt)-i.f.f1(q-eps*I[:,j],nothing,i.t+s.c*i.dt))/(2eps) for j in eachindex(q)]...)
  z-=(I-i.dt*s.γ*J)\res
 end
 s.failed=true;return z
end
irkpath="lib/OrdinaryDiffEqStabilizedIRK/src/irkc_perform_step.jl";irk=read(joinpath(root,irkpath),String)
for expr in Meta.parseall(irk).args
 if expr isa Expr && expr.head==:function && name_of(expr.args[1])==:perform_step! && occursin("IRKCConstantCache",string(expr.args[1]))
  # Dispatch-only rename to the minimal fixture cache; body is unmodified.
  eval(Meta.parse(replace(string(expr),"IRKCConstantCache"=>"ICache")))
 end
end
irkruns=[];irksteps=[]
for stages in (2,4,10)
 runs=[]
 for n in (8,16,32)
  q=[1.,.4];h=1/n
  f1=(q,p,t)->-2q-.1q.^2;f2=(q,p,t)->q+.1*[1.,.4].^2*exp(-2t)
  cache=ICache(stages,f1(q,nothing,0.),f2(q,nothing,0.),NL(q,zero(q),0.,0.,false))
  i=Oracle(0.,h,q,q,(f1=f1,f2=f2),nothing,f1(q,nothing,0.)+f2(q,nothing,0.),zero(q),(adaptive=false,),Stats(0,0),Any[zero(q),zero(q)],0.,false)
  for j in 1:n
   i.t=(j-1)*h;i.uprev=copy(i.u);perform_step!(i,cache,false)
   if j==1;push!(irksteps,Dict("stages"=>stages,"h"=>h,"output"=>i.u));end
  end
  push!(runs,Dict("n"=>n,"error"=>norm(i.u-[1.,.4]*exp(-1),Inf)))
 end
 push!(irkruns,Dict("stages"=>stages,"runs"=>runs,"order"=>log2(runs[end-1]["error"]/runs[end]["error"])))
end
json(stdout,Dict("nordsieck"=>records,"irkc_steps"=>irksteps,"irkc_refinements"=>irkruns,
 "source_sha256"=>Dict(path=>bytes2hex(sha256(text)),irkpath=>bytes2hex(sha256(irk))),
 "qualification"=>"Actual Nordsieck coefficient/Pascal/rescale functions; actual IRKC constant-cache step body at supplied degrees with independent dense Newton replacement for source nonlinear solver and no adaptive estimator. Minimal dispatch fixtures, not full SciML startup/controllers."));println()
