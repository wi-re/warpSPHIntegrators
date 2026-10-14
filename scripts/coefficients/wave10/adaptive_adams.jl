# Pinned mature VCABM defect maps and the actual accepted-order branch.
using SHA,Pkg
root=ARGS[1];tree=bytes2hex(Pkg.GitTools.tree_hash(root))
@assert tree=="bae9b8e5f1d00e51c554648e5094e215164a19f8"
helper=read(joinpath(@__DIR__,"../wave9/higher_stages.jl"),String)
include_string(Main,"function json(io,value)"*split(split(helper,"function json(io,value)";limit=2)[2],"\nschemes=Dict()";limit=2)[1])
abstract type OrdinaryDiffEqMutableCache end
struct VCABMCache end
mutable struct VCABMConstantCache
 dts::Vector{Float64};c::Matrix{Float64};g::Vector{Float64}
 ϕ_n::Vector{Vector{Float64}};ϕstar_n::Vector{Vector{Float64}};ϕstar_nm1::Vector{Vector{Float64}}
 β::Vector{Float64};ϕ_np1::Vector{Vector{Float64}};ξ::Float64;ξ0::Float64;order::Int
end
file="lib/OrdinaryDiffEqAdamsBashforthMoulton/src/adams_utils.jl"
source=read(joinpath(root,file),String)
# Mutable broadcast branches remain unexecuted; the macro shim only removes
# their annotation. Constant-cache helper bodies execute unchanged.
macro var".."(args...);esc(last(args));end
include_string(Main,source,file)
records=[]
for p in 3:5, irregular in (false,true)
 times=irregular ? [-.9,-.7,-.46,-.31,-.1,0.][6-p:6] : collect(-p:0).*.1
 h=irregular ? .08 : .1
 n=p+3;cache=VCABMConstantCache(zeros(13),zeros(13,13),zeros(13),
  [zeros(n) for _ in 1:13],[zeros(n) for _ in 1:13],[zeros(n) for _ in 1:13],
  zeros(13),[zeros(n) for _ in 1:13],0.,0.,p)
 for i in 1:p+1
  step=i<p+1 ? times[i+1]-times[i] : h
  cache.dts=vcat(step,cache.dts[1:end-1]);du=zeros(n);du[i]=1.
  ϕ_and_ϕstar!(cache,du,min(i,p))
  if i<p+1;cache.ϕstar_nm1=copy(cache.ϕstar_n);end
 end
 g_coefs!(cache,p+1)
 du=zeros(n);du[p+2]=1.;ϕ_np1!(cache,du,p+1)
 main=(cache.g[p+1]-cache.g[p])*cache.ϕ_np1[p+1]/h
 lower1=(cache.g[p]-cache.g[p-1])*cache.ϕ_np1[p]/h
 lower2=(cache.g[p-1]-cache.g[p-2])*cache.ϕ_np1[p-1]/h
 expand_ϕ_and_ϕstar!(cache,p+1)
 du=zeros(n);du[p+3]=1.;ϕ_np1!(cache,du,p+2)
 upper=γstar[p+2]*cache.ϕ_np1[p+2]
 push!(records,Dict("order"=>p,"times"=>times,"h"=>h,"main"=>main,
  "lower1"=>lower1,"lower2"=>lower2,"upper"=>upper))
end
perform_file="lib/OrdinaryDiffEqAdamsBashforthMoulton/src/adams_bashforth_moulton_perform_step.jl"
perform=read(joinpath(root,perform_file),String)
vcabm=split(split(perform,"@muladd function perform_step!(integrator, cache::VCABMConstantCache";limit=2)[2],"function initialize!(integrator, cache::VCABMCache)";limit=2)[1]
branch="if max(errm2, errm1) <= OrdinaryDiffEqCore.get_EEst(integrator)"*
 split(split(vcabm,"if max(errm2, errm1) <= OrdinaryDiffEqCore.get_EEst(integrator)";limit=2)[2],"end # if";limit=2)[1]*"end"
module OrdinaryDiffEqCore
get_EEst(i)=i.est
set_EEst!(i,v)=(i.est=v)
end
mutable struct PolicyIntegrator
 est::Float64
end
include_string(Main,"function choose(cache,integrator,order,max_order,errm2,errm1,errp1)\n"*branch*"\nreturn cache.order\nend",perform_file)
choices=[]
for p in 3:5, error in (.001,.1,.9), lower2 in (.00001,.3,2.), lower1 in (.00001,.3,2.), upper in (.00001,.3,2.)
 # Actual branch mutates order; use the minimal mutable cache holder.
 cache=VCABMConstantCache([],zeros(0,0),[],[],[],[],[],[],0.,0.,p)
 i=PolicyIntegrator(error);selected=choose(cache,i,p,5,lower2,lower1,upper)
 push!(choices,Dict("order"=>p,"error"=>error,"lower_errors"=>[lower2,lower1],"upper_error"=>upper,"source_next_order"=>selected,"source_next_error"=>i.est))
end
json(stdout,Dict("monorepo_tree"=>tree,"package_version"=>"2.2.0",
 "source_sha256"=>Dict(file=>bytes2hex(sha256(source)),perform_file=>bytes2hex(sha256(perform))),
 "defects"=>records,"choices"=>choices,
 "qualification"=>"Unchanged constant-cache helper branches on basis histories and actual mature accepted-order branch. Source lower boundary permits order 2; package bounds orders to 3–5. Startup, maturity, FSAL, rejection and dt policies are separate."));println()
