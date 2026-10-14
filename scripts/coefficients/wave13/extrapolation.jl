using LinearAlgebra,SHA,Pkg
root=ARGS[1];tree=bytes2hex(Pkg.GitTools.tree_hash(root))
@assert tree=="bae9b8e5f1d00e51c554648e5094e215164a19f8"
helper=read(joinpath(@__DIR__,"../wave9/higher_stages.jl"),String)
include_string(Main,"function json(io,value)"*split(split(helper,"function json(io,value)";limit=2)[2],"\nschemes=Dict()";limit=2)[1])
macro muladd(ex);esc(ex);end
macro threaded(args...);esc(last(args));end
isthreaded(value)=value
struct ExtrapolationMidpointDeuflhard
 min_order::Int;init_order::Int;max_order::Int;sequence::Symbol;threading::Bool;sequence_factor::Int
end
struct ExtrapolationMidpointHairerWanner end
struct ImplicitDeuflhardExtrapolation end
struct ImplicitHairerWannerExtrapolation end
mutable struct AitkenNevilleConstantCache
 dtpropose::Float64;T::Matrix{Vector{Float64}};cur_order::Int;work::Any;A::Any;step_no::Int
end
mutable struct ExtrapolationMidpointDeuflhardConstantCache
 n_curr::Int;coefficients::Any;stage_number::Any;Q::Vector{Float64}
end
mutable struct Stats
 nf::Int
end
module OrdinaryDiffEqCore
increment_nf!(stats,n)=(stats.nf+=n)
end
mutable struct Integrator
 t::Float64;dt::Float64;uprev::Vector{Float64};u::Vector{Float64};f::Function;p::Any
 fsalfirst::Vector{Float64};fsallast::Vector{Float64};k::Vector{Any};stats::Stats;opts::Any;alg::Any;columns::Any
end
unwrap_alg(i,args...)=i.alg
step_file="lib/OrdinaryDiffEqExtrapolation/src/extrapolation_perform_step.jl"
coefficient_file="lib/OrdinaryDiffEqExtrapolation/src/extrapolation_caches.jl"
source=read(joinpath(root,step_file),String);coefficients=read(joinpath(root,coefficient_file),String)
block="struct extrapolation_coefficients"*split(coefficients,"struct extrapolation_coefficients";limit=2)[2]
include_string(Main,split(block,"\nfunction create_extrapolation_coefficients(";limit=3)[1]*"\nfunction create_extrapolation_coefficients("*split(block,"\nfunction create_extrapolation_coefficients(";limit=3)[2],coefficient_file)
start="function perform_step!(integrator, cache::AitkenNevilleConstantCache, repeat_step = false)"
body=start*split(split(source,start;limit=2)[2],"\nfunction initialize!";limit=2)[1]
unbroadcast(s)=replace(replace(s,r"@\.\.\s+broadcast\s*=\s*false\s+"=>""),r"@\.\.\s+"=>"")
include_string(Main,unbroadcast(body),step_file)
start="function perform_step!(\n        integrator, cache::ExtrapolationMidpointDeuflhardConstantCache,\n        repeat_step = false\n    )"
body=start*split(split(source,start;limit=2)[2],"\nfunction initialize!";limit=2)[1]
body=replace(body,"# Save the latest approximation and update FSAL\n    integrator.u = u"=>"# Capture local columns for independent comparison weights\n    integrator.columns = copy(T)\n    integrator.u = u")
include_string(Main,body,step_file)
records=Dict()
for base in ("euler","midpoint"), order in (base=="euler" ? (2:8) : (2:2:10)), forced in (false,true)
 calls=Any[];t=.3;h=.2;initial=[.7,-.4]
 f=(y,p,time)->begin
  value=[-2*y[1]+.1*y[2]^2+(forced ? .2*sin(time) : 0),-.5*y[2]+.2*y[1]^2+(forced ? .1*cos(time) : 0)]
  push!(calls,Dict("t"=>time,"y"=>copy(y),"f"=>copy(value)));value
 end
 first=f(initial,nothing,t)
 if base=="euler"
  alg=(threading=false,);cache=AitkenNevilleConstantCache(h,fill(zeros(2),order,order),order,nothing,nothing,1)
 else
  levels=order÷2;alg=ExtrapolationMidpointDeuflhard(0,levels-1,levels-1,:harmonic,false,2)
  tab=levels>1 ? create_extrapolation_coefficients(Float64,alg) : extrapolation_coefficients([1],ones(1,1),ones(1),zeros(0,0),Float64[])
  cache=ExtrapolationMidpointDeuflhardConstantCache(levels-1,tab,nothing,zeros(levels))
 end
 i=Integrator(t,h,copy(initial),copy(initial),f,nothing,first,first,Any[first,first],Stats(1),
  (adaptive=false,reltol=1e-6,internalnorm=(x,t)->norm(x)),alg,nothing)
 perform_step!(i,cache,false)
 if base=="euler"
  comparison=cache.T[order,order-1];main_weights=Float64[];comparison_weights=Float64[]
 else
  l=order÷2;tab=cache.coefficients
  main_weights=tab.extrapolation_scalars[l]*tab.extrapolation_weights[1:l,l]
  comparison_weights=l>1 ? [0.;tab.extrapolation_scalars_2[l-1]*tab.extrapolation_weights_2[1:l-1,l-1]] : [0.]
  comparison=l>1 ? sum(comparison_weights[j]*i.columns[j] for j in 1:l) : initial+h*first
 end
 records[base*string(order)*(forced ? "_forced" : "_autonomous")]=Dict("base"=>base,"order"=>order,"t"=>t,"h"=>h,
  "initial"=>initial,"state"=>i.u,"error"=>i.u-comparison,"main_column_weights"=>main_weights,
  "comparison_column_weights"=>comparison_weights,"calls"=>calls)
end
json(stdout,Dict("monorepo_tree"=>tree,"records"=>records,
 "source_sha256"=>Dict(step_file=>bytes2hex(sha256(source)),coefficient_file=>bytes2hex(sha256(coefficients))),
 "qualification"=>"Fixed-order serial AitkenNeville and Deuflhard midpoint constant-cache bodies; broadcast/muladd/thread annotations removed, rational generic coefficient constructor, local midpoint columns captured for independent comparison weights. Controllers, order selection, FSAL/work shortcuts, parallel columns and dense output not reproduced."));println()
