using SHA,Pkg
root=ARGS[1];tree=bytes2hex(Pkg.GitTools.tree_hash(root))
@assert tree=="bae9b8e5f1d00e51c554648e5094e215164a19f8"
helper=read(joinpath(@__DIR__,"../wave9/higher_stages.jl"),String)
include_string(Main,"function json(io,value)"*split(split(helper,"function json(io,value)";limit=2)[2],"\nschemes=Dict()";limit=2)[1])
struct QNDF{M} end
struct QNDFCache end
mutable struct QNDFConstantCache
 order::Int;nconsteps::Int;consfailcnt::Int;EEst1::Float64;EEst2::Float64
end
mutable struct Integrator
 dt::Float64;est::Float64;is_disco_step::Bool;cache::QNDFConstantCache
end
module OrdinaryDiffEqCore
get_EEst(i)=i.est
end
get_gamma(i)=1.2
get_current_qmax(i,q)=q
get_qmax(i)=5.
get_qsteady_min(i)=.9
get_qsteady_max(i)=1.2
file="lib/OrdinaryDiffEqBDF/src/controllers.jl";source=read(joinpath(root,file),String)
start="function step_accept_controller!(integrator, cache::Union{QNDFCache, QNDFConstantCache}, alg::QNDF{max_order}, q) where {max_order}"
body=start*split(split(source,start;limit=2)[2],"\n\"\"\"";limit=2)[1]
include_string(Main,body,file)
records=[]
for p in 1:5, error in (.001,.1,.9), lower in (.00001,.3,2.), upper in (.00001,.3,2.)
 c=QNDFConstantCache(p,100,0,lower,upper);i=Integrator(.1,error,false,c)
 h=step_accept_controller!(i,c,QNDF{5}(),nothing)
 push!(records,Dict("order"=>p,"error"=>error,"lower_error"=>lower,"upper_error"=>upper,"next_order"=>c.order,"next_h_ratio"=>h/.1))
end
json(stdout,Dict("monorepo_tree"=>tree,"source_sha256"=>Dict(file=>bytes2hex(sha256(source))),"records"=>records,
 "qualification"=>"Unchanged QNDF accepted controller with mature nconsteps, default safety/deadband constants, nonzero normalized errors and no discontinuity; package policy differs in maturity/hold/deadband/retries"));println()
