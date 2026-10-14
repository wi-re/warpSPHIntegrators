# Execute pinned constant-cache step bodies using an independent dense Newton
# increment solve and minimal integrator/cache scaffolding. No SciML runtime.
using LinearAlgebra, SHA, Pkg
root=ARGS[1];tree=bytes2hex(Pkg.GitTools.tree_hash(root))
@assert tree=="bae9b8e5f1d00e51c554648e5094e215164a19f8"
json_source=read(joinpath(@__DIR__,"../wave9/higher_stages.jl"),String)
include_string(Main,"function json(io,value)"*split(split(json_source,"function json(io,value)";limit=2)[2],"\nschemes=Dict()";limit=2)[1])
macro muladd(ex);esc(ex);end
module OrdinaryDiffEqCore
increment_nf!(stats,n)=(stats.nf+=n)
end
mutable struct DenseNewton
    tmp::Vector{Float64};z::Vector{Float64};c::Float64
end
mutable struct MEBDF2ConstantCache
    nlsolver::DenseNewton
end
mutable struct CNLF2ConstantCache
    k2::Vector{Float64};uprev2::Vector{Float64};nlsolver::DenseNewton
end
struct SplitForce
    f1::Function;f2::Function
end
(f::SplitForce)(u,p,t)=f.f1(u,p,t)+f.f2(u,p,t)
mutable struct Stats
    nf::Int;nf2::Int
end
mutable struct Integrator
    t::Float64;dt::Float64;uprev::Vector{Float64};u::Vector{Float64}
    f::Any;p::Any;fsalfirst::Vector{Float64};fsallast::Vector{Float64}
    k::Vector{Vector{Float64}};alg::Any;startup::Bool;stats::Stats
    jacobian::Function;solve_times::Vector{Float64}
end
unwrap_alg(i,flag)=i.alg
markfirststage!(nlsolver)=nothing
isnewton(nlsolver)=false
set_new_W!(nlsolver,flag)=nothing
nlsolvefail(nlsolver)=false
is_startup_step(i)=i.startup
function nlsolve!(nlsolver,i,cache,repeat_step)
    f=i.f isa SplitForce ? i.f.f1 : i.f
    time=i.t+nlsolver.c*i.dt;push!(i.solve_times,time)
    z=copy(nlsolver.z)
    for _ in 1:15
        y=nlsolver.tmp+z
        residual=z-i.dt*f(y,i.p,time)
        maximum(abs,residual)<1e-14 && break
        matrix=Matrix{Float64}(I,2,2)-i.dt*i.jacobian(y,time)
        z-=matrix\residual
    end
    @assert maximum(abs,z-i.dt*f(nlsolver.tmp+z,i.p,time))<1e-13
    nlsolver.z=z;return z
end
files=Dict("bdf"=>"lib/OrdinaryDiffEqBDF/src/bdf_perform_step.jl",
    "imex"=>"lib/OrdinaryDiffEqIMEXMultistep/src/imex_multistep_perform_step.jl")
hashes=Dict()
for (key,start,stop) in (("bdf","@muladd function perform_step!(integrator, cache::MEBDF2ConstantCache",
                         "\nfunction initialize!(integrator, cache::MEBDF2Cache)"),
                        ("imex","function perform_step!(integrator, cache::CNLF2ConstantCache",
                         "\nfunction initialize!(integrator, cache::CNLF2Cache)"))
    file=joinpath(root,files[key]);source=read(file,String)
    hashes[files[key]]=bytes2hex(sha256(source))
    body=start*split(split(source,start;limit=2)[2],stop;limit=2)[1]
    include_string(Main,body,file)
end
records=Dict()
for nonaut in (false,true)
    f1=(y,p,t)->[-2*y[1]+.1*y[2]^2+(nonaut ? .2*sin(t) : 0),
        -.5*y[2]+.2*y[1]^2+(nonaut ? .1*cos(t) : 0)]
    f2=(y,p,t)->[.3*y[2]+(nonaut ? .05*sin(t) : 0),-.2*y[1]^2+(nonaut ? .03*cos(t) : 0)]
    jac=(y,t)->[-2. .2*y[2];.4*y[1] -.5]
    for method in ("mebdf2","cnlf2_startup","cnlf2_warm")
        t=.3;h=.1;y=[.7,-.4];old=[.73,-.38]
        f=method=="mebdf2" ? f1 : SplitForce(f1,f2)
        solver=DenseNewton(zeros(2),zeros(2),1.)
        cache=method=="mebdf2" ? MEBDF2ConstantCache(solver) : CNLF2ConstantCache(f1(old,nothing,t-h),copy(old),solver)
        i=Integrator(t,h,copy(y),copy(y),f,nothing,f(y,nothing,t),zeros(2),[zeros(2),zeros(2)],
            (extrapolant=:constant,),method=="cnlf2_startup",Stats(0,0),jac,Float64[])
        perform_step!(i,cache,false)
        records[method*(nonaut ? "_forced" : "_autonomous")]=Dict("state"=>i.u,"solve_times"=>i.solve_times,
            "initial"=>y,"previous"=>old,"t"=>t,"h"=>h)
    end
end
json(stdout,Dict("monorepo_tree"=>tree,"source_sha256"=>hashes,
    "qualification"=>"Unchanged constant-cache step bodies with independent dense Newton increment solves and minimal cache/integrator mocks; no upstream nonlinear solver/controller equivalence claim",
    "records"=>records));println()
