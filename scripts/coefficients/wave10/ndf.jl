# Pinned quasi-constant NDF warm kernel and difference-transform oracle.
using LinearAlgebra,SHA,Pkg
root=ARGS[1];tree=bytes2hex(Pkg.GitTools.tree_hash(root))
@assert tree=="bae9b8e5f1d00e51c554648e5094e215164a19f8"
json_source=read(joinpath(@__DIR__,"../wave9/higher_stages.jl"),String)
include_string(Main,"function json(io,value)"*split(split(json_source,"function json(io,value)";limit=2)[2],"\nschemes=Dict()";limit=2)[1])
utils_file="lib/OrdinaryDiffEqBDF/src/bdf_utils.jl";step_file="lib/OrdinaryDiffEqBDF/src/bdf_perform_step.jl"
utils=read(joinpath(root,utils_file),String);steps=read(joinpath(root,step_file),String)
algorithms=read(joinpath(root,"lib/OrdinaryDiffEqBDF/src/algorithms.jl"),String)
defaults=split(algorithms,"function QNDF(";limit=2)[2]
kappa=eval(Meta.parse("("*match(r"kappa = \((.*?)\)"s,defaults).captures[1]*")"))
unbroadcast(s)=replace(replace(s,r"@\.\.\s+broadcast\s*=\s*false\s+"=>""),r"@\.\.\s+"=>"")
include_string(Main,split(utils,"\nfunction R!";limit=2)[1])
include_string(Main,"function calc_R!"*split(split(utils,"function calc_R!";limit=2)[2],"\nfunction calc_R(";limit=2)[1])
include_string(Main,unbroadcast("function update_D!"*split(split(utils,"function update_D!";limit=2)[2],"\nconst γₖ";limit=2)[1]))
include_string(Main,match(r"const γₖ = [^\n]+",utils).match)
module ArrayInterface
ismutable(x)=false
end
module OrdinaryDiffEqCore
increment_nf!(stats,n)=(stats.nf+=n)
get_EEst(i)=0.
end
const COEFFICIENT_MULTISTEP=:coefficient
mutable struct DenseNewton
    tmp::Vector{Float64};z::Vector{Float64};γ::Float64;α::Float64;method::Symbol
end
mutable struct QNDFConstantCache{max_order}
    dtprev::Float64;order::Int;D::Vector{Vector{Float64}};U::Matrix{Float64};R::Matrix{Float64};RU::Matrix{Float64}
    nlsolver::DenseNewton;γₖ::Any;prevD::Vector{Vector{Float64}};prevorder::Int;nconsteps::Int;consfailcnt::Int
end
struct Force
    mass_matrix::UniformScaling
    forced::Bool
end
(f::Force)(y,p,t)=[-2*y[1]+.1*y[2]^2+(f.forced ? .2*sin(t) : 0),
    -.5*y[2]+.2*y[1]^2+(f.forced ? .1*cos(t) : 0)]
mutable struct Stats
    nf::Int
end
mutable struct Integrator
    t::Float64;dt::Float64;uprev::Vector{Float64};u::Vector{Float64};f::Force;p::Any
    cache::Any;alg::Any;stats::Stats;opts::Any;derivative_discontinuity::Bool
    fsallast::Vector{Float64};k::Vector{Vector{Float64}}
end
unwrap_alg(i,flag)=i.alg
markfirststage!(solver)=nothing
nlsolvefail(solver)=false
function nlsolve!(solver,i,cache,repeat_step)
    y=copy(solver.z);h=solver.γ*i.dt
    for _ in 1:15
        residual=y-h*(solver.tmp+i.f(y,i.p,i.t+i.dt))
        maximum(abs,residual)<1e-14 && break
        jac=[-2. .2*y[2];.4*y[1] -.5]
        y-=(Matrix{Float64}(I,2,2)-h*jac)\residual
    end
    @assert maximum(abs,y-h*(solver.tmp+i.f(y,i.p,i.t+i.dt)))<1e-13
    return y
end
start="function perform_step!(\n        integrator, cache::QNDFConstantCache{max_order},"
stop="\nfunction initialize!(integrator, cache::QNDFCache"
body=start*split(split(steps,start;limit=2)[2],stop;limit=2)[1]
include_string(Main,unbroadcast(body),step_file)
records=Dict()
for order in 1:5, ratio in (.5,1.,1.5), forced in (false,true)
    old_h=.1;h=ratio*old_h;t=.3;y=[.7,-.4]
    D=[(.1^j).*[.4+.1*j,-.2+.03*j] for j in 1:7];original=deepcopy(D)
    U=zeros(5,5);U!(5,U)
    solver=DenseNewton(zeros(2),zeros(2),1.,1.,:none)
    cache=QNDFConstantCache{5}(old_h,order,D,U,zeros(5,5),zeros(5,5),solver,Float64.(γₖ),deepcopy(D),order,5,0)
    calc_R!(cache.R,ratio,order);mul!(cache.RU,cache.R,U)
    transformed=[sum(original[i]*cache.RU[i,j] for i in 1:order) for j in 1:order]
    integrator=Integrator(t,h,copy(y),copy(y),Force(I,forced),nothing,cache,(kappa=kappa,),Stats(0),
        (adaptive=false,dense=true,calck=false),false,zeros(2),[zeros(2) for _ in 1:5])
    perform_step!(integrator,cache,false)
    correction=integrator.u-solver.z;error_constant=Float64(kappa[order]*γₖ[order]+1//(order+1))
    records["p$order"*"_r$ratio"*(forced ? "_forced" : "_autonomous")]=Dict("order"=>order,"ratio"=>ratio,
        "initial"=>y,"t"=>t,"h"=>h,"previous_h"=>old_h,"input_differences"=>original,
        "transformed_differences"=>transformed,"difference_transform"=>cache.RU[1:order,1:order],
        "predictor"=>solver.z,"state"=>integrator.u,"error"=>error_constant*correction,
        "output_differences"=>cache.D[1:order+2],"beta"=>solver.γ,"error_constant"=>error_constant)
end
json(stdout,Dict("monorepo_tree"=>tree,"kappa"=>collect(Float64.(kappa)),"harmonics"=>collect(Float64.(γₖ)),
    "source_sha256"=>Dict(utils_file=>bytes2hex(sha256(utils)),step_file=>bytes2hex(sha256(steps))),
    "qualification"=>"Pinned QNDF constant-cache warm step and U/calc_R/update_D helpers; broadcast annotations removed, independent dense Newton solver, supplied warm differences/order/ratios; startup and upstream adaptive order/controller not reproduced",
    "records"=>records));println()
