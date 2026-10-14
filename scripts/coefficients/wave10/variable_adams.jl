# Extract variable-step Adams coefficient maps from pinned constant-cache helpers.
using LinearAlgebra, SHA, Pkg
root=ARGS[1]
tree=bytes2hex(Pkg.GitTools.tree_hash(root))
@assert tree=="bae9b8e5f1d00e51c554648e5094e215164a19f8"
file=joinpath(root,"lib/OrdinaryDiffEqAdamsBashforthMoulton/src/adams_utils.jl")
source=read(file,String)
section(start,stop)=split(split(source,start;limit=2)[2],stop;limit=2)[1]
constant_branches(text)=replace(text,r"(?s)if cache isa OrdinaryDiffEqMutableCache\n.*?else\n(.*?)end"=>s"\1")
first="function ϕ_and_ϕstar!(cache, du, k)"*section("function ϕ_and_ϕstar!(cache, du, k)","function ϕ_and_ϕstar!(cache::Union")
other="function ϕ_np1!(cache, du_np1, k)"*section("function ϕ_np1!(cache, du_np1, k)","# Coefficients for the implicit Adams methods")
include_string(Main,constant_branches(first),file)
include_string(Main,constant_branches(other),file)
json_source=read(joinpath(@__DIR__,"../wave9/higher_stages.jl"),String)
include_string(Main,"function json(io,value)"*split(split(json_source,"function json(io,value)";limit=2)[2],"\nschemes=Dict()";limit=2)[1])
mutable struct AdamsOracleCache
    dts::Vector{Float64}
    c::Matrix{Float64}
    g::Vector{Float64}
    ϕ_n::Vector{Vector{Float64}}
    ϕstar_n::Vector{Vector{Float64}}
    ϕstar_nm1::Vector{Vector{Float64}}
    β::Vector{Float64}
    ϕ_np1::Vector{Vector{Float64}}
end
records=Dict()
for p in 3:5, irregular in (false,true)
    times=irregular ? [-.7,-.46,-.31,-.1,0.][6-p:5] : collect((-(p-1)):0).*.1
    h=irregular ? .08 : .1
    cache=AdamsOracleCache(zeros(p+1),zeros(p+1,p+1),zeros(p+1),
        [zeros(p+1) for _ in 1:p+1],[zeros(p+1) for _ in 1:p+1],
        [zeros(p+1) for _ in 1:p+1],zeros(p+1),[zeros(p+1) for _ in 1:p+1])
    for k in 1:p
        step=k<p ? times[k+1]-times[k] : h
        cache.dts=vcat(step,cache.dts[1:end-1])
        du=zeros(p+1);du[k]=1
        ϕ_and_ϕstar!(cache,du,k)
        if k<p
            cache.ϕstar_nm1=copy(cache.ϕstar_n)
        end
    end
    g_coefs!(cache,p+1)
    ab=sum(cache.g[i]*cache.ϕstar_n[i] for i in 1:p)
    predictor=sum(cache.g[i]*cache.ϕstar_n[i] for i in 1:p-1)
    du=zeros(p+1);du[end]=1
    ϕ_np1!(cache,du,p+1)
    corrected=predictor+cache.g[p]*cache.ϕ_np1[p]
    records["p$(p)_$(irregular ? "irregular" : "constant")"]=Dict(
        "times"=>times,"h"=>h,"ab"=>ab/h,"abm_predictor"=>predictor/h,
        "abm"=>corrected/h,"ab_error"=>cache.g[p]*cache.ϕstar_n[p]/h,
        "abm_source_error"=>(cache.g[p+1]-cache.g[p])*cache.ϕ_np1[p+1]/h)
end
perform_file=joinpath(root,"lib/OrdinaryDiffEqAdamsBashforthMoulton/src/adams_bashforth_moulton_perform_step.jl")
perform_source=read(perform_file,String)
@assert occursin("251 * k + 646 * k1 - 264 * k2 + 106 * k3 - 19 * k4",perform_source)
fixed=Dict("predictor"=>reverse(records["p5_constant"]["ab"][1:5]),
    "corrector"=>reverse(records["p5_constant"]["abm"][2:6]),
    "qualification"=>"Classical AB5/AM5 PECE coefficients; repository uses DP5 startup and an explicit AB5 predictor, not the pinned source's complete cache/startup implementation")
json(stdout,Dict("monorepo_tree"=>tree,"package_version"=>"2.2.0",
    "source_file"=>"lib/OrdinaryDiffEqAdamsBashforthMoulton/src/adams_utils.jl",
    "source_sha256"=>bytes2hex(sha256(source)),
    "qualification"=>"Pinned constant-cache helper branches evaluated on vector basis histories; mutable broadcast branches omitted; startup/controller not reproduced",
    "perform_source_sha256"=>bytes2hex(sha256(perform_source)),"fixed_abm5"=>fixed,
    "records"=>records));println()
