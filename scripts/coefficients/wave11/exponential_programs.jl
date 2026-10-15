include_string(Main,split(read(joinpath(@__DIR__,"exponential.jl"),String),"records=[]\nfor name";limit=2)[1],joinpath(@__DIR__,"exponential.jl"))
struct Sym <: Number
 x::Any
end
S(x::Sym)=x
S(x)=Sym(x)
Base.zero(::Type{Sym})=Sym(0)
Base.zero(::Sym)=Sym(0)
Base.one(::Type{Sym})=Sym(1)
Base.one(::Sym)=Sym(1)
Base.conj(x::Sym)=x
Base.transpose(x::Sym)=x
Base.convert(::Type{Sym},x::Number)=S(x)
Base.promote_rule(::Type{Sym},::Type{T}) where T<:Number=Sym
for op in (:+,:-,:*,:/)
 @eval Base.$op(a::Sym,b::Sym)=Sym(Any[$(string(op)),a.x,b.x])
end
Base.:-(a::Sym)=Sym(Any["*",-1,a.x])
Base.:^(a::Sym,k::Integer)=k==0 ? one(a) : prod(fill(a,k))
Base.exp(A::Matrix{Sym})=reshape([Sym(Any["phi",0,A[1].x])],1,1)
phi(A::Matrix{Sym},k)=[reshape([Sym(Any["phi",j,A[1].x])],1,1) for j in 0:k]
struct SymBasis
 A;v
end
arnoldi(A::Matrix{Sym},v;kwargs...)=SymBasis(A,v)
phiv(t,K::SymBasis,k;kwargs...)=phiv(t,K.A,K.v,k;kwargs...)
mutable struct SymbolOracle
 t;dt;uprev;u;f;p;fsalfirst;fsallast;opts;stats;k;EEst;alg
end
calc_J(i::SymbolOracle,cache)=reshape([Sym("Z")],1,1)
struct SymbolForce
 calls
end
function (f::SymbolForce)(u,p,t)
 name="F"*string(length(f.calls))
 push!(f.calls,Dict("state"=>u[1].x,"time"=>t))
 return [Sym(name)]
end
# Semilinear cached kernels require an actual SplitFunction dispatch; f2 returns
# full formal rates minus Z*u, recording each duplicate call separately.
# Redefine field scaffolds before including source in production version.
programs=Dict()
for name in METHODS
 calls=[];full=SymbolForce(calls);y=[Sym("Y")];Z=reshape([Sym("Z")],1,1)
 force=name in SEMILINEAR ? SplitFunction(LinearPart(Z),(u,p,t)->full(u,p,t)-Z*u) : full
 alg=getfield(Main,Symbol(name))(!(name in SEMILINEAR),1,0,false)
 opts=(adaptive=true,abstol=1.,reltol=1e-14,internalnorm=(x,t=0.)->0.,internalopnorm=opnorm)
 i=SymbolOracle(0.,1.,y,y,force,nothing,force(y,nothing,0.),zero(y),opts,Stats(0,0),Any[zero(y),zero(y)],0.,alg)
 cache=getfield(Main,Symbol(name*"ConstantCache"))(name in SEMILINEAR ? expRK_operators(alg,1.,Z) : nothing)
 last_error[]=nothing;perform_step!(i,cache,false)
 programs[name]=Dict("calls"=>calls,"output"=>i.u[1].x,"error"=>last_error[]===nothing ? nothing : last_error[][1].x)
end
json(stdout,programs)
