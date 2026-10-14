# Actual pinned warm kernels, with explicitly corrected startup-history slots.
# Reuse source cache allocation and accepted-state advancement from the audit.
redirect_stdout(devnull) do
 include(joinpath(@__DIR__,"irkn_audit.jl"))
end
function nonlinear_run(name,n)
 h=1/n;i,_,_=setup(name,h,"autonomous");calls=[]
 i.f=SplitForce((out,v,q,p,t)->begin
  target=exp(.7*t);out .= -4*q .- .2*q.^3 .+ 4.49*target .+ .2*target^3
  push!(calls,Dict("t"=>t,"q"=>copy(q),"a"=>copy(out)))
 end,i.f.f2)
 i.f.f1(i.fsalfirst.x[1],i.uprev.x[1],i.uprev.x[2],nothing,0.)
 rows=[]
 for step in 1:n
  i.iter=step;i.t=(step-1)*h;i.tprev=max(0.,i.t-h);i.u=copy(i.uprev)
  perform_step!(i,i.cache,false)
  if step==1
   tab=i.cache.tab;v0,q0=i.uprev.x;a0=zero(q0)
   i.f.f1(a0,v0,q0,nothing,0.)
   copyto!(i.cache.tmp2.x[2],a0);copyto!(i.cache.tmp2.x[1],i.cache.k.x[1])
   qm=q0+h*tab.c1*v0+h*h*tab.a21*a0
   i.f.f1(i.cache.k₂.x[1],v0,qm,nothing,tab.c1*h)
   if name=="IRKN4"
    qm=q0+h*tab.c2*v0+h*h*tab.a32*i.cache.k₂.x[1]
    i.f.f1(i.cache.k₃.x[1],v0,qm,nothing,tab.c2*h)
   end
  end
  push!(rows,Dict("t"=>step*h,"q"=>copy(i.u.x[2]),"v"=>copy(i.u.x[1])))
  old=copy(i.uprev);i.uprev=copy(i.u);i.uprev2=old
  for j in 1:2;copyto!(i.fsalfirst.x[j],i.cache.k.x[j]);end
 end
 return Dict("steps"=>n,"h"=>h,"trajectory"=>rows,"calls"=>calls,
  "position_error"=>abs(i.u.x[2][1]-exp(.7)),"velocity_error"=>abs(i.u.x[1][1]-.7*exp(.7)))
end
records=Dict(name=>[nonlinear_run(name,n) for n in (16,32,64,128)] for name in ("IRKN3","IRKN4"))
for (name,rows) in records
 order=name=="IRKN3" ? 3 : 4
 for key in ("position_error","velocity_error")
  measured=fitted_order(rows,key)
  @assert measured!==nothing && measured>order-.2 (name,key,[r[key] for r in rows])
 end
end
json(stdout,Dict("monorepo_tree"=>tree,"source_sha256"=>Dict(f=>bytes2hex(sha256(sources[f])) for f in files),
 "transformation"=>"Actual mutable bootstrap and warm kernels; after the first step replace only old/end physical accelerations and correctly timed old substages. Source-only extra startup evaluations remain in calls. This is a corrected-seed transformation, not unmodified source equivalence.",
 "problem"=>"q''=-4q-.2q^3+4.49exp(.7t)+.2exp(2.1t); q=exp(.7t), q0=1, v0=.7",
 "records"=>records));println()
