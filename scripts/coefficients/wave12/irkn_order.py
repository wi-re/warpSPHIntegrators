"""Independent exact Taylor jets for the fixed-step improved Nyström recurrences.

Treat f(t,q) partials as independent symbols. D=partial_t+v*partial_q+
f*partial_v differentiates along the exact trajectory. Old substage history
is expanded at its actual old anchor, including IRKN3's n-2 acceleration.
This proves local component consistency through the claimed orders; source
startup, zero stability, ownership and measured work have separate gates.
"""
import json
import sympy as s
h,q,v=s.symbols('h q v');p=4
jets={(i,j):s.Symbol(f'f_{i}_{j}') for i in range(7) for j in range(7-i)}
f=jets[0,0]
def D(expr):
 result=s.diff(expr,q)*v+s.diff(expr,v)*f
 for (i,j),symbol in jets.items():
  if i+j<6:result+=s.diff(expr,symbol)*(jets[i+1,j]+v*jets[i,j+1])
 return s.expand(result)
qd=[q,v,f]
for i in range(4):qd.append(D(qd[-1]))
def cut(expr,n):return s.series(s.expand(expr),h,0,n+1).removeO().expand()
def trajectory(derivative,c,n):return sum(qd[k+derivative]*(c*h)**k/s.factorial(k) for k in range(n+1))
def force(c,position,n):
 delta=cut(position-q,n)
 return cut(sum(jets[i,j]*(c*h)**i*delta**j/s.factorial(i)/s.factorial(j) for i in range(n+1) for j in range(n+1-i)),n)
records={}
for p in [3,4]:
 def nodes(k):
  oldq=trajectory(0,k,p);oldv=trajectory(1,k,p);a=trajectory(2,k-(p==3),p-2)
  c,a21=(s.Rational(1,2),s.Rational(1,8)) if p==3 else (s.Rational(1,4),s.Rational(1,32))
  B=force(k+c,oldq+c*h*oldv+h*h*a21*a,p-1)
  if p==4:
   C=force(k+s.Rational(3,4),oldq+s.Rational(3,4)*h*oldv+h*h*s.Rational(9,32)*B,p-1)
   return B,C
  return B,
 new=nodes(0);old=nodes(-1);aold=trajectory(2,-1,p-1);vold=trajectory(1,-1,p-1)
 if p==3:
  pos=q+h*(s.Rational(3,2)*v-vold/2)+h*h*s.Rational(5,12)*(new[0]-old[0])
  vel=v+h*(s.Rational(2,3)*f+aold/3+s.Rational(5,6)*(new[0]-old[0]))
 else:
  pos=q+h*(s.Rational(3,2)*v-vold/2)+h*h*(s.Rational(7,24)*(new[0]-old[0])+(new[1]-old[1])/8)
  vel=v+h*(s.Rational(19,18)*f-aold/18-(new[0]-old[0])/6+s.Rational(11,18)*(new[1]-old[1]))
 residuals=[cut(pos-trajectory(0,1,p),p),cut(vel-trajectory(1,1,p),p)]
 assert residuals==[0,0],(p,residuals)
 records[f'IRKN{p}']={'position_residual_through_order':str(residuals[0]),
  'velocity_residual_through_order':str(residuals[1]),'order':p}
print(json.dumps({'construction':'Independent exact symbolic Taylor jets for a smooth general f(t,q), exact old trajectory anchors and correctly timed substages',
 'sympy':s.__version__,'records':records},indent=2))
