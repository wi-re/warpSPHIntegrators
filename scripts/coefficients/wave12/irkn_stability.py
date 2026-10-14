"""Independent oscillator amplification matrix, including retained force history."""
import json
import numpy as np


def amplification(order,z):
    # Coordinates q, h*v, h*v_old, h²*a_old, h²*B_old, [h²*C_old].
    def advance(state):
        q,v,vo,ao,bo,*co=state;a=z*q
        b=z*(q+(.5 if order==3 else .25)*v+(ao/8 if order==3 else a/32))
        c=z*(q+.75*v+9/32*b) if order==4 else None
        db=b-bo;dc=c-co[0] if order==4 else None
        q1=q+1.5*v-.5*vo+(5/12*db if order==3 else 7/24*db+dc/8)
        v1=v+(2/3*a+ao/3+5/6*db if order==3 else 19/18*a-ao/18-db/6+11/18*dc)
        return np.array([q1,v1,v,a,b]+([c] if order==4 else []))
    return np.column_stack([advance(x) for x in np.eye(order+2)])


records={}
for order in [3,4]:
    matrix=amplification(order,0.);roots=np.linalg.eigvals(matrix)
    assert sum(abs(roots-1)<1e-14)==2 and sum(abs(roots)<1e-14)==order
    rows=[]
    for frequency_step in [0.,.01,.1,.25,.5,1.,1.5,2.]:
        roots=np.linalg.eigvals(amplification(order,-frequency_step**2))
        rows.append(dict(omega_h=frequency_step,spectral_radius=float(max(abs(roots))),
            roots=[[float(z.real),float(z.imag)] for z in roots]))
    records[f'IRKN{order}']=rows
assert records['IRKN3'][4]['spectral_radius']<1
assert records['IRKN4'][4]['spectral_radius']>1.00003
print(json.dumps(dict(coordinates='q, h*v, h*v_old, h^2*a_old, h^2*B_old, optional h^2*C_old',
    zero_force='Two physical unit roots (q drift and constant v); all history-only roots zero. Drift coupling is O(h) in physical coordinates, so powers stay bounded on fixed time intervals.',
    qualification='Linear oscillator amplification including parasitic history modes. IRKN4 has slight amplification even at small positive omega*h; no undamped-oscillator absolute-stability or long-time energy claim.',
    numpy=np.__version__,records=records),indent=2))
