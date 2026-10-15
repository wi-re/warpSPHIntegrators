"""solvers.matrix_functions — extracted numerical definitions with unchanged formulas."""


import torch


def operator_function(expr,Z,cache):
    """Evaluate a scalar entire-function expression on the same square operator."""
    n=Z.shape[0];I=torch.eye(n,dtype=Z.dtype,device=Z.device)
    if isinstance(expr,(int,float)):return expr*I
    op,*args=expr
    if op=='z':return Z
    if op=='phi':
        k,c=args;key=(k,c)
        if key not in cache:
            # Augmented-block exponential avoids cancellation near zero and the
            # unscaled Taylor overflow/cancellation of the historical driver.
            blocks=[]
            for i in range(k+1):
                blocks.append(torch.cat([c*Z if i==j==0 else I if j==i+1 else torch.zeros_like(Z)
                    for j in range(k+1)],dim=1))
            cache[key]=torch.matrix_exp(torch.cat(blocks,dim=0))[:n,k*n:(k+1)*n]
        return cache[key]
    if op=='pow':return torch.linalg.matrix_power(operator_function(args[0],Z,cache),args[1])
    values=[operator_function(a,Z,cache) for a in args]
    if op=='add':return sum(values,torch.zeros_like(Z))
    if op=='mul':
        value=I
        for v in values:value=value@v
        return value
    raise ValueError('Invalid operator expression')
