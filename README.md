## DATE : 15/09/2026





## AIM : 






Introduce matrix operations used in ML: transpose, inverse, solve linear systems, eigen 
decomposition.





## ALGORITHM (brief): 





Direct use of numpy.linalg routines; relate to linear regression normal equations. 






## THEORY: 







Linear algebra (matrix inversion, eigen-decomposition) underpins PCA, linear models, 
and many optimization problems. Conditioning/invertibility matters.




```

## INPUT


import numpy as np 
A = np.array([[2.0, 1.0],[1.0, 3.0]]) 
b = np.array([1.0, 2.0]) 
print("Matrix A:\n", A) 
print("\nTranspose A^T:\n", A.T) 
print("\nInverse A^-1:\n", np.linalg.inv(A)) 
eigvals, eigvecs = np.linalg.eig(A) 
print("\nEigenvalues:", np.round(eigvals,4)) 
x = np.linalg.solve(A,b) 
print("\nSolve A x = b --> x:", np.round(x,4))



## OUTPUT:


Matrix A: 
[[2. 1.] 
[1. 3.]] 
Transpose A^T: 
[[2. 1.] 
[1. 3.]] 
Inverse A^-1: 
[[ 0.6 -0.2] 
[-0.2  0.4]] 
Eigenvalues: [3.3028 1.6972] 
Solve A x = b --> x: [0.2 0.6]

```



## RESULT




Students see concrete matrix computations and how to solve linear systems; relate 
solve() to normal equations in linear regression. 
