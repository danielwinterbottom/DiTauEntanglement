import numpy as np

def EntanglementVariables(C, Bplus=np.array([[0],[0],[0]]), Bminus=np.array([[0],[0],[0]])):
    '''
    Compute concurrence and m12 variables
    Note Bplus and Bminus are not expected to affect these variables at all but they are used as optional inputs in case some intermediate matrices are to be returned as well 
    '''

    # Pauli matrices
    sig1 = np.array([[0, 1],
                     [1, 0]])
    
    sig2 = np.array([[0, -1j],
                     [1j, 0]], dtype=complex)
    
    sig3 = np.array([[1, 0],
                     [0, -1]])
    
    pauli_matrices = [sig1, sig2, sig3]
    
    # identity matrix
    I    = np.array([[1, 0],
                     [0, 1]])
    
    rho1 = np.kron(I, I)
    
    rho2=sum(Bplus[i] * np.kron(pauli_matrices[i], I) for i in range(3))
    
    rho3=sum(Bminus[j] * np.kron(I, pauli_matrices[j]) for j in range(3))
    
    rho4 = np.zeros((4, 4), dtype=complex)  # Initialize a 4x4 complex matrix
    for i in range(3):
        for j in range(3):
            rho4 += C[i, j] * np.kron(pauli_matrices[i], pauli_matrices[j])
    
    
    rho = 1./4*(rho1+rho2+rho3+rho4) 

   
    trace = np.trace(rho)

    rhostar = np.conj(rho)
    
    z = np.kron(pauli_matrices[1], pauli_matrices[1])
   
    # using formulas from https://journals.aps.org/prd/pdf/10.1103/PhysRevD.107.093002 
    # seems to be different to https://arxiv.org/pdf/2405.09201 but gives the expected answer
    rho_tilde = z*rhostar*z
    
    R = np.sqrt(np.sqrt(rho)*rho_tilde*np.sqrt(rho))
    R_EVs = sorted(np.linalg.eigvals(R),reverse=True)
    con = R_EVs[0]-sum(R_EVs[1:])


    M = C*C.T
    M_EVs = sorted(np.linalg.eigvals(M),reverse=True)
    m12 = M_EVs[0]+M_EVs[1]
  
    # out also try alternative formulation from https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.80.2245

    return (con.real, m12)

if __name__ == '__main__':

    C = np.array([[0.4878, 0,       0     ],
              [0,      -0.4878, 0.0011],
              [0,      0.0011,  1     ]])

    B = np.array([[0],
              [0.0001],
              [0.2194]])

    B = np.array([[0],
                  [0],
                  [0]])

#    C = np.array([[1, 0, 0],
#                  [0, 1, 0],
#                  [0, 0, -1]]) # matrix for scalar Higgs    

    con, m12 = EntanglementVariables(C,B,B)

    print('concurrence = %.4f' % con)
    print('m12 = %.3f' % m12)
