import numpy as np


def matrix(A,B):
    if len(A[0])!=len(B):
        return None
    C = np.zeros((len(A),len(B[0])))
    C = np.matmul(A,B)
    # for i in range(len(A)):
    #     for j in range(len(B[0])):
    #         for k in range(len(B)):
    #             C[i][j] += A[i][k] * B[k][j]
    return C

# def matmul_tp(X,W,block_m,block_k,block_n):

def block_matmul(A,B):
    n,k = A.shape
    k2,m= B.shape

    # A1 = A[0:n/2,:]
    # A2 = A[n/2:,:]
    # B1 = B[:,0:m/2]
    # B2 = B[:,m/2:]

    A1 = A[:,0:k//2]
    A2 = A[:,k//2:]
    B1 = B[0:k//2,:]
    B2 = B[k//2:,:]

    C = np.zeros(n,m)

    C1 = matrix(A1,B1) 
    C2 = matrix(A2,B2)
    C = C1 + C2

    return C






    C = np.zeros((n,m))

    for i0 in range(0,n,block_size):
        for j0 in range(0,m,block_size):
            for k0 in range(0,k,block_size):
                i_end = min(i0+block_size,n)
                j_end = min(j0+block_size,m)
                k_end = min(k0+block_size,k)

                A_block = A[i0:i_end,k0:k_end]
                B_block = B[k0:k_end,j0:j_end]

                C[i0:i_end,j0:j_end] = matrix(A_block,B_block)
    return C

if __name__ == '__main__':
    A = np.random.rand(1000,2000)
    B = np.random.rand(2000,5000)
    C_naive = matrix(A,B)
    C_block = block_matmul(A,B)
    print("diff",C_block-C_naive)
    # C = block_matmul(A,B,100)
    # print(C)

