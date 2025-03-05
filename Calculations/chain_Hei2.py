import numpy as np
import numpy.linalg as LA
import opt_einsum as oe
from scipy.linalg import expm


def heisenberg_ham(h):
    d = 2

    su=np.array([[0, 1], [0, 0]])#自旋升算符
    sm=np.array([[0, 0], [1, 0]])#自旋降算符
    sz=np.array([[1, 0], [0, -1]])/2#z方向自旋算符

    # sx           = np.array([[0,1 ], [1,0]])/2
    # sy_reduce    = np.array([[0,-1], [1,0]])/2
    # sz           = np.array([[1, 0], [0,-1]])/2

    # ham = np.kron(sx,sx) - np.kron(sy_reduce,sy_reduce) + np.kron(sz,sz)
    ham=(0.5*np.kron(su,sm)+0.5*np.kron(sm,su)+h*np.kron(sz,sz)).reshape(d,d,d,d)

    return ham

def calc_gate(tau, h):
    gate = expm(-tau*h.reshape(d**2,d**2)).reshape(d,d,d,d)
    return gate

def apply_gate_update(tau, ntimes,
                       hab, hba,
                       sab, sba,
                       A, B):
    # global sab,sba,asab,asba,A,B#只需要修改这6个全局变量，因而只声明这6个
    
    gab = calc_gate(tau, hab)
    gba = calc_gate(tau, hba)

    for i in range(ntimes):
        c1=sba.shape[0]
        #A-B演化
        res=oe.contract('ijkl,ab,bkc,cd,dle,ef->aijf',gab,np.diag(sba),A,np.diag(sab),B,np.diag(sba))
        res_norm = np.linalg.norm(res)
        print("estimate val: ", -np.log(res_norm)/tau)

        U,s,Vh=LA.svd(res.reshape(d*c1,d*c1))
        c2=min(chi,len(s))#奇异值最多不超过chi个

        sba_inv = 1/sba
        A=oe.contract('ij,jkl->ikl',np.diag(sba_inv),U[:,:c2].reshape(c1,d,c2))
        sab=s[:c2]/LA.norm(s[:c2])
        B=oe.contract('ijk,kl->ijl',Vh[:c2,:].reshape(c2,d,c1),np.diag(sba_inv))

        if i == ntimes-1:
            ab_sss_L=oe.contract('ij,ik,jlm,kln->mn',np.diag(sab),np.diag(sab),B.conj(),B)
            ab_sss_R=oe.contract('ikl,jkm,ln,mn->ij',B.conj(),B,np.diag(sba),np.diag(sba))

        #B-A演化
        res=oe.contract('ijkl,ab,bkc,cd,dle,ef->aijf',gba,np.diag(sab),B,np.diag(sba),A,np.diag(sab))
        U,s,Vh=LA.svd(res.reshape(d*c2,d*c2))
        c3=min(chi,len(s))

        sab_inv = 1/sab
        B=oe.contract('ij,jkl->ikl',np.diag(sab_inv),U[:,:c3].reshape(c2,d,c3))
        sba=s[:c3]/LA.norm(s[:c3])
        A=oe.contract('ijk,kl->ijl',Vh[:c3,:].reshape(c3,d,c2),np.diag(sab_inv))

        if i == ntimes-1:
            ba_sss_L=oe.contract('ij,ik,jlm,kln->mn',np.diag(sab),np.diag(sab),B.conj(),B)
            ba_sss_R=oe.contract('ikl,jkm,ln,mn->ij',B.conj(),B,np.diag(sba),np.diag(sba))


    '''说明sba出问题了'''

    ab_diff_L = LA.norm(ab_sss_L - np.eye(chi,chi))
    ab_diff_R = LA.norm(ab_sss_R - np.eye(chi,chi))

    ba_diff_L = LA.norm(ba_sss_L - np.eye(chi,chi))
    ba_diff_R = LA.norm(ba_sss_R - np.eye(chi,chi))

    print(f"[gate_ab] B: diff_L = {ab_diff_L:-20.15e},          diff_R = {ab_diff_R:-20.15e}")
    print(f"[gate_ba] B: diff_L = {ba_diff_L:-20.15e},          diff_R = {ba_diff_R:-20.15e}")
    # print(LA.norm(sss_L-np.eye(chi,chi)))

    return A, B, sab, sba

if __name__ == "__main__":
    chi=30; d=2#MPS的几何指标与物理指标维数
    h=1#sz的权重
    A=np.random.randn(chi,d,chi)
    B=np.random.randn(chi,d,chi)
    sab,sba=np.ones(chi),np.ones(chi)


    hab  = heisenberg_ham(h)
    hba  = heisenberg_ham(h)

    for tau in [1, 0.1, 0.05, 0.02, 0.01]:
        A, B, sab, sba = apply_gate_update(tau, 2000, hab, hba, sab, sba, A, B)


