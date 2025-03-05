#一维链Heisenberg模型TEBD求基态以及基态能
import torch
import numpy as np
import opt_einsum as oe
from scipy.linalg import expm
from numpy import linalg as LA
import matplotlib.pyplot as plt
chi=50; d=2#MPS的几何指标与物理指标维数
h=1#sz的权重
A=np.random.randn(chi,d,chi)
B=np.random.randn(chi,d,chi)
su=np.array([[0, 1], [0, 0]])#自旋升算符
sm=np.array([[0, 0], [1, 0]])#自旋降算符
sz=np.array([[1, 0], [0, -1]])/2#z方向自旋算符
sab,sba,asab,asba=np.ones(chi),np.ones(chi),np.ones(chi),np.ones(chi)#键矩阵及其逆矩阵的对角元
#两体哈密顿量
hab=(0.5*np.kron(su,sm)+0.5*np.kron(sm,su)+h*np.kron(sz,sz)).reshape(d,d,d,d)
# hab=np.kron(sz,sz).reshape(d,d,d,d)
# hab = np.eye(d**2,d**2).reshape(d,d,d,d)
hba=hab
tau=0.1#演化间隔τ
#演化算符
gab=expm(-1*tau*hab.reshape(d**2,d**2)).reshape(d,d,d,d)
gba=expm(-1*tau*hba.reshape(d**2,d**2)).reshape(d,d,d,d)

def ctg(num):#更新tau和gab,gba
    global tau,gab,gba
    tau=num
    print(f"tau = {tau:.15f}")
    gab=expm(-tau*hab.reshape(d**2,d**2)).reshape(d,d,d,d)
    gba=expm(-tau*hba.reshape(d**2,d**2)).reshape(d,d,d,d)
    return 0

lim=1e-10#范数差判据
# nshape=(d*chi,d*chi)#每次演化后得到的张量被reshape后的维数

def ch(verbose=False):#演化一步A,B
    global sab,sba,asab,asba,A,B#只需要修改这6个全局变量，因而只声明这6个
    c1=sba.shape[0]
    #A-B演化
    res=oe.contract('ijkl,ab,bkc,cd,dle,ef->aijf',gab,np.diag(sba),A,np.diag(sab),B,np.diag(sba))
    U,s,Vh=LA.svd(res.reshape(d*c1,d*c1))
    c2=min(chi,len(s))#奇异值最多不超过chi个
    A=oe.contract('ij,jkl->ikl',np.diag(asba),U[:,:c2].reshape(c1,d,c2))
    sab=s[:c2]/LA.norm(s[:c2])
    
    asab=1/(sab+1e-20)
    # asab=1/(sab)
    B=oe.contract('ijk,kl->ijl',Vh[:c2,:].reshape(c2,d,c1),np.diag(asba))

    ab_sss_L=oe.contract('ij,ik,jlm,kln->mn',np.diag(sab),np.diag(sab),B.conj(),B)
    ab_sss_R=oe.contract('ikl,jkm,ln,mn->ij',B.conj(),B,np.diag(sba),np.diag(sba))

    
    #B-A演化
    res=oe.contract('ijkl,ab,bkc,cd,dle,ef->aijf',gba,np.diag(sab),B,np.diag(sba),A,np.diag(sab))
    U,s,Vh=LA.svd(res.reshape(d*c2,d*c2))
    c3=min(chi,len(s))
    B=oe.contract('ij,jkl->ikl',np.diag(asab),U[:,:c3].reshape(c2,d,c3))
    sba=s[:c3]/LA.norm(s[:c3])
    
    asba=1/(sba+1e-20)
    # print(f"1/sba = {asba}")
    # asba=1/(sba)
    # print(LA.norm(asba)**2,LA.norm(sba)**2)
    #print(sba)
    A=oe.contract('ijk,kl->ijl',Vh[:c3,:].reshape(c3,d,c2),np.diag(asab))
    # print(LA.norm(Vh)**2)
    # print(LA.norm((Vh[:c3,:])@(Vh[:c3,:].T))**2)
    # sss=oe.contract('ikl,jkm,ln,mn->ij',A.conj(),A,np.diag(sab),np.diag(sab))
    # sss=oe.contract('ij,ik,jlm,kln->mn',np.diag(sba),np.diag(sba),A.conj(),A)

    ba_sss_L=oe.contract('ij,ik,jlm,kln->mn',np.diag(sab),np.diag(sab),B.conj(),B)
    ba_sss_R=oe.contract('ikl,jkm,ln,mn->ij',B.conj(),B,np.diag(sba),np.diag(sba))


    '''说明sba出问题了'''
    if verbose:
        ab_diff_L = LA.norm(ab_sss_L - np.eye(chi,chi))
        ab_diff_R = LA.norm(ab_sss_R - np.eye(chi,chi))

        ba_diff_L = LA.norm(ba_sss_L - np.eye(chi,chi))
        ba_diff_R = LA.norm(ba_sss_R - np.eye(chi,chi))

        print(f"[gate_ab] B: diff_L = {ab_diff_L:-20.15e},          diff_R = {ab_diff_R:-20.15e}")
        print(f"[gate_ba] B: diff_L = {ba_diff_L:-20.15e},          diff_R = {ba_diff_R:-20.15e}")
        # print(LA.norm(sss_L-np.eye(chi,chi)))
    return 0


def getsm():#计算左背景sig和右背景mu
    ds=A.shape[0]
    sig=np.random.rand(ds,ds)
    dm=B.shape[2]
    mu=np.random.rand(dm,dm)

    # sig = np.eye(ds,ds)
    # mu  = np.eye(dm,dm)
    # ii=0
    # print(LA.norm(sig),LA.norm(A),LA.norm(B),LA.norm(B.conj()),LA.norm(sab),LA.norm(sba))
    # print(LA.norm(sig))
    # transfer_mat_left = oe.contract("ik,jl,kmn,lmo,np,oq,prs,qrt->ij st",
    #                                 np.diag(sba),np.diag(sba),A.conj(),A,np.diag(sab),np.diag(sab),B.conj(),B
    #                                 ).reshape(ds**2,ds**2)
    # w,v = np.linalg.eig(transfer_mat_left)
    # print(w)
    # print(v[:,0].reshape(ds,ds))
    # input()

    #=======================================
    while True:
        # ii+=1
        # print(ii)
        sig1=oe.contract('ij,ik,jl,kmn,lmo,np,oq,prs,qrt->st',sig,np.diag(sba),np.diag(sba),A.conj(),A,np.diag(sab),np.diag(sab),B.conj(),B)
        coeff = LA.norm(sig1)
        sig1=sig1/coeff
        # print(f"coeff={coeff:.15f}")
        #print(LA.norm(sig1),LA.norm(A),LA.norm(B),LA.norm(B.conj()),LA.norm(sab),LA.norm(sba))
        # print(LA.norm(sig1))
        if LA.norm(sig1-sig)<lim:
            sig=sig1
            break
        sig=sig1
    while True:
        mu1=oe.contract('ikl,jkm,ln,mo,npq,opr,qs,rt,st->ij',A.conj(),A,np.diag(sab),np.diag(sab),B.conj(),B,np.diag(sba),np.diag(sba),mu)
        coeff = LA.norm(mu1)
        mu1=mu1/coeff
        # print(f"coeffR={coeff}")
        # print(LA.norm(mu1),LA.norm(A),LA.norm(B),LA.norm(B.conj()),LA.norm(sab),LA.norm(sba))
        if LA.norm(mu1-mu)<lim:
            mu=mu1
            break
        mu=mu1

    return sig,mu

def getH():#计算哈密顿量H/N
    sig,mu=getsm()
    Ham=oe.contract('ij,ik,jl,kmo,lnp,oq,pr,qsu,rtv,msnt,uw,vx,wx->',sig,np.diag(sba),np.diag(sba),A.conj(),A,np.diag(sab),np.diag(sab),B.conj(),B,hab,np.diag(sba),np.diag(sba),mu)
    # print(
    #     f"""
    #     hab = {hab}

    #     <H> = {Ham}
    #     """
    # )
    return Ham

def inn():#计算MPS的内积
    sig,mu=getsm()
    # print(f"sig={sig}")
    inp=oe.contract('ij,ik,jl,kmn,lmo,np,oq,prs,qrt,su,tv,uv->',sig,np.diag(sba),np.diag(sba),A.conj(),A,np.diag(sab),np.diag(sab),B.conj(),B,np.diag(sba),np.diag(sba),mu)

    return inp

def TEBD():
    # while tau>0.0000000001:
    for j in [0.1,0.05,0.02,0.01,0.008,0.005,0.002,0.001,0.0001,0.00001]:
    # global tau
    # tau = 0.1
        ctg(j)
        for i in range(1000):
            ch(verbose=(i%200==0))
        # ctg()
    
    H1=getH()

    # while True:
    #     ch()
    #     H2=getH()
    #     if abs(H1-H2)<lim:
    #         break
    #     H1=H2
    return H1

'''for h in range (0,0.01,2):
    #两体哈密顿量
    hab=(np.kron(su,sm)+np.kron(sm,su)+h*np.kron(sz,sz)).reshape(d,d,d,d)
    hba=hab
    tau=0.1#演化间隔τ
    #演化算符
    gab=expm(-1*tau*hab.reshape(d**2,d**2)).reshape(d,d,d,d)
    gba=expm(-1*tau*hba.reshape(d**2,d**2)).reshape(d,d,d,d)'''
H=TEBD()
N=inn()
print(H,N,H/N)
#print("\n")
#print(A[A>1e-2])
#sig,mu=getsm()
#print(LA.norm(A))
#print(sab,sba)