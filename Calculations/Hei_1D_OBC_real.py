#一维链自旋模型TEBD求基态以及基态能(有限长开边界，带正则化)

###切记update时的截断一定要用svd, 否则就会算不对

import torch
import numpy as np
import opt_einsum as oe
from torch import linalg as LA
import os

torch.set_default_dtype(torch.float64)#默认数据类型修改为更高精度

def initialize(n_site, chi):#初始化, 创建MPS序列以及中心处的键矩阵lam
    d=2
    MPS  =  []
    # lam  =  torch.ones(chi)
    # MPS.append(torch.randn(1,d,chi))
    # for i in range (1,n_site-1):
    #     MPS.append(torch.randn(chi,d,chi))
    # MPS.append(torch.randn(chi,d,1))
    

    #converge faster
    lam =  torch.ones(1)
    a          =  torch.zeros(1,d,1)
    for i in range(n_site):
        a[0][(i+1)%2][0] = 0 
        a[0][i%2][0]     = 1
        MPS.append(a.clone())#不能append(a),否则初始MPS内所有元素都一样,都是最后一次更新后的a
    return  MPS, lam
# MPS, lam = initialize(50,6)
# print(lam[0],'\n',1/lam[0],'\n',LA.norm(lam[0]))
# print(len(MPS),len(lam))
# U,s,Vh=LA.svd(MPS[0])
# U=U[:,:len(s)]
# print('!\n',U)
# Vh=Vh[:len(s),:]
# ori = oe.contract('ij,j,jk->ik',U,s,Vh)
# print(ori-MPS[0].numpy())

def inner(n_site, MPS, lam):#内积, 本来需要复共轭, 但是全程都是实数, 故也用实数加快速度
    res = oe.contract('ijk,ijl,k,l->kl', MPS[0], MPS[0], lam, lam)
    for i in range (1,n_site-1):
        res = oe.contract('il,ikj,lkm->jm', res, MPS[i], MPS[i])
    # prod = np.float64(oe.contract('ik,ij,kj->', res, MPS[n_site-1], MPS[n_site-1]).item())
    prod = oe.contract('ij,ikl,jkl->', res, MPS[n_site-1], MPS[n_site-1]).item()

    return prod

def check_canonicalization(MPS, lam ,ca_center, n_site): #验证每个点的正则化, 具体形式是0~center-1左正则, center~n-1右正则
    l_check=[]
    l_check.append(("lam", LA.norm(lam).item()))

    judge_standard = 1e-14

    #0~center-1
    for i in range(ca_center):
        c1 = MPS[i].size(2)
        ans=LA.norm(torch.eye(c1,c1)-oe.contract('ijk,ijl->kl', MPS[i],MPS[i])).item()
        if (ans>judge_standard):
            l_check.append((i,ans))
    #center~n-1
    for i in range(ca_center, n_site):
        c1 = MPS[i].size(0)
        ans= LA.norm(torch.eye(c1,c1)-oe.contract('ikl,jkl->ij', MPS[i],MPS[i])).item()
        if (ans>judge_standard):
            l_check.append((i,ans))

    return l_check


def ave_mpo( MPS, lam, op1, op2):#算符平均值, op1为单体算符,如没有应该输入zeros(d,d), op2为最近邻两体算符,如没有应该输入zeros(d,d,d,d))
    ans = 0
    #0~1
    ans += oe.contract('ilj,j,jmk,lmno,inp,p,pok->', MPS[0], lam, MPS[1], op2, MPS[0], lam, MPS[1]).item()  #两点平均值
    ans += oe.contract('ilj,j,jnk,lm,imo,o,onk->', MPS[0], lam, MPS[1], op1, MPS[0], lam, MPS[1]).item()    #左单点平均值
    ans += oe.contract('ilj,j,jmk,mn,ilo,o,onk->', MPS[0], lam, MPS[1], op1, MPS[0], lam, MPS[1]).item()    #右单点平均值
    ml   =MPS[1]
    mr   =MPS[2]
    ld   =lam

    #1~n-1
    for i in range(1, n_site-1):
        c1       = ml.size(0)
        c3       = mr.size(2)
        res      = oe.contract('i,ijk,klm->ijlm', ld, ml, mr).reshape(c1*d,d*c3)
        U, s, Vh = LA.svd(res)
        c2       = min(chi, len(s))
        ml       = U[:, :c2].reshape(c1,d,c2)
        ld       = s[:c2]/LA.norm(s[:c2])
        mr       = Vh[:c2, :].reshape(c2,d,c3)
        #单点平均值只算右侧，因为左侧上一步算过了
        ans     += oe.contract('ilj,j,jmk,lmno,inp,p,pok->', ml, ld, mr, op2, ml, ld, mr).item()
        ans     += oe.contract('ilj,j,jmk,mn,ilo,o,onk->', ml, ld, mr, op1, ml, ld, mr).item()
        ml       = Vh[:c2, :].reshape(c2,d,c3)
        if i<n_site-2:
            mr = MPS[i+2]
        

    return ans

def canonicalize(n_site, ca_center, MPS, lam):#将正则中心从1(0~1之间)挪到ca_center(ca_center-1~ca_center之间)(QR分解,中心使用svd)
    MPS2 = []
    #0号点单独算,因为有lam且维数最小
    c1   = MPS[0].size(0)
    c3   = len(lam)
    res  = oe.contract('ijk,k->ijk', MPS[0], lam).reshape(c1*d,c3)
    Q, R = LA.qr(res)
    c2   = Q.size(1)
    MPS2.append(Q.reshape(c1,d,c2))

    #1~center-2
    for i in range(1, ca_center-1):
        c1  = R.size(0)
        c3  = MPS[i].size(2)
        res = oe.contract('ij,jkl->ikl', R, MPS[i]).reshape(c1*d,c3)
        Q, R= LA.qr(res)
        c2  = Q.size(1)
        MPS2.append(Q.reshape(c1,d,c2))

    #center-1~center
    c1       = R.size(0)
    c3       = MPS[ca_center].size(2)
    res      = oe.contract('ij,jkl,lmn->ikmn',R, MPS[ca_center-1], MPS[ca_center]).reshape(c1*d,d*c3)
    U, s, Vh = LA.svd(res)
    c2       = min(chi, len(s))
    MPS2.append(U[:, :c2].reshape(c1,d,c2))
    ld       = s[:c2]/LA.norm(s[:c2])
    MPS2.append(Vh[:c2, :].reshape(c2,d,c3))
    # print((MPS2[ca_center-1].shape),len(ld),(MPS2[ca_center].shape))
    #center+1~n-1,已经正则化了,直接加入
    for i in range(ca_center+1, n_site):
        MPS2.append(MPS[i])

    return MPS2, ld


def Hei_ham(Jx,Jy,Jz,h):#哈密顿量
    d=2

    sx           = torch.tensor([[0, 1], [1, 0]])/2
    sy_reduce    = torch.tensor([[0,-1], [1, 0]])/2
    sz           = torch.tensor([[1, 0], [0,-1]])/2

    ham1         = h*sz
    ham2         = (Jx*torch.kron(sx,sx) - Jy*torch.kron(sy_reduce,sy_reduce) + Jz*torch.kron(sz,sz)).reshape(d,d,d,d)

    return ham1, ham2

# ham1, ham2 = Hei_ham(1,1,1,1)
# ham1+=torch.tensor([[0, 1], [0, 0]])
# print(ham2,"\n",ham1.T)

def build_gate(tau, ham1, ham2):#建gate
    d=2

    dtemp1, utemp1 = LA.eigh(ham1)
    dex1  = torch.exp(-tau*dtemp1)
    gate1 = oe.contract('ij,j,jk->ik',utemp1,dex1,utemp1.T)
    # gate1 = torch.eye(d,d)

    dtemp2, utemp2 = LA.eigh(ham2.reshape(d**2,d**2))
    dex2  = torch.exp(-tau*dtemp2)
    # m     = oe.contract('ij,j,jk->ik',utemp2,dex2,utemp2.T)
    # print(m)
    gate2 = oe.contract('ij,j,jk->ik',utemp2,dex2,utemp2.T).reshape(d,d,d,d)
    
    return gate1, gate2

# ham1, ham2   = Hei_ham(1,1,1,0)
# gate1 ,gate2 = build_gate(1,ham1,ham2)
# print(ham1,"\n",ham2.reshape(2**2,2**2),"\n",gate1,"\n",gate2.reshape(2**2,2**2))

def apply_gate(tau, chi, ntimes, ham1, ham2, MPS, lam):#演化,先向右再向左sweep,等价于作用一个-2τH的gate

    gate1, gate2 = build_gate(tau, ham1, ham2)


    for itime in range(ntimes):#做ntimes次
        #向右依次演化
        #0~1
        c1       = MPS[0].size(0)
        c3       = MPS[1].size(2)
        res      = oe.contract('ik,jl,klmn,omp,p,pnq->oijq', gate1, gate1, gate2, MPS[0], lam, MPS[1]).reshape(c1*d,d*c3)
        U, s, Vh = LA.svd(res)
        c2       = min(chi, len(s))
        MPS[0]   = U[:, :c2].reshape(c1,d,c2)
        lam      = s[:c2]/LA.norm(s[:c2])
        MPS[1]   = Vh[:c2, :].reshape(c2,d,c3)
        #1~n-1
        for  i in range(1, n_site-1):
            c1       = MPS[i].size(0)
            c3       = MPS[i+1].size(2)
            res      = oe.contract('jk,iklm,n,nlo,omp->nijp', gate1, gate2, lam, MPS[i], MPS[i+1]).reshape(c1*d,d*c3)
            U, s, Vh = LA.svd(res)
            c2       = min(chi, len(s))
            MPS[i]   = U[:, :c2].reshape(c1,d,c2)
            lam      = s[:c2]/LA.norm(s[:c2])
            MPS[i+1] = Vh[:c2, :].reshape(c2,d,c3)
        #向左依次演化
        #n-1~n-2
        c1            = MPS[n_site-2].size(0)
        c3            = MPS[n_site-1].size(2)
        res           = oe.contract('ik,jl,klmn,omp,p,pnq->oijq', gate1, gate1, gate2, MPS[n_site-2], lam, MPS[n_site-1]).reshape(c1*d,d*c3)
        U, s, Vh      = LA.svd(res)
        c2            = min(chi, len(s))
        MPS[n_site-2] = U[:, :c2].reshape(c1,d,c2)
        lam           = s[:c2]/LA.norm(s[:c2])
        MPS[n_site-1] = Vh[:c2, :].reshape(c2,d,c3)
        #n-2~0
        for  i in range(n_site-2, 0, -1):
            c1       = MPS[i-1].size(0)
            c3       = MPS[i].size(2)
            res      = oe.contract('ij,jklm,nlo,omp,p->nikp', gate1, gate2, MPS[i-1], MPS[i], lam).reshape(c1*d,d*c3)
            U, s, Vh = LA.svd(res)
            c2       = min(chi, len(s))
            MPS[i-1] = U[:, :c2].reshape(c1,d,c2)
            lam      = s[:c2]/LA.norm(s[:c2])
            MPS[i]   = Vh[:c2, :].reshape(c2,d,c3)

        if((itime+1)%10 == 0 or (itime+1)%10 == 1):  #调试用，每10次循环的开头和结尾输出
            print('tau= ', f"{tau:.6f}","  step= ", itime)
            E_gs          = ave_mpo(MPS, lam, ham1, ham2)                                               #基态能
            inner_product = inner(n_site, MPS, lam)                                                     #|ψ|^2
            # MPS2, ld      = canonicalize(n_site, ca_center, MPS, lam)                                   #中心正则化
            # ee_spectrum   = -2*(ld**2 )*torch.log(ld)                                                   #中心纠缠谱
            # ee            = torch.sum(ee_spectrum).item()                                               #中心纠缠熵
            # l_check       = check_canonicalization(MPS2, ld, ca_center, n_site)                         #正则化不满足标准的点及其误差列表
            
            # mz1           = oe.contract('ilj,j,jnk,lm,imo,o,onk->',MPS2[ca_center-1],ld, MPS2[ca_center],sz,MPS2[ca_center-1],ld, MPS2[ca_center]).item()
            # mx1           = oe.contract('ilj,j,jnk,lm,imo,o,onk->',MPS2[ca_center-1],ld, MPS2[ca_center],sx,MPS2[ca_center-1],ld, MPS2[ca_center]).item()
            # mz2           = oe.contract('ilj,j,jmk,mn,ilo,o,onk->',MPS2[ca_center-1],ld, MPS2[ca_center],sz,MPS2[ca_center-1],ld, MPS2[ca_center]).item()
            # mx2           = oe.contract('ilj,j,jmk,mn,ilo,o,onk->',MPS2[ca_center-1],ld, MPS2[ca_center],sx,MPS2[ca_center-1],ld, MPS2[ca_center]).item()
            # smz           = mz1+mz2
            # ms1           = np.sqrt(mz1**2+mx1**2)
            # ms2           = np.sqrt(mz2**2+mx2**2)

            with open(file_path, 'a') as f:#a 是追加写入， w是覆盖写入
                print('tau=', f"{tau:.6f}","  step=", itime, "  E=", E_gs, " |ψ|^2=", f"{inner_product:.8f}" , file= f)
        # E_gs          = ave_mpo(ca_center, MPS, lam, ham1, ham2)
        # inner_product = inner(n_site, MPS, lam)
        # print('tau= ', f"{tau:.6f}","  step= ", itime, "  E= ", E_gs, " |ψ|^2= ", f"{inner_product:.8f}" )
        # MPS, lam =canonicalize(n_site, ca_center, MPS, lam)

    return MPS, lam


if __name__ =="__main__":
    n_site             =  20                                #MPS长度      
    chi_list           =  [30, 60, 90, 120]                 #几何维数
    chi_list           =  [10]                               #几何维数
    d                  =  2                                 #物理指标维数
    ca_center          =  n_site//2                         #正则中心位置,默认为MPS链中心
    Jx                 =  1
    Jy                 =  1
    Jz                 =  1
    h                  =  0
    ham1, ham2         =  Hei_ham(Jx, Jy, Jz, h)
    MPS, lam           =  initialize(n_site, chi_list[0])
    inner_product_list =  []
    ee_list            =  []
    E_list             =  []

    filename          = f"n={n_site}_Jx={Jx:.2f}_Jy={Jy:.2f}_Jz={Jz:.2f}_h={h:.2f}_chimax={max(chi_list)}.txt" #运行结果存储文件名
    current_directory = os.path.dirname(os.path.abspath(__file__))
    target_folder     = os.path.join(current_directory, 'datas_out')
    os.makedirs(target_folder, exist_ok=True)                                          #如果不存在目标文件夹就创建它,也可以使用下面注释的内容
    # if not os.path.exists(target_folder):
    #     os.makedirs(target_folder)
    file_path         = os.path.join(target_folder, filename)
    #将文件初始化为一个空文档
    with open(file_path, 'w') as f:
        pass

    # with open(file_path, 'a') as f:
    #     for i in range(n_site):
    #         print(MPS[i],file=f)

    # for tt, tau in enumerate([1, 0.1, 0.05, 0.01, 5e-3, 1e-3, 1e-4, 1e-5, 0]):
    for tt, tau in enumerate([0.1, 0.1]):
        chi      = chi_list[min(tt, len(chi_list)-1)]
        MPS, lam = apply_gate(tau/2, chi, 50, ham1, ham2, MPS, lam)
        
    #测试要用sx, sz
    sx            = torch.tensor([[0, 1], [1, 0]])/2
    sz            = torch.tensor([[1, 0], [0,-1]])/2
    MPS2, ld      = canonicalize(n_site, ca_center, MPS, lam)                                   #中心正则化
    ee_spectrum   = -2*(ld[ld>=1e-45]**2 )*torch.log(ld[ld>=1e-45])                             #中心纠缠谱,小于1e-45的数计算会NaN
    ee            = torch.sum(ee_spectrum).item()                                               #中心纠缠熵
    l_check       = check_canonicalization(MPS2, ld, ca_center, n_site)                         #正则化不满足标准的点及其误差列表
    
    mz1           = oe.contract('ilj,j,jnk,lm,imo,o,onk->',MPS2[ca_center-1],ld, MPS2[ca_center],sz,MPS2[ca_center-1],ld, MPS2[ca_center]).item()
    mx1           = oe.contract('ilj,j,jnk,lm,imo,o,onk->',MPS2[ca_center-1],ld, MPS2[ca_center],sx,MPS2[ca_center-1],ld, MPS2[ca_center]).item()
    mz2           = oe.contract('ilj,j,jmk,mn,ilo,o,onk->',MPS2[ca_center-1],ld, MPS2[ca_center],sz,MPS2[ca_center-1],ld, MPS2[ca_center]).item()
    mx2           = oe.contract('ilj,j,jmk,mn,ilo,o,onk->',MPS2[ca_center-1],ld, MPS2[ca_center],sx,MPS2[ca_center-1],ld, MPS2[ca_center]).item()
    smz           = mz1+mz2
    ms1           = np.sqrt(mz1**2+mx1**2)
    ms2           = np.sqrt(mz2**2+mx2**2)
    with open(file_path, 'a') as f:#a 是追加写入， w是覆盖写入
            print("Finally smz=",smz," ms1=",ms1," ms2=",ms2," Se=", ee," Check_result=", l_check, file= f)

    # print("|ψ|^2=  ", inner_product_list, "\n", "entangle entropy = ", ee_list,  "\n", "E_gs = ", E_list)
    
    #将结果写入文件
    # with open(file_path, 'w') as f:
    #     f.write()
        
    