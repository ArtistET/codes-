import torch
import numpy as np
import opt_einsum as oe
import matplotlib.pyplot as plt
from torch import linalg as LA
from Hei_1D_OBC_real import Hei_ham, build_gate
import os
from argparse import ArgumentParser
torch.set_default_dtype(torch.float64)#默认数据类型修改为更高精度
#'-h','-help' 不能用,会和帮助命令冲突, 如果要设默认值,则type必须与默认值类型一致且前面必须加'-'
parser = ArgumentParser()
parser.add_argument('-n', type=int, default=50)            #格点总数
parser.add_argument('-chi', type=int, default=10)          #几何维数最大值
parser.add_argument('-Jx', type=float, default=1)          
parser.add_argument('-Jy', type=float, default=0)           
parser.add_argument('-Jz', type=float, default=0)          
parser.add_argument('-fieldh', type=float, default=1)      
parser.add_argument('-ca_center', type=int, default=None)  #正则中心,在主函数开头初始化,默认为n//2,它位于MPS[ca_center-1]和MPS[ca_center]之间
parser.add_argument('-d', type=int, default=2)             #物理维数
parser.add_argument('-tau', type=float, default=0.1)       #虚时演化步长
parser.add_argument('-ntimes', type=int, default=10)       #一个MPO等于演化多少次
parser.add_argument('-MPO_time', type=int, default=10)      #MPO作用次数,ntimes*MPO_time=gate总演化次数
parser.add_argument('-tsample', type=int, default=100)     #采样次数

class TEBD_MC:
    def __init__(self, n, chi, Jx, Jy, Jz, h, ham1, ham2, gate1, gate2, ca_center, d, ntimes, MPO_time, tsample):
        self.n          = n
        self.chi        = chi
        self.d          = d
        self.ham1       = ham1
        self.ham2       = ham2
        self.gate1      = gate1
        self.gate2      = gate2
        self.ca_center  = ca_center
        self.Jx         = Jx
        self.Jy         = Jy
        self.Jz         = Jz
        self.h          = h
        self.MPS_shape  = [1, d, 1]
        self.MPO_shape  = [1]+[d]*2+[1]
        self.ntimes     = ntimes
        self.MPO_time   = MPO_time
        self.tsample    = tsample
        self.list_E     = []
        self.E_p        = 0
        self.tot_Ew     = 0
        self.totw       = 0     
        self.indices    = []     #指标集,表示当前product state中每个点处不为0的指标,在这里,|0>对应|↑>,|1>对应|↓>
        # self.leftw      = [torch.tensor([1]).reshape(1,1)]*n     #左侧权重张量,leftw[i]表示第i个点左侧的权重张量
        # self.rightw     = [torch.tensor([1]).reshape(1,1)]*n     #右侧权重张量,rightw[i]表示第n-1-i个点右侧的权重张量
        self.wtensor    = [[torch.tensor([1.]).reshape(1,1)]*n, [torch.tensor([1.]).reshape(1,1)]*n]  #wtensor[0]=leftw, wtensor[1]=rightw
        self.initialize()

    
    def initialize(self): #初始化MPS和MPO
        self.setMPS()
        
        self.MPO = []
        for i in range(self.n):
            self.MPO.append(torch.eye(self.d,self.d).reshape(self.MPO_shape))
        
        self.lamO = torch.tensor([1])
        # print(self.MPO[0], self.MPO[0].shape, len(self.MPO))
        # print(self.lamS, self.lamO)
        self.form_MPO_1by1()
    
    def recover_MPS(self):#根据指标集将MPS恢复到product state
        self.MPS=[]
        for i in range(self.n):
            a = torch.zeros(self.d)
            # a=torch.randn(self.d)
            a[self.indices[i]] = 1
            self.MPS.append(a.reshape(self.MPS_shape))

    def setMPS(self): #初始化MPS
        self.MPS = []
        for i in range(self.n):
            # a=torch.randn(self.d)
            a         = torch.zeros(self.d)
            rd_idx    = np.random.randint(0, self.d)
            a[rd_idx] = 1
            self.indices.append(rd_idx)
            self.MPS.append(a.reshape(self.MPS_shape))
        
        self.lamS = torch.tensor([1])
        # print(self.MPS[0], self.indices[0])
        # print(self.MPS[1], self.indices[1])

    def reform_gate(self):#把gate1, gate2 重组为单点的gate, 之后要作用在MPO上
        self.ou      =[]                                          #作用在MPO上方(除了两端)
        self.od      =[]                                          #作用在MPO下方(除了两端)
        res          = self.gate2.permute(0,2,1,3).reshape(self.d**2,self.d**2)
        # q, r         = LA.qr(res)
        U,s,Vh       = LA.svd(res)
        q            = U
        r            = oe.contract('i,ij->ij',s,Vh)
        c            = q.size(1)
        l1           = q.reshape(self.d,self.d,c)
        r1           = r.reshape(c,self.d,self.d)
        self.left_u  = oe.contract('ijk,jl->ilk', l1, self.gate1) #作用于MPO[0]上方
        self.right_u = oe.contract('ijk,kl->ijl', r1, self.gate1) #作用于MPO[n-1]上方
        self.left_d  = oe.contract('ij,jkl->ikl', self.gate1, l1) #作用于MPO[0]下方
        self.right_d = oe.contract('ij,kjl->kil', self.gate1, r1) #作用于MPO[n-1]下方
        self.ou.append(oe.contract('ijk,klm->ijlm', r1, self.left_u))
        self.ou.append(oe.contract('ijk,ljm->limk', l1, self.right_u))
        self.od.append(oe.contract('ijk,ljm->limk', self.left_d, r1))
        self.od.append(oe.contract('ijk,klm->ijlm', self.right_d, l1))
        


    def form_MPO(self):#构建MPO, 这种方法更加高效,不再是ntimes*tau而是tau*(4^ntimes)
        self.reform_gate()
        #第一个来回,重组之后的gate作用在MPO上
        c                  = self.left_d.size(2)
        res                = oe.contract('ijk,ljmn,mop->lioknp', self.left_u, self.MPO[0], self.left_d)
        self.MPO[0]        = res.reshape(1,d,d,c**2)
        res                = oe.contract('ijk,lkmn,omp->ilojpn', self.right_u, self.MPO[self.n-1], self.right_d)
        self.MPO[self.n-1] = res.reshape(c**2,d,d,1)
        for i in range(1,self.n-1):
            res            = oe.contract('ijkl,mkno,pnqr->impjqlor', self.ou[i%2], self.MPO[i], self.od[i%2])
            self.MPO[i]    = res.reshape(c**2,d,d,c**2)
        for i in range (self.n-1,0,-1):#正则化MPO
            c1             = self.MPO[i-1].size(0)
            c3             = self.MPO[i].size(3)
            res            = oe.contract('ijkl,lmno,o->ijkmno', self.MPO[i-1], self.MPO[i], self.lamO).reshape(c1*d*d,d*d*c3)
            U, s, Vh       = LA.svd(res)
            c2             = min(len(s), self.chi)
            self.MPO[i-1]  = U[:, :c2].reshape(c1,d,d,c2)
            self.lamO      = s[:c2]/LA.norm(s[:c2])
            self.MPO[i]    = Vh[:c2, :].reshape(c2,d,d,c3)

        #之后ntimes个来回, MPO和自己作用
        for j in range(self.ntimes):
            #从左到右sweep
            c3             = self.MPO[1].size(3)
            res            = oe.contract('kinl,l,ljom,psnq,q,qtor->kpisjtmr', self.MPO[0], self.lamO, self.MPO[1], self.MPO[0], self.lamO, self.MPO[1]).reshape(d**2,d*d*c3*c3)  
            U, s, Vh       = LA.svd(res)
            c2             = min(self.chi, len(s))
            self.MPO[0]    = U[:,:c2].reshape(1,d,d,c2)
            self.lamO      = s[:c2]/LA.norm(s[:c2])
            self.MPO[1]    = Vh[:c2,:].reshape(c2,d,d,c3**2)
            for i in range (2,self.n):
                c0            = self.MPO[i].size(0)
                c1            = self.MPO[i-1].size(0)
                c3            = self.MPO[i].size(3)
                NO            = oe.contract('ijkl,mnko->imjnlo', self.MPO[i], self.MPO[i]).reshape(c0**2,d,d,c3**2)
                res           = oe.contract('i,ijkl,lmno->ijkmno', self.lamO, self.MPO[i-1], NO).reshape(c1*d*d,d*d*c3*c3)
                U, s, Vh      = LA.svd(res)
                c2            = min(self.chi, len(s))
                self.MPO[i-1] = U[:,:c2].reshape(c1,d,d,c2)
                self.lamO     = s[:c2]/LA.norm(s[:c2])
                self.MPO[i]   = Vh[:c2,:].reshape(c2,d,d,c3**2)
            #从右到左sweep
            c1                 = self.MPO[self.n-2].size(0)
            res                = oe.contract('kinl,l,ljom,psnq,q,qtor->kpisjtmr', self.MPO[self.n-2], self.lamO, self.MPO[self.n-1], self.MPO[self.n-2], self.lamO, self.MPO[self.n-1]).reshape(c1*c1*d*d,d**2)
            U, s, Vh           = LA.svd(res)
            c2                 = min(self.chi, len(s))
            self.MPO[self.n-2] = U[:,:c2].reshape(c1**2,d,d,c2)
            self.lamO      = s[:c2]/LA.norm(s[:c2])
            self.MPO[1]    = Vh[:c2,:].reshape(c2,d,d,1)
            for i in range(self.n-2, 0, -1):
                c0            = self.MPO[i-1].size(3)
                c1            = self.MPO[i-1].size(0)
                c3            = self.MPO[i].size(3)
                NO            = oe.contract('ijkl,mnko->imjnlo', self.MPO[i-1], self.MPO[i-1]).reshape(c1**2,d,d,c0**2)
                res           = oe.contract('ijkl,lmno,o->ijkmno', NO, self.MPO[i], self.lamO).reshape(c1*c1*d*d,d*d*c3)
                U, s, Vh      = LA.svd(res)
                c2            = min(self.chi, len(s))
                self.MPO[i-1] = U[:,:c2].reshape(c1**2,d,d,c2)
                self.lamO     = s[:c2]/LA.norm(s[:c2])
                self.MPO[i]   = Vh[:c2,:].reshape(c2,d,d,c3)



    def form_MPO_1by1(self): #用两种gate构建MPO
        for i in range(self.ntimes):
            # if (i%20==0 or i==self.ntimes-1):
            #     print("ntime_number: ", i)
            #     self.apply_MPO()
            #     E = self.ave_mpo(self.ham1, self.ham2)
            #     self.list_E.append(E)
            #     # self.recover_MPS()
            #     with open(file_path, 'a') as f:
            #         print("now= ", i,"  E= ", E, file=f)
            #         print(self.MPO[0])
            #     # self.recover_MPS()


            #从左到右sweep
            c3          = self.MPO[1].size(3)
            res         = oe.contract('ik,jl,klmn,omrp,p,pnsq->oirjsq', self.gate1, self.gate1, self.gate2, self.MPO[0], self.lamO, self.MPO[1])
            U, s, Vh    = LA.svd(res.reshape(d**2,d*d*c3))
            c2          = min(self.chi, len(s))
            self.MPO[0] = U[:, :c2].reshape(1,d,d,c2)
            self.lamO   = s[:c2]/LA.norm(s[:c2])
            self.MPO[1] = Vh[:c2,:].reshape(c2,d,d,c3)
            # print(self.MPO[0].shape, self.lamO.shape, self.MPO[1].shape)
            # print(self.MPO[0], self.lamO, self.MPO[1])
            for i in range(1, self.n-1):
                c1            = self.MPO[i].size(0)
                c3            = self.MPO[i+1].size(3)
                res           = oe.contract('jk,iklm,n,nlqo,omrp->niqjrp', self.gate1, self.gate2, self.lamO, self.MPO[i], self.MPO[i+1]).reshape(c1*d*d,d*d*c3)
                U, s, Vh      = LA.svd(res)
                c2            = min(self.chi,len(s))
                self.MPO[i]   = U[:, :c2].reshape(c1,d,d,c2)
                self.lamO     = s[:c2]/LA.norm(s[:c2])
                self.MPO[i+1] = Vh[:c2,:].reshape(c2,d,d,c3)
            
            #从右到左sweep
            c1           = self.MPO[-2].size(0)
            res          = oe.contract('ik,jl,klmn,omrp,p,pnsq->oirjsq', self.gate1, self.gate1, self.gate2, self.MPO[-2], self.lamO, self.MPO[-1])
            U, s, Vh     = LA.svd(res.reshape(c1*d*d,d**2))
            c2           = min(self.chi, len(s))
            self.MPO[-2] = U[:, :c2].reshape(c1,d,d,c2)
            self.lamO    = s[:c2]/LA.norm(s[:c2])
            self.MPO[-1] = Vh[:c2,:].reshape(c2,d,d,1)
            for i in range(self.n-2, 0, -1):
                c1            = self.MPO[i-1].size(0)
                c3            = self.MPO[i].size(3)
                res           = oe.contract('ij,jklm,nlqo,omrp,p->niqkrp', self.gate1, self.gate2, self.MPO[i-1], self.MPO[i], self.lamO).reshape(c1*d*d,d*d*c3)
                U, s, Vh      = LA.svd(res)
                c2            = min(self.chi,len(s))
                self.MPO[i-1] = U[:, :c2].reshape(c1,d,d,c2)
                self.lamO     = s[:c2]/LA.norm(s[:c2])
                self.MPO[i]   = Vh[:c2,:].reshape(c2,d,d,c3)

            
    def apply_MPO(self):  #MPO作用于MPS(product state)上
        # c1          = self.MPO[0].size(0)   #其实不需要c1,因为两端的几何维数由于开边界条件都是1,无论MPS还是MPO
        c3          = self.MPO[1].size(3)
        c4          = self.MPS[1].size(2)
        res         = oe.contract('kinl,l,ljom,pnq,q,qor->kpijmr', self.MPO[0], self.lamO, self.MPO[1], self.MPS[0], self.lamS, self.MPS[1]).reshape(d, d*c3*c4)
        U, s, Vh    = LA.svd(res)
        c2          = min(self.chi, len(s))
        self.MPS[0] = U[:, :c2].reshape(1,d,c2)
        self.lamS   = s[:c2]/LA.norm(s[:c2])
        self.MPS[1] = Vh[:c2,:].reshape(c2,d,c3*c4)
        for i in range(2,self.n):
            c1            = self.MPO[i].size(0)
            c2            = self.MPS[i].size(0)
            c3            = self.MPO[i].size(3)
            c4            = self.MPS[i].size(2)
            NS            = oe.contract('ijlk,mln->imjkn', self.MPO[i],self.MPS[i]).reshape(c1*c2,d,c3*c4)
            c0            = self.MPS[i-1].size(0)
            res           = oe.contract('i,ijk,klm->ijlm',self.lamS,self.MPS[i-1],NS).reshape(c0*d,d*c3*c4)
            U, s, Vh      = LA.svd(res)
            c5            = min(self.chi, len(s))
            self.MPS[i-1] = U[:, :c5].reshape(c0,d,c5)
            self.lamS     = s[:c5]/LA.norm(s[:c5])
            self.MPS[i]   = Vh[:c5,:].reshape(c5,d,c3*c4)
        self.pull_back()
        

    
    def pull_back(self):  #把正则中心拉回到至 1(0~1之间)
        #n-1单独算,因为有lamS且维数最小
        c1                 = len(self.lamS)
        res                = oe.contract('i,ijk->ijk', self.lamS, self.MPS[self.n-1]).reshape(c1,d)
        Q, R               = LA.qr(res.T)
        c2                 = Q.size(1)
        self.MPS[self.n-1] = Q.T.reshape(c2,d,1)
        for i in range(self.n-2,1,-1):#n-2~2
            c1          = self.MPS[i].size(0)
            c3          = R.size(0)
            res         = oe.contract('ijk,lk->ijl', self.MPS[i], R).reshape(c1,d*c3)
            Q, R        = LA.qr(res.T)
            c2          = Q.size(1)
            self.MPS[i] = Q.T.reshape(c2,d,c3)

        #0~1单独算,因为要给出新的lamS
        c3          = R.size(0)
        res         = oe.contract('ijk,klm,nm->ijln', self.MPS[0], self.MPS[1], R).reshape(d,d*c3)
        U, s, Vh    = LA.svd(res)
        c2          = min(self.chi, len(s))
        self.MPS[0] = U[:, :c2].reshape(1,d,c2)
        self.lamS   = s[:c2]/LA.norm(s[:c2])
        self.MPS[1] = Vh[:c2,:].reshape(c2,d,c3)


    def update(self):
        for i in range(self.MPO_time):
            # print(i)
            self.apply_MPO()
            # E = self.ave_mpo(self.ham1, self.ham2)
            # self.list_E.append(E)
            # with open(file_path, 'a') as f:
            #     print("ntimes= ", self.ntimes," E= ", E, file=f)
            self.do_MC(self.MPO_time-i)
            self.recover_MPS()



    def do_MC(self, mode):
        for i in range(1, self.n): #第一次sweep从左到右,因此先初始化所有右权重张量
            # self.wtensor[1][i] = oe.contract('ij,jk->ik',self.MPS[self.n-i][:,self.indices[self.n-i],:],self.wtensor[1][i-1])
            self.updatew(1,i,0)
        w0 = (oe.contract('ij,jk->ik',self.MPS[0][:,self.indices[0],:],self.wtensor[1][self.n-1]).item())**2
        if(mode == 1):
            for i in range(self.n-1):
                self.E_p += self.ham2[self.indices[i],self.indices[i+1],self.indices[i],self.indices[i+1]].item()
                self.E_p += self.ham1[self.indices[i],self.indices[i]].item()
            self.E_p     += self.ham1[self.indices[self.n-1],self.indices[self.n-1]].item()
            self.tot_Ew  += w0*self.E_p
            self.totw    += w0
        for i in range(self.tsample): #tsample次随机行走
            sweepd = (i//self.n)%2 #sweep方向,0为从左到右,1为从右到左
            sweepn = i%self.n      #本次sweep在第几个点(会受到方向影响)
            if sweepd == 0:        #当前在哪个点
                num = sweepn
                w   = (oe.contract('ij,jk,kl->il', self.wtensor[0][num], self.MPS[num][:,(self.indices[num]+1)%2,:],self.wtensor[1][self.n-1-num]).item())**2
                # if(mode == 1):
                #     print(w,w0,w/w0)
                a   = torch.rand(1).item()
                if(a<w/w0):
                    print('!',w/w0,w,w0)
                    if mode==1:
                        
                        self.E_p          += (self.ham1[(self.indices[num]+1)%2][(self.indices[num]+1)%2]-self.ham1[self.indices[num]][self.indices[num]]).item()
                        if num>0 :
                            self.E_p      += (self.ham2[self.indices[num-1]][(self.indices[num]+1)%2][self.indices[num-1]][(self.indices[num]+1)%2]-self.ham2[self.indices[num-1]][self.indices[num]][self.indices[num]][self.indices[num]]).item()
                        if num<self.n-1 :
                            self.E_p      += (self.ham2[(self.indices[num]+1)%2][self.indices[num+1]][(self.indices[num]+1)%2][self.indices[num+1]]-self.ham2[self.indices[num]][self.indices[num+1]][self.indices[num]][self.indices[num+1]]).item()
                    self.indices[num] = (self.indices[num]+1)%2
                    w0                = w
                if sweepn<self.n-1:
                    self.updatew(0, sweepn+1, 1)

            else:
                num = self.n-1-sweepn
                w   = (oe.contract('ij,jk,kl->il', self.wtensor[0][num], self.MPS[num][:,(self.indices[num]+1)%2,:],self.wtensor[1][self.n-1-num]).item())**2
                a   = torch.rand(1).item()
                if(a<w/w0):
                    print('!',w/w0,w,w0)
                    if mode==1:
                        self.E_p          += (self.ham1[(self.indices[num]+1)%2][(self.indices[num]+1)%2]-self.ham1[self.indices[num]][self.indices[num]]).item()
                        if num>0 :
                            self.E_p      += (self.ham2[self.indices[num-1]][(self.indices[num]+1)%2][self.indices[num-1]][(self.indices[num]+1)%2]-self.ham2[self.indices[num-1]][self.indices[num]][self.indices[num]][self.indices[num]]).item()
                        if num<self.n-1 :
                            self.E_p      += (self.ham2[(self.indices[num]+1)%2][self.indices[num+1]][(self.indices[num]+1)%2][self.indices[num+1]]-self.ham2[self.indices[num]][self.indices[num+1]][self.indices[num]][self.indices[num+1]]).item()
                    self.indices[num] = (self.indices[num]+1)%2
                    w0                = w
                if sweepn<self.n-1:    
                    self.updatew(1, sweepn+1, 1)

            if mode==1:
                with open(file_path, 'a') as f:
                    print("Ep= ", self.E_p, file=f)
                self.tot_Ew       += w0*self.E_p
                self.totw         += w0
            

    def updatew(self, udtype, num, mode): #mode=0/1表示初始化/修改 udtype=0/1表示向右/向左 num就是do_MC中的sweepn
        if udtype==0 and num==1 :
            ind = (self.indices[num-1]+mode)%2
            self.wtensor[0][1] = oe.contract('ij,jk,k->ik',self.wtensor[0][0], self.MPS[0][:,ind,:], self.lamS)
        elif udtype==1 and num==self.n-1:
            ind = (self.indices[self.n-num]+mode)%2
            self.wtensor[1][num] = oe.contract('i,ij,jk->ik', self.lamS, self.MPS[1][:,ind,:],self.wtensor[1][num-1])
        else:
            if udtype==0:
                ind = (self.indices[num-1]+mode)%2
                self.wtensor[0][num] = oe.contract('ij,jk->ik',self.wtensor[0][num-1], self.MPS[num-1][:,ind,:])
            else:
                # print(self.n-num, num)
                ind = (self.indices[self.n-num]+mode)%2
                # print(self.MPS[self.n-num][:,ind,:].dtype, self.wtensor[1][num-1].dtype)
                self.wtensor[1][num] = oe.contract('ij,jk->ik', self.MPS[self.n-num][:,ind,:],self.wtensor[1][num-1])

    
    def ave_mpo(self,op1,op2): #算符平均值, op1为单体算符,如没有应该输入zeros(d,d), op2为最近邻两体算符,如没有应该输入zeros(d,d,d,d))
        ans=0
        ans += oe.contract('ilj,j,jmk,lmno,inp,p,pok->', self.MPS[0], self.lamS, self.MPS[1], op2, self.MPS[0], self.lamS, self.MPS[1]).item()  #两点平均值
        ans += oe.contract('ilj,j,jnk,lm,imo,o,onk->', self.MPS[0], self.lamS, self.MPS[1], op1, self.MPS[0], self.lamS, self.MPS[1]).item()    #左单点平均值
        ans += oe.contract('ilj,j,jmk,mn,ilo,o,onk->', self.MPS[0], self.lamS, self.MPS[1], op1, self.MPS[0], self.lamS, self.MPS[1]).item()    #右单点平均值
        ml   = self.MPS[1]
        mr   = self.MPS[2]
        ld   = self.lamS
        for i in range(1, self.n-1):
            c1       = ml.size(0)
            c3       = mr.size(2)
            res      = oe.contract('i,ijk,klm->ijlm', ld, ml, mr).reshape(c1*d,d*c3)
            U, s, Vh = LA.svd(res)
            c2       = min(self.chi, len(s))
            ml       = U[:, :c2].reshape(c1,d,c2)
            ld       = s[:c2]/LA.norm(s[:c2])
            mr       = Vh[:c2, :].reshape(c2,d,c3)
            #单点平均值只算右侧，因为左侧上一步算过了
            ans     += oe.contract('ilj,j,jmk,lmno,inp,p,pok->', ml, ld, mr, op2, ml, ld, mr).item()
            ans     += oe.contract('ilj,j,jmk,mn,ilo,o,onk->', ml, ld, mr, op1, ml, ld, mr).item()
            ml       = Vh[:c2, :].reshape(c2,d,c3)
            if i<self.n-2:
                mr = self.MPS[i+2]
        return ans

    def check_canonicalization(self):  #检查正则化是否正确, MPS和MPO都检查, 第一位都要单独考虑,因为除了它都是左正则
        judge_standard = 1e-14
        check_MPS = []
        check_MPS.append(("lamS", LA.norm(self.lamS).item()))
        check_MPO = []
        check_MPO.append(("lamO", LA.norm(self.lamO).item()))
        #先检验MPS
        c         = self.MPS[0].size(2)
        ans       = LA.norm(torch.eye(c,c)-oe.contract('ijk,ijl->kl', self.MPS[0], self.MPS[0])).item()
        if ans>judge_standard:
            check_MPS.append((0, ans))
        for i in range(1, self.n):
            # print(i)
            c     = self.MPS[i].size(0)
            ans   = LA.norm(torch.eye(c,c)-oe.contract('ikl,jkl->ij', self.MPS[i], self.MPS[i])).item()
            if ans>judge_standard: 
                check_MPS.append((i, ans))
        #再检验MPO
        c         = self.MPO[0].size(3)
        ans       = LA.norm(torch.eye(c,c)-oe.contract('ijkl,ijkm->lm', self.MPO[0], self.MPO[0])).item()
        if ans>judge_standard:
            check_MPO.append((0, ans))
        for i in range(1, self.n):
            c     = self.MPO[i].size(0)
            ans   = LA.norm(torch.eye(c,c)-oe.contract('iklm,jklm->ij', self.MPO[i], self.MPO[i])).item()
        with open(file_path, 'a') as f:
                print("checkS= ", check_MPS, file=f)
                print("checkO= ", check_MPO, file=f)




if __name__ =="__main__":
    args           = parser.parse_args()
    if args.ca_center is None:
        args.ca_center = args.n//2
    d              = args.d
    # print(args)
    ham1, ham2     = Hei_ham(Jx = args.Jx, Jy = args.Jy, Jz = args.Jz, h=args.fieldh)
    gate1, gate2   = build_gate(tau=args.tau/2, ham1=ham1, ham2=ham2)
    # TEBD_MC_solver = TEBD_MC(n=args.n, chi=args.chi, Jx=args.Jx, Jy=args.Jy, Jz=args.Jz, h=args.fieldh, ham1=ham1, ham2=ham2, gate1=gate1, gate2=gate2, ca_center=args.ca_center, d=args.d, ntimes=args.ntimes, MPO_time=args.MPO_time, tsample=args.tsample)
    # TEBD_MC_solver.form_MPO()
    # TEBD_MC_solver.apply_MPO()
    # TEBD_MC_solver.initialize()
    filename          = f"MC_Jx={args.Jx:.2f}_Jy={args.Jy:.2f}_Jz={args.Jz:.2f}_h={args.fieldh:.2f}_chi={args.chi}.txt" #运行结果存储文件名
    current_directory = os.path.dirname(os.path.abspath(__file__))
    target_folder     = os.path.join(current_directory, 'datas_out')
    os.makedirs(target_folder, exist_ok=True)                                          #如果不存在目标文件夹就创建它,也可以使用下面注释的内容
    file_path         = os.path.join(target_folder, filename)
    #将文件初始化为一个空文档
    with open(file_path, 'w') as f:
        pass
    # for nt in [10,20,30,40,50,60,70,80,90,100]:
    # for nt in [400]:
        TEBD_MC_solver = TEBD_MC(n=args.n, chi=args.chi, Jx=args.Jx, Jy=args.Jy, Jz=args.Jz, h=args.fieldh, ham1=ham1, ham2=ham2, gate1=gate1, gate2=gate2, ca_center=args.ca_center, d=args.d, ntimes=args.ntimes, MPO_time=args.MPO_time, tsample=args.tsample)
        TEBD_MC_solver.update()
        ans = TEBD_MC_solver.tot_Ew/TEBD_MC_solver.totw
        # print(TEBD_MC_solver.tot_Ew, TEBD_MC_solver.totw)
        print(ans)
        # TEBD_MC_solver.check_canonicalization()
    # plt.plot(list(range(TEBD_MC_solver.MPO_time)),TEBD_MC_solver.list_E)
    # plt.show()
    # print(gate1, gate2)
        
    