#一维链自旋模型Worldline Monte Carlo求基态以及基态能(有限长开边界,只有local update)
import torch
import numpy as np
import opt_einsum as oe
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats as stats
from torch import linalg as LA
import seaborn as sns
import os
from argparse import ArgumentParser
torch.set_default_dtype(torch.float64)#默认数据类型修改为更高精度
#'-h','-help' 不能用,会和帮助命令冲突, 如果要设默认值,则type必须与默认值类型一致且前面必须加'-'
parser = ArgumentParser()
parser.add_argument('-n', type=int, default=20)                #格点总数
parser.add_argument('-Jx', type=float, default=1)          
parser.add_argument('-Jy', type=float, default=1)           
parser.add_argument('-Jz', type=float, default=1)          
parser.add_argument('-fieldh', type=float, default=0)      
parser.add_argument('-tau', type=float, default=0.1)           #虚时演化步长
parser.add_argument('-dtau', type=float,default=1e-6)          #虚时演化步长变化率,用于计算能量时的求导
parser.add_argument('-ntimes', type=int, default=100)          #演化多少次
parser.add_argument('-heatn', type=int, default=300000)        #热化消耗的次数
parser.add_argument('-tsample', type=int, default=30000000)     #采样次数
parser.add_argument('-step_gap', type=float, default=100)         #每隔若干步采样一次以保证样本之间不关联
parser.add_argument('-w', type=float, default=[1,1,1])         #分别为β-dtau, β, β+dtau处的权重,为了之后算能量时进行微分计算使用
class WLQMC:
    def __init__(self, n, Jx, Jy, Jz, fieldh, tau, dtau, ntimes, tsample, step_gap, heatn, w):
        self.n       = n
        self.m       = ntimes*2
        self.Jx      = Jx
        self.Jy      = Jy
        self.Jz      = Jz
        self.fieldh  = fieldh
        self.tau     = tau
        self.dtau    = dtau
        self.ntimes  = ntimes
        self.tsample = tsample
        self.w       = w
        self.lE      = []    #每1000步的能量总和
        self.lw      = []    #每1000步的取样个数
        self.lcheck  = []    #用于检测E分布是否为正态分布
        self.E_ave   = 0
        self.w_ave   = 0
        self.heatn   = heatn
        self.step_gap= step_gap    #每隔若干步采样一次以保证样本之间不关联
        self.initialize_plaquettes()
        self.initialize_config()
        # print(self.config)
        # print(self.getID(self.m-2,self.n-2))
    
    def initialize_config(self): #初始构型：每行相同且为一个长度为n的0/1随机列表,也就是每个点对应一条直的世界线或是没有世界线
        self.config = []
        # sumdot=0
        l           = np.random.randint(0,2,self.n)
        l           = [1,0]*10 #相同初态测试WLQMC稳定性
        for i in range(self.m+1): #m层gate对应m+1行格点
            self.config.append(l.copy())
        gmode       = 0   #gmode+2k~gmode+2k+1为一组gate, gmode 根据行数奇偶切换
        self.plaq   = np.zeros((self.m,self.n-1),dtype=int)
        # self.wconfig= [np.zeros((self.m,self.n-1))]*3
        #不能像上面一样初始化！会导致所有矩阵被同步修改然后修2个礼拜的bug！！！！
        self.wconfig= [np.zeros((self.m,self.n-1)) for _ in range(3)]
        # print(self.wconfig)
        for i in range(self.m):#初始化plaquette
            for j in range(self.n-1):
                self.plaq[i][j] = self.getID(i,j)
                if  (j+gmode)%2 == 0:#对有gate的plaquette赋值并且计入对总权重的贡献     
                    # sumdot+=1
                    for k in range(3):
                        self.w[k]*=self.ptype[k][self.plaq[i][j]]
                        self.wconfig[k][i][j] = self.ptype[k][self.plaq[i][j]]
                        # if(self.ptype[k][self.getID(i,j)]==0):
                        #     print((i,j,self.getID(i,j),gmode,self.config[i][j],self.config[i][j+1],self.config[i+1][j],self.config[i+1][j+1]))        
            # print(self.plaq[i])
            gmode   = (gmode+1)%2  #这里第i行考虑的是gate的第i行
        # print(self.w)
        self.sumE   = 0
        self.sumE2  = 0
        self.sumw   = 0
        # self.sumE   = np.log(self.w[0]/self.w[2])/(self.m*self.dtau)
        # self.sumw   = 1
        # print(self.sumE, self.sumw, self.sumE/self.sumw)
    
    def initialize_plaquettes(self): #初始化可能用到的plaquettes
        self.ptype        = [[0]*16,[0]*16,[0]*16]#代入不同tau,计算不同plaquette代表的权重
        #空(0)
        self.ptype[0][0]  = np.exp(-(self.tau-self.dtau)*self.Jz*0.25)*np.exp((self.tau-self.dtau)*self.fieldh)
        self.ptype[1][0]  = np.exp(-(self.tau)*self.Jz*0.25)*np.exp((self.tau)*self.fieldh)
        self.ptype[2][0]  = np.exp(-(self.tau+self.dtau)*self.Jz*0.25)*np.exp((self.tau+self.dtau)*self.fieldh)
        
        #两侧(15)
        self.ptype[0][15] = np.exp(-(self.tau-self.dtau)*self.Jz*0.25)*np.exp(-(self.tau-self.dtau)*self.fieldh)
        self.ptype[1][15] = np.exp(-(self.tau)*self.Jz*0.25)*np.exp(-(self.tau)*self.fieldh)
        self.ptype[2][15] = np.exp(-(self.tau+self.dtau)*self.Jz*0.25)*np.exp(-(self.tau+self.dtau)*self.fieldh)
        
        #左(5)
        self.ptype[0][5]  = np.exp((self.tau-self.dtau)*self.Jz*0.25)*np.cosh((self.tau-self.dtau)*(self.Jx+self.Jy)*0.25)
        self.ptype[1][5]  = np.exp((self.tau)*self.Jz*0.25)*np.cosh((self.tau)*(self.Jx+self.Jy)*0.25)
        self.ptype[2][5]  = np.exp((self.tau+self.dtau)*self.Jz*0.25)*np.cosh((self.tau+self.dtau)*(self.Jx+self.Jy)*0.25)
        
        #右(10)
        self.ptype[0][10] = np.exp((self.tau-self.dtau)*self.Jz*0.25)*np.cosh((self.tau-self.dtau)*(self.Jx+self.Jy)*0.25)
        self.ptype[1][10] = np.exp((self.tau)*self.Jz*0.25)*np.cosh((self.tau)*(self.Jx+self.Jy)*0.25)
        self.ptype[2][10] = np.exp((self.tau+self.dtau)*self.Jz*0.25)*np.cosh((self.tau+self.dtau)*(self.Jx+self.Jy)*0.25)
        
        #主对角线(6)
        self.ptype[0][6]  = -np.exp(-(self.tau-self.dtau)*self.Jz*0.25)*np.sinh((self.tau-self.dtau)*(self.Jx+self.Jy)*0.25)
        self.ptype[1][6]  = -np.exp(-(self.tau)*self.Jz*0.25)*np.sinh((self.tau)*(self.Jx+self.Jy)*0.25)
        self.ptype[2][6]  = -np.exp(-(self.tau+self.dtau)*self.Jz*0.25)*np.sinh((self.tau+self.dtau)*(self.Jx+self.Jy)*0.25)
        
        #副对角线(9)
        self.ptype[0][9]  = -np.exp(-(self.tau-self.dtau)*self.Jz*0.25)*np.sinh((self.tau-self.dtau)*(self.Jx+self.Jy)*0.25)
        self.ptype[1][9]  = -np.exp(-(self.tau)*self.Jz*0.25)*np.sinh((self.tau)*(self.Jx+self.Jy)*0.25)
        self.ptype[2][9]  = -np.exp(-(self.tau+self.dtau)*self.Jz*0.25)*np.sinh((self.tau+self.dtau)*(self.Jx+self.Jy)*0.25)
        # print(self.ptype[0])
        # print(self.ptype[1])
        # print(self.ptype[2])
    
    def getID(self, i,j): #返回坐标(i,j)对应的plaquette类型
        return self.config[i][j]+2*self.config[i][j+1]+4*self.config[i+1][j]+8*self.config[i+1][j+1]
    
    def local_update(self, nt): #一次local update(i,j为没有gate的plaquette的坐标,权重及其变化由其四周的plaquette贡献)
        wo  = [1,1,1] #尝试更新前的权重
        wn  = [1,1,1] #尝试更新后的权重
        i,j =self.find_possible()
        #plaq[i][j]四周的权重乘积
        if i>0:
            wo[0]*=self.wconfig[0][i-1][j]
            wo[1]*=self.wconfig[1][i-1][j]
            wo[2]*=self.wconfig[2][i-1][j]
        if i<self.m-1:
            wo[0]*=self.wconfig[0][i+1][j]
            wo[1]*=self.wconfig[1][i+1][j]
            wo[2]*=self.wconfig[2][i+1][j]
        if j>0:
            wo[0]*=self.wconfig[0][i][j-1]
            wo[1]*=self.wconfig[1][i][j-1]
            wo[2]*=self.wconfig[2][i][j-1]
        if j<self.n-2:
            wo[0]*=self.wconfig[0][i][j+1]
            wo[1]*=self.wconfig[1][i][j+1]
            wo[2]*=self.wconfig[2][i][j+1]
        
        jd = np.random.random()
        # with open(file_path, 'a') as f:
            # print(self.plaq, file=f)
            # print(i,j, self.plaq[i][j], file=f)

        if self.plaq[i][j]==5: #将这个plaquette左边的世界线移动到右边,引发周围四个权重变化
            if j>0:
                # print(self.plaq[i][j-1])
                wn[0]*=self.ptype[0][self.plaq[i][j-1]-10]
                wn[1]*=self.ptype[1][self.plaq[i][j-1]-10]
                wn[2]*=self.ptype[2][self.plaq[i][j-1]-10]
            if j<self.n-2:
                # print(self.plaq[i][j+1])
                wn[0]*=self.ptype[0][self.plaq[i][j+1]+5]
                wn[1]*=self.ptype[1][self.plaq[i][j+1]+5]
                wn[2]*=self.ptype[2][self.plaq[i][j+1]+5]
            if i>0:
                # print(self.plaq[i-1][j])
                wn[0]*=self.ptype[0][self.plaq[i-1][j]+4]
                wn[1]*=self.ptype[1][self.plaq[i-1][j]+4]
                wn[2]*=self.ptype[2][self.plaq[i-1][j]+4]
            if i<self.m-1:
                # print(self.plaq[i+1][j])
                wn[0]*=self.ptype[0][self.plaq[i+1][j]+1]
                wn[1]*=self.ptype[1][self.plaq[i+1][j]+1]
                wn[2]*=self.ptype[2][self.plaq[i+1][j]+1]

            # if wn[1]/wo[1]>1:
            #         with open(file_path, 'a') as f:
            #             print("now is", nt,"p= ", wn[1]/wo[1]," w_new= ",self.w[1]*wn[1]/wo[1]," w_old= ",self.w[1], file=f)
            if jd<(wn[1]/wo[1]):
                # if wn[1]/wo[1]>1:
                # with open(file_path, 'a') as f:
                    # print(self.w, file=f)
                    # print("now is", nt,"p= ", wn[1]/wo[1]," w_new= ",self.w[1]*wn[1]/wo[1]," w_old= ",self.w[1], file=f)
                    # print(wn[1],'  ',wo[1],file=f)
                    # print("wo_should_be ", self.ptype[1][self.plaq[i-1][j]]*self.ptype[1][self.plaq[i+1][j]]*self.ptype[1][self.plaq[i][j-1]]*self.ptype[1][self.plaq[i][j+1]], file=f)
                    # print("old_check=  ",self.checkw(), file=f)
                    # print(self.plaq, file=f)
                    # print(i,j, self.plaq[i][j], file=f)
                    # print("  ",self.plaq[i+1][j],file=f)
                    # if j>0:
                    #     print(self.plaq[i][j-1],end=' ',file=f)
                    # print(self.plaq[i][j],end=' ',file=f)
                    # if j<self.n-2:
                    #     print(self.plaq[i][j+1],file=f)
                    # else:
                    #     print(file=f)
                    # print("  ",self.plaq[i-1][j],file=f)
                    # print("updated", file=f)
                self.plaq[i][j]=10
                for k in range(3):
                    self.w[k]*= wn[k]/wo[k]
                if j>0:
                    self.plaq[i][j-1]      -= 10
                    self.plaq[i-1][j-1]    -= 8
                    self.plaq[i+1][j-1]    -= 2
                    for k in range(3):
                        self.wconfig[k][i][j-1] = self.ptype[k][self.plaq[i][j-1]]
                if j<self.n-2:
                    self.plaq[i][j+1]      += 5
                    self.plaq[i-1][j+1]    += 4
                    self.plaq[i+1][j+1]    += 1
                    for k in range(3):    
                        self.wconfig[k][i][j+1] = self.ptype[k][self.plaq[i][j+1]]
                if i>0:
                    self.plaq[i-1][j]      += 4
                    for k in range(3):
                        self.wconfig[k][i-1][j] = self.ptype[k][self.plaq[i-1][j]]
                if i<self.m-1:
                    self.plaq[i+1][j]      += 1
                    for k in range(3):
                        self.wconfig[k][i+1][j] = self.ptype[k][self.plaq[i+1][j]]
                # if self.w[1]<1e-8:
                #     print(self.w[1],self.checkw())
                #     self.plot_figure()
                # with open(file_path, 'a') as f:
                #     print("check_new= ",self.checkw(), file=f)
                #     print(wn[1]/wo[1],self.w, file=f)
                    # print("now number: ",nt," now dE= ",(np.log(self.w[0]/self.w[2])/(self.m*self.dtau))," now_w= ", self.w, file=f)

        elif self.plaq[i][j]==10: #将这个plaquette右边的世界线移动到左边,引发周围四个权重变化
            if j>0:
                wn[0]*=self.ptype[0][self.plaq[i][j-1]+10]
                wn[1]*=self.ptype[1][self.plaq[i][j-1]+10]
                wn[2]*=self.ptype[2][self.plaq[i][j-1]+10]
            if j<self.n-2:
                wn[0]*=self.ptype[0][self.plaq[i][j+1]-5]
                wn[1]*=self.ptype[1][self.plaq[i][j+1]-5]
                wn[2]*=self.ptype[2][self.plaq[i][j+1]-5]
            if i>0:
                wn[0]*=self.ptype[0][self.plaq[i-1][j]-4]
                wn[1]*=self.ptype[1][self.plaq[i-1][j]-4]
                wn[2]*=self.ptype[2][self.plaq[i-1][j]-4]
            if i<self.m-1:
                wn[0]*=self.ptype[0][self.plaq[i+1][j]-1]
                wn[1]*=self.ptype[1][self.plaq[i+1][j]-1]
                wn[2]*=self.ptype[2][self.plaq[i+1][j]-1]

            # if wn[1]/wo[1]>1:
            #         with open(file_path, 'a') as f:
            #             print("now is", nt,"p= ", wn[1]/wo[1]," w_new= ",self.w[1]*wn[1]/wo[1]," w_old= ",self.w[1], file=f)
            if jd<(wn[1]/wo[1]):
                # if wn[1]/wo[1]>1:
                # with open(file_path, 'a') as f:
                    # print(self.w, file=f)
                    # print("now is", nt,"p= ", wn[1]/wo[1]," w_new= ",self.w[1]*wn[1]/wo[1]," w_old= ",self.w[1], file=f)
                    # print(wn[1],'  ',wo[1],file=f)
                    # print("wo_should_be ", self.ptype[1][self.plaq[i-1][j]]*self.ptype[1][self.plaq[i+1][j]]*self.ptype[1][self.plaq[i][j-1]]*self.ptype[1][self.plaq[i][j+1]], file=f)
                    # print("old_check=  ",self.checkw(), file=f)
                        # print(self.plaq, file=f)
                        # print(i,j, self.plaq[i][j], file=f)
                    # print("  ",self.plaq[i+1][j],file=f)
                    # if j>0:
                    #     print(self.plaq[i][j-1],end=' ',file=f)
                    # print(self.plaq[i][j],end=' ',file=f)
                    # if j<self.n-2:
                    #     print(self.plaq[i][j+1],file=f)
                    # else:
                    #     print(file=f)
                    # print("  ",self.plaq[i-1][j],file=f)
                    # print("updated", file=f)
                self.plaq[i][j]=5
                for k in range(3):
                    self.w[k]*= wn[k]/wo[k]
                if j>0:
                    self.plaq[i][j-1]      += 10
                    self.plaq[i-1][j-1]    += 8
                    self.plaq[i+1][j-1]    += 2
                    for k in range(3): 
                        self.wconfig[k][i][j-1] = self.ptype[k][self.plaq[i][j-1]]
                if j<self.n-2:
                    self.plaq[i][j+1]      -= 5
                    self.plaq[i-1][j+1]    -= 4
                    self.plaq[i+1][j+1]    -= 1
                    for k in range(3):
                        self.wconfig[k][i][j+1] = self.ptype[k][self.plaq[i][j+1]]
                if i>0:
                    self.plaq[i-1][j]      -= 4 
                    for k in range(3):
                        self.wconfig[k][i-1][j] = self.ptype[k][self.plaq[i-1][j]]
                if i<self.m-1:
                    self.plaq[i+1][j]      -= 1
                    for k in range(3):
                        self.wconfig[k][i+1][j] = self.ptype[k][self.plaq[i+1][j]]
                # if self.w[1]<1e-8:
                #     print(self.w[1],self.checkw())
                #     self.plot_figure()
                # with open(file_path, 'a') as f:
                #     print("check_new= ",self.checkw(), file=f)
                #     print(wn[1]/wo[1],self.w, file=f)
                    # print("now number: ",nt," now dE= ",(np.log(self.w[0]/self.w[2])/(self.m*self.dtau))," now_w= ", self.w, file=f)
                
        
        #无论有没有更新，都要计算结果并且添加到sum中
        if nt>=self.heatn and nt%self.step_gap==0:  #热化完成再计数,并且隔几步采一次样保证样本之间不关联
            self.sumE   += (np.log(self.w[0]/self.w[2])/(self.m*self.dtau))
            self.sumE2  += (np.log(self.w[0]/self.w[2])/(self.m*self.dtau))**2
            self.sumw   += 1
            self.lcheck.append(np.log(self.w[0]/self.w[2])/(self.m*self.dtau))
        
        # if  nt%self.step_gap==0: 
        #     self.E_ave  += (np.log(self.w[0]/self.w[2])/(self.m*self.dtau))
        #     self.w_ave  += 1
        # if nt%1000==999: #每1000步记录一次
        #     self.lE.append(self.E_ave/self.w_ave)
        #     self.E_ave = 0
        #     self.w_ave = 0
        
            


    def find_possible(self): #找到一个可以更新的plaquette
        i  = np.random.randint(1,self.m-1)
        j  = np.random.randint(0,self.n-1)
        while (self.update_able(i,j)==0):
            i  = np.random.randint(1,self.m-1)
            j  = np.random.randint(0,self.n-1)
        return i,j
    
    def update_able(self, i, j): #判断一个plaquette是否可以更新,这需要判断这个plaquette是否有gate,以及周围的plaquette是否在更新后会出现问题
        if (i+j)%2==1:#位置是否是没有gate的plaquette
            if self.plaq[i][j]==5:#该位置是否可以更新
                if j==0 or self.plaq[i][j-1]==10 or self.plaq[i][j-1]==15: #左侧更新后是否会出问题
                    if j== self.n-2 or self.plaq[i][j+1]==0 or self.plaq[i][j+1]==10: #右侧更新后是否会出问题
                        if self.plaq[i-1][j]==5 or self.plaq[i-1][j]==6: #下侧更新后是否会出问题
                            if self.plaq[i+1][j]==5 or self.plaq[i+1][j]==9: #上侧更新后是否会出问题
                                return 1
                
            elif self.plaq[i][j]==10:#该位置是否可以更新
                if j==0 or self.plaq[i][j-1]==0 or self.plaq[i][j-1]==5: #左侧更新后是否会出问题
                    if j== self.n-2 or self.plaq[i][j+1]==5 or self.plaq[i][j+1]==15: #右侧更新后是否会出问题
                        if self.plaq[i-1][j]==10 or self.plaq[i-1][j]==9: #下侧更新后是否会出问题
                            if self.plaq[i+1][j]==10 or self.plaq[i+1][j]==6: #上侧更新后是否会出问题
                                return 1
                 
        return 0
    
    def run_MC(self):
        for i in range(self.tsample):
            if i%1000000==0 :
                print("now step number: ", i)
            #     with open(file_path, 'a') as f:
            #         print("now number: ",i," now energy= ",self.sumE/self.sumw, file=f)
            #         print("E(ω)= ", (np.log(self.w[0]/self.w[2])/(self.m*self.dtau)),"E(ω')= ",(self.w[0]-self.w[2])/(self.m*self.dtau*self.w[1])," w= ",self.w,file=f)
            self.local_update(i)
            # if self.w[1]<1e-8:
            #     print(self.w[1],self.checkw())
            #     self.plot_figure()
        self.E_gs = self.sumE/self.sumw
        self.E_err= (np.sqrt(self.sumE2/(self.sumw-1) - (self.sumE/self.sumw)*(self.sumE/(self.sumw-1))))/(np.sqrt(self.sumw))#standard error 要在标准差(standard deviation)的基础上除以sqrt(n)
    
    def checkw(self):#所有有权重plaquette的权重乘积,与更新结果对比校验
        ans=1
        for i in range (self.m):
            for j in range (self.n-1):
                if (i+j)%2==0:
                    ans*=self.wconfig[1][i][j]
        return ans
    
    def draw_square(self, ax, x, y, size=1, ptype=0, color="black"):#画一个构型中的一块plaquette
        """
        绘制带有不同样式的方形，包括加粗边、对角线、阴影等，并支持左下角加粗点。
        
        参数：
        - ax: Matplotlib 轴对象
        - x, y: 方形左下角坐标
        - size: 方形边长
        """
        # 画基础方形
        square = patches.Rectangle((x, y), size, size, linewidth=2, edgecolor="gray", facecolor='none')
        ax.add_patch(square)
    
        # 处理加粗边框
        lw_bold = 3  # 加粗线宽度
        if (x+y)%2 ==0:
            if ptype in [5, 15]:
                ax.plot([x, x], [y, y+size], color=color, linewidth=lw_bold)  # 左边框
            if ptype in [10, 15]:
                ax.plot([x+size, x+size], [y, y+size], color=color, linewidth=lw_bold)  # 右边框
            if ptype == 9:
                ax.plot([x, x+size], [y, y+size], color=color, linewidth=lw_bold)  # 主对角线
            if ptype == 6:
                ax.plot([x, x+size], [y+size, y], color=color, linewidth=lw_bold)  # 副对角线
    
        # # 处理加粗点
        # if bold_corner%2==1:
        #     ax.scatter(x, y, color=color, s=lw_bold*20)  # 加粗左下角的点
        else:
            if ptype in [5, 7, 13, 15]:
                ax.plot([x, x], [y, y+size], color=color, linewidth=lw_bold)  # 左边框
            if ptype in [10, 11, 14, 15]:
                ax.plot([x+size, x+size], [y, y+size], color=color, linewidth=lw_bold)  # 右边框
            ax.fill([x, x, x+size, x+size], [y, y+size, y+size, y], color='gray', alpha=0.5)
    
        ax.set_xlim(0, self.n-1)
        ax.set_ylim(0, self.m)

    def plot_figure(self):#画出当前构型
        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(self.m):
            for j in range(self.n-1):
                self.draw_square(ax, j, i, ptype=self.plaq[i][j])
        
        # ax.set_aspect('equal')
        plt.show()

    def check_heating(self):
        x = np.linspace(1, len(self.lE), len(self.lE))
        fig, axes = plt.subplots(figsize=(10, 10))
        axes.plot(x, self.lE, label="Energy")
        axes.set_xlabel("Number of Samples ")
        axes.set_ylabel("Energy")
        axes.legend()
        # fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        # axes[0,0].plot(x, self.lE, label="Energy")
        # axes[0,0].set_xlabel("Number of Samples")
        # axes[0,0].set_ylabel("Energy")
        # axes[0,0].legend()
        # axes[0,1].plot(x, self.lw, label="w")
        # axes[0,1].set_xlabel("Number of Samples")
        # axes[0,1].set_ylabel("w")
        # axes[0,1].legend()
        # axes[1,0].plot(x, self.lwerr, label="werr")
        # axes[1,0].set_xlabel("Number of Samples")
        # axes[1,0].set_ylabel("werr")
        # axes[1,0].legend()
        plt.show()

    def check_normal(self): #检验得到的数据是否符合正态分布
        fig, axes = plt.subplots(1,2,figsize=(20, 10))
        #1.直方图 + KDE, 直观判断
        axes[0].hist(self.lcheck, bins=100, density=True, alpha=0.6, color='g', label="Histogram")
        
        # KDE 曲线
        sns.kdeplot(self.lcheck, color="red", label="KDE Curve", ax=axes[0])
        
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Histogram and KDE")
        axes[0].legend()

        #2. Q-Q 图, 定量判断
        stats.probplot(self.lcheck, dist="norm", plot=axes[1])
        axes[1].set_title("Q-Q Plot")

        # Anderson-Darling 检验
        anderson_test = stats.anderson(self.lcheck, dist='norm')
        l_confidence  = [1,5]
        fig.suptitle(f'not normal', fontsize=16)
        for i in range(2):
            if anderson_test.statistic<anderson_test.critical_values[i]:
                fig.suptitle(f'normal ,confidence= {l_confidence[i]}%', fontsize=16)    
                break
        
        plt.show()


if __name__ =="__main__":
    args           = parser.parse_args()
    # print(args.dtau)
    # args.dtau      = args.dtau/args.ntimes
    # print(args.dtau)
    filename          = f"WLMC_Jx={args.Jx:.2f}_Jy={args.Jy:.2f}_Jz={args.Jz:.2f}_h={args.fieldh:.2f}.txt" #运行结果存储文件名
    current_directory = os.path.dirname(os.path.abspath(__file__))
    target_folder     = os.path.join(current_directory, 'datas_out')
    os.makedirs(target_folder, exist_ok=True)                                          #如果不存在目标文件夹就创建它,也可以使用下面注释的内容
    file_path         = os.path.join(target_folder, filename)
    #将文件初始化为一个空文档
    with open(file_path, 'w') as f:
        pass
    MC_solver = WLQMC(args.n, args.Jx, args.Jy, args.Jz, args.fieldh, args.tau, args.dtau, args.ntimes, args.tsample, args.step_gap, args.heatn, args.w)
    # print(MC_solver.dtau, MC_solver.dtau*MC_solver.m)
    # print(MC_solver.ptype[1][6]/MC_solver.ptype[1][9])
    with open(file_path, 'a') as f:
        print("original_w= ", MC_solver.w, file=f)
        print("types= ", MC_solver.ptype[1], file=f)
        # print(MC_solver.config[0], file=f)
        # print(MC_solver.config[1], file=f)
        # print(MC_solver.plaq, file=f)
        # print(MC_solver.wconfig[1][0], file=f)
        # print(MC_solver.wconfig[1][1], file=f)
    MC_solver.run_MC()
    # with open(file_path, 'a') as f:
    #     print(MC_solver.plaq, file=f)

    print("finall E_gs_min= ", MC_solver.E_gs-MC_solver.E_err," finall E_gs=", MC_solver.E_gs, " finall E_gs_max= ", MC_solver.E_gs+MC_solver.E_err)
    # print("last w is ",MC_solver.checkw())
    # MC_solver.check_heating()
    MC_solver.check_normal()
    # MC_solver.plot_figure()
