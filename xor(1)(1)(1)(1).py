#异或

import numpy as np
import pandas as pd


X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
Y = np.array([[0.],[1.],[1.],[0.]])

w = np.array([[0.0543,-0.0291],[0.0579,0.0999]])
v = np.array([[0.0801,0.0605]])

θ = np.array([[-0.0703,-0.0939]])#隐藏层阈值
γ = np.array([[-0.0109]])#输出层阈值

α = β = 0.6#学习速率

def fun(x):#sigmoid(x):
    return 1 / (1 + np.exp(-x))
def dfun(x):#sigmoid(x)导函数
    y = 1 / (1 + np.exp(-x))
    return y*(1-y)

def forward(cx,cw,cb):#计算单层网络
    cy = np.dot(cw, cx.T)
    cy1 = cy.reshape(1, len(cy))
    cd1 = np.zeros(shape=cy1.shape)
    for j in range(cy1.size):
        cd1[0][j] = cy1[0][j]-cb[0][j]
        cy1[0][j] = fun(cd1[0][j])
    return cy1,cd1

def computerError1(cy,cy1,cdl):
    error = np.dot((cy-cy1),dfun(cdl))
    return error

def computerError2(cw,error,cd1):
    temp1 = np.dot(cw.T,error)
    temp2 = dfun(cd1)
    error = np.multiply(temp1.T ,temp2)
    return error

for m in range(4000):

    print('第'+str(m)+'次------')
    for i in range(X.shape[0]):
        x = np.array(X[i]).reshape(1, len(X[i]))
        b,bd = forward(x,w,θ)
        c,bc = forward(b,v,γ)
        print("预期：" + str(Y[i]) + "隐藏层:"+str(b.tolist())+"输出层:" + str(c))
        error1 = computerError1(Y[i],c,bc)
        error2 = computerError2(v,error1,bd)
        for p in range(v.shape[0]):
            for q in range(v.shape[1]):
                v[p][q] = v[p][q] + α*error1[0][p]*b[0][q]
        γ = γ + α * error1

        for p in range(w.shape[0]):
            for q in range(w.shape[1]):
                w[p][q] = w[p][q] + β*error2[0][p]*x[0][q]
        θ = θ + β*error2

print(γ.tolist())
print(θ.tolist())
print(w.tolist())
print(v.tolist())









