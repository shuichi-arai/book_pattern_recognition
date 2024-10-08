#!/usr/bin/env python3
#coding:utf-8
"""
最急降下法による評価関数最小解
2次元データ
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def train(vecW, matX, labels, rho):
    """
    最急降下法
        vecW:   重みベクトル
        matX:   学習パターン行列
        labels: 教師信号ベクトル (1 or -1)
        rho:    学習率
    """

    eps = (np.dot(matX, vecW) - labels) # Xw-b
    J = (eps**2).sum()/len(labels)
    rJ = (matX.T * eps).T   # ∂J
    
    nabJ = np.zeros(3)
    for vecX in rJ:
        nabJ += vecX
    nabJ *= (2.0 / len(labels))

    vecW = vecW - rho * nabJ # Wの修正

    return vecW, J


def show2D( vecW, X1, X2, ax ):
    """
    2次元表示
    """
    ax.cla() # 描画クリア

    ax.grid()
    ax.set_xlim(-6,6)
    ax.set_ylim(-6,6)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect('equal', 'datalim')

    # 学習用パターン
    ax.scatter(X1[:,1], X1[:,2], c='red' , marker="o")
    ax.scatter(X2[:,1], X2[:,2], c='blue', marker="o")

    # 分離境界線
    x_fig = np.array(np.arange(-6,7,1))
    y_fig = -(vecW[1]/vecW[2])*x_fig - (vecW[0]/vecW[2])

    ax.plot(x_fig,y_fig, c='green')


def show3D( vecW, X1, X2, ax ):
    """
    3次元表示
    """
    ax.cla() # 描画クリア

    ax.grid()
    ax.set_zlim(0, 6)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x0")

    # 学習用パターン
    ax.scatter(X1[:,1], X1[:,2], X1[:,0], c='red' , marker="o")
    ax.scatter(X2[:,1], X2[:,2], X2[:,0], c='blue', marker="o")

    # 分離境界面
    x = np.arange(-6, 6, 0.5)
    y = np.arange(-6, 6, 0.5)
    X, Y = np.meshgrid(x, y)
    Z = -vecW[1]/vecW[0]*X - vecW[2]/vecW[0]*Y 

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.3, color='green')


def show( vecW, X1, X2, sp1, sp2 ):
    """
    3D + 2D 表示
    """
    print("w = ",  vecW )

    show3D(vecW, X1, X2, sp1)
    show2D(vecW, X1, X2, sp2)
    plt.pause(0.001)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-r', '--rho', type=float, default=0.001, help='learning rate rho')
    parser.add_argument('-s', '--seed', type=int, default=1, help='random seed')
    parser.add_argument('-n', '--num', type=int, default=100, help='number of data')
    parser.add_argument('-l', '--loop', type=int, default=1000, help='loop count')
    args = parser.parse_args()    


    # 乱数の種を与えるか
    if args.seed > 0:
        np.random.seed(args.seed)

    train_num = args.num # 学習データ数
    # class1の学習データ
    mu1= [1.0,1.0]
    var1 = [[1.0, 1.0], [1.0,3.0]]
    sample1 = np.random.multivariate_normal(mu1, var1, train_num//2)

    # class2の学習データ
    mu2= [3.0, -2.0]
    var2 = [[1.0,-1.0], [-1.0,2.0]]
    sample2 = np.random.multivariate_normal(mu2, var2, train_num//2)

    # class1の教師ラベル
    label_X1 = np.ones(train_num//2)                        # ラベル1

    # class2の教師ラベル
    label_X2 = np.ones(train_num//2) * -1                   # ラベル-1

    x0 = np.ones(train_num//2) # x0は常に1

    # 拡張特徴ベクトルを並べた行列
    X1 = np.c_[x0, sample1]
    X2 = np.c_[x0, sample2]

    # 全学習データを統合
    matX   = np.r_[X1, X2]
    labels = np.r_[label_X1, label_X2]
    #print('matX=', matX)
    #print('labels=', labels)

    vecW = np.array([1.0,0.0,1.0]) # 初期の重みベクトル 適当に決める


    # グラフ描画の準備
    fig = plt.figure(figsize=(12,6))
    fig.suptitle('Steepest Descent Method (2-dimensional patterns)')
    sp1 = fig.add_subplot(1, 2, 1, projection='3d')
    sp2 = fig.add_subplot(1, 2, 2)

    # 最急降下法での解法
    for j in range(args.loop):
        vecW, J = train(vecW, matX, labels, args.rho) # 学習

        if j%10 == 0: 
            print("[%d] J= %7.5f  " % (j, J), end="")
            show(vecW, X1, X2, sp1, sp2) # 表示

    show(vecW, X1, X2, sp1, sp2) # 表示
    plt.show() # 最後は表示をkeep
