#!/usr/bin/env python3
#coding: utf-8
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import argparse
import time

"""
twoLayerNeuralNetwork.py
２層パーセプトロン 
forループの代わりに行列演算にした高速化版

入力層 - 中間層 - 出力層の2層構造で固定

中間層の活性化関数にはtanh関数, sigmoid logistic関数, ReLU関数が使える
出力層の活性化関数にはtanh関数, sigmoid logistic関数, ReLU関数, 恒等関数, softmax関数が使える


注意

 評価関数Jに何を使うかで，出力層のδの計算が異なる．

 [回帰問題の場合] 
 Jに二乗誤差を選ぶのが普通
 出力層の活性化関数に恒等関数を選べば
 出力層のδは (g3 - t)

 [2クラス問題の場合]
 Jに交差エントロピーを選び
 出力層の活性化関数にsigmoid関数を選べば，
 出力層のδは (g3 - t)
 
 [多クラス問題の場合]
 Jに交差エントロピーを選び，
 出力層の活性化関数にsoftmax関数を選べば，
 出力層のδは (g3 - t)

 このような評価関数と活性化関数の組み合わせの場合は
 出力層のδは全て (g3 - t) となる．

 それ以外の組み合わせの場合，
 出力層のδには出力層の活性化関数の微分とJの微分が入る．(中間層と同じ)
"""

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

#
# 活性化関数
#
### sigmoid関数 ###
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid'(x) = sigmoid(x)*(1-sigmoid(x))
# この関数では入力はsigmoid(x)であることに注意 (逆伝播用)[式(8.6)]
def sigmoid_deriv(sig_x):
    return sig_x * (1 - sig_x)

### tanh関数 ###
def tanh(x):
    return np.tanh(x)

# tanh'(x) = 1 - tanh(x)**2
# この関数では入力はtanh(x)であることに注意 (逆伝播用)[式(8.8)] 
def tanh_deriv(tanh_x):
    return 1.0 - tanh_x ** 2

### ReLU関数 ###
def ReLU(x):
    return np.max(x, 0)

# この関数では入力はReLU(x)であることに注意 (逆伝播用)
def ReLU_deriv(relu_x):
    return 1 if relu_x > 0 else 0

### 恒等写像関数 ###
def identity(x):
    return x

# この関数では入力はidentity(x)であることに注意 (逆伝播用)
def identity_deriv(ident_x):
    return 1

### softmax関数 ###
def softmax(x):
    temp = np.exp(x)
    if x.ndim == 1:
        #print('softmax= ', np.round(temp / np.sum(temp), 3))
        return temp / np.sum(temp)
    else:
        result = 1.0 / np.sum(temp, axis=1)
        #print('softmax= ', np.round((temp.T / np.sum(temp, axis=1).T).T, 3))
        return (temp.T / np.sum(temp, axis=1).T).T

def softmax_deriv(x):
    eprint("### Not defined! ###")
    sys.exit()
    return 1



#
# ２層パーセプトロン
#
class TwoLayerPerceptron:
    def __init__(self, numInput, numMiddle, numOutput, actFuncMiddle="tanh", actFuncOutput="sigmoid", seed=-1, addBias=False):
        """多層パーセプトロンを初期化
        numInput:   入力層のユニット数（バイアスユニットは除く）
        numMiddle:  中間層のユニット数（バイアスユニットは除く）
        numOutput:  出力層のユニット数
        actMiddle:  中間層の活性化関数（tanh or sigmoid or ReLU）
        actOutput:  出力層の活性化関数（tanh or sigmoid or ReLU or identity or softmax）
        """
        # 引数の指定に合わせて中間層の活性化関数とその微分関数を設定
        self.__actFuncMiddle = actFuncMiddle
        if   actFuncMiddle == "tanh":
            self.__actMiddle = tanh
            self.__actMiddle_deriv = tanh_deriv
        elif actFuncMiddle == "sigmoid":
            self.__actMiddle = sigmoid
            self.__actMiddle_deriv = sigmoid_deriv
        elif actFuncMiddle == "ReLU":
            self.__actMiddle = ReLU
            self.__actMiddle_deriv = ReLU_deriv
        else:
            eprint ("ERROR: actMiddle is tanh or sigmoid or ReLU")
            sys.exit()

        # 引数の指定に合わせて出力層の活性化関数とその微分関数を設定
        self.__actFuncOutput = actFuncOutput
        if   actFuncOutput == "tanh":
            self.__actOutput = tanh
            self.__actOutput_deriv = tanh_deriv
        elif actFuncOutput == "sigmoid":
            self.__actOutput = sigmoid
            self.__actOutput_deriv = sigmoid_deriv
        elif actFuncOutput == "ReLU":
            self.__actOutput = ReLU
            self.__actOutput_deriv = ReLU_deriv
        elif actFuncOutput == "softmax":
            self.__actOutput = softmax
            self.__actOutput_deriv = softmax_deriv
        elif actFuncOutput == "identity":
            self.__actOutput = identity
            self.__actOutput_deriv = identity_deriv
        else:
            eprint ("ERROR: actOutput is tanh or sigmoid or ReLU or softmax or identity")
            sys.exit()

        # バイアスユニットを追加するか否か
        self.__addBias = addBias
        self.__extendedInput = False

        # バイアスユニットを追加する場合は入力層と中間層のユニット数は+1
        self.__numInput  = numInput+1  if addBias else numInput
        self.__numMiddle = numMiddle+1 if addBias else numMiddle
        self.__numOutput = numOutput

        # 乱数の種を設定
        if seed > 0:
            np.random.seed(seed)

        # 重みを (-1.0, 1.0)の一様乱数で初期化
        self.__weightMiddle = np.random.uniform(-1.0, 1.0, (self.__numMiddle, self.__numInput))  # 入力層-中間層間
        self.__weightOutput = np.random.uniform(-1.0, 1.0, (self.__numOutput, self.__numMiddle)) # 中間層-出力層間
        if addBias: self.__weightMiddle[0] = np.zeros(self.__weightMiddle.shape[1])

        print(40*"=")
        print("バイアス有り               : ", self.__addBias)
        print("入力層Unit数(含むbias unit): ", self.__numInput)   
        print("中間層Unit数(含むbias unit): ", self.__numMiddle)  
        print("出力層Unit数               : ", self.__numOutput)  
        print("中間層の活性化関数         : ", self.__actFuncMiddle)  
        print("出力層の活性化関数         : ", self.__actFuncOutput) 
        print(40*"=")

    def get_arch(self):
        """ Network層数を返す(bias unitも含む)"""
        return self.__numInput, self.__numMiddle, self.__numOutput, self.__actFuncMiddle, self.__actFuncOutput, self.__addBias

    def get_middle_weight(self):
        """ 中間層の重み行列 w[middle][input] """
        return self.__weightMiddle

    def get_output_weight(self):
        """ 出力層の重み行列 w[middle][input] """
        return self.__weightOutput


    def fit(self, X, T, eval_func, learning_rate=0.1, epochs=10000):
        """訓練データを用いてネットワークの重みを更新する"""

        # 入力データの最初の列にバイアスユニットの入力1を追加
        if self.__addBias and self.__extendedInput == False: 
            X = np.hstack([np.ones([X.shape[0], 1]), X])
            # print( "X.shape: ", X.shape)
            self.__extendedInput = True     # 入力ベクトルは拡張済


        # 逐次学習
        # 訓練データからランダムサンプリングして重みを更新をepochs回繰り返す
        for k in range(epochs):
            print ("\r [epoch# %5d] "%(k), end="")

            # 訓練データをランダムに並び替える
            index = np.random.permutation(np.arange(len(X)))
            X_shuffle = X[index]
            t_shuffle = T[index]

            # 重み更新
            self.fit_local(X_shuffle, t_shuffle, learning_rate)

            # 更新済重みを用いて順伝播によりネットワークの出力を計算し誤差算出
            g2 = self.__actMiddle(np.dot(X_shuffle, self.__weightMiddle.T))
            g3 = self.__actOutput(np.dot(g2, self.__weightOutput.T))
            print("Loss = %10.8f"%(eval_func(g3, t_shuffle)), end="")
            if k%100 == 0: print("")

        print("\n")

    def fit_local(self, X, T, learning_rate):
        # 入力データの最初の列にバイアスユニットの入力1を追加
        if self.__addBias and self.__extendedInput == False: 
            X = np.hstack([np.ones([X.shape[0], 1]), X])

        for x, t in zip(X, T):
            """ 訓練データ1サンプルでネットワーク重みを1回更新する """
            # [順] 入力を順伝播させて中間層の出力を計算 (g2[numMiddle]=weightMiddle[numMiddle][numInput] x[numInput])
            g2 = self.__actMiddle(np.dot(self.__weightMiddle, x))     # 式(8.21)
            if self.__addBias: g2[0] = 1.0

            # [順] 中間層の出力を順伝播させて出力層の出力を計算 (g3[numOutput]=weightOutput[numOutput][numMiddle] g2[numMiddle])
            g3 = self.__actOutput(np.dot(g2, self.__weightOutput.T))  # 式(8.22)

            # 出力層の誤差を計算 (deltaOutput[numOutput])
            # 評価関数J: 二乗誤差，　　　　活性化関数: 恒等写像                 [Sec.8-6-4]
            # 評価関数J: 交差エントロピー，活性化関数: シグモイド (2クラス)     [Sec.8-6-4]
            # 評価関数J: 交差エントロピー，活性化関数: ソフトマックス (多クラス)[Sec.8-6-4]
            deltaOutput = g3 - t 
            #deltaOutput = self.__actOutput_deriv(g3) * (g3 - t) # 評価関数J: 二乗誤差, 活性化関数任意

            # [逆] 出力層の誤差を逆伝播させて中間層の誤差を計算 
            # (deltaMiddle[numMiddle]=actMiddle_deriv(g2)[numMiddle] * __weightOutput.T[1][numOutput] deltaOutput[numOutput])
            deltaMiddle = self.__actMiddle_deriv(g2) * np.dot(deltaOutput, self.__weightOutput)

            # [逆] 中間層の誤差を用いて中間層の重みを更新
            # 行列演算になるので2次元ベクトルに変換する必要がある
            x = np.atleast_2d(x)
            deltaMiddle = np.atleast_2d(deltaMiddle)
            #print('deltaMiddle=', deltaMiddle)
            #print('x=', x)
            #print('dot(deltaMiddle,x)=', np.dot(deltaMiddle.T, x))
            if self.__addBias: deltaMiddle[:,0] = 0.0
            self.__weightMiddle -= learning_rate * np.dot(deltaMiddle.T, x)
            #print('Wmiddle:', self.__weightMiddle)

            # 出力層の誤差を用いて出力層の重みを更新
            g2 = np.atleast_2d(g2)
            deltaOutput = np.atleast_2d(deltaOutput)
            self.__weightOutput -= learning_rate * np.dot(deltaOutput.T, g2)
            #print('Wout:', self.__weightOutput)


    def conv_onehot(self, labels, num_class):
        """ num_class問題の教師ラベルベクトルをone hot表現の行列に変換"""
        onehot = np.zeros((len(labels), num_class), dtype=float)
        for label, supervised in zip(labels, onehot):
            supervised[label] = 1
        return onehot


    def predict(self, X):
        """データ群Xに対するnetworkの出力値を予測"""
        # 入力データの最初の列にバイアスユニットの入力1を追加
        if self.__addBias: 
            X = np.hstack([np.ones([X.shape[0], 1]), X])

        # 順伝播によりネットワークの出力を計算
        g2 = self.__actMiddle(np.dot(X, self.__weightMiddle.T))
        if self.__addBias: g2[:,0] = 1.0
        g3 = self.__actOutput(np.dot(g2, self.__weightOutput.T))
        return g2, g3

    def infer_2class(self, X, T):
        """ 
        未知バターンの認識
            X: 学習パターン行列
            T: 教師信号ベクトル (1 or 0)
        """
        g2, g3 = self.predict(X) # まずNetworkに通す

        if g3.shape[1] != 1:
            eprint("2-class識別なので出力層のunit数は1のはず. g3.shape[1]=", g3.shape[1])
            sys.exit()

        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for label, estimated in zip(T, g3):
           if (label == 1) and (estimated >= 0.5): TP += 1
           if (label == 0) and (estimated <  0.5): TN += 1
           if (label == 0) and (estimated >= 0.5): FP += 1
           if (label == 1) and (estimated <  0.5): FN += 1

        print("          ------------------")
        print("          |      |推定結果 |")
        print("          |      | Pos Neg |")
        print("          |------+---------|")
        print("|TP FN|   |   Pos| %3d %3d |" % (TP, FN))
        print("|     | = |真    |         |")
        print("|FP TN|   |   Neg| %3d %3d |" % (FP, TN))
        print("          ------------------")
        accuracy = (TP+TN) / len(T)
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        F_measure = 2*precision*recall / (precision + recall)
        print("Accuracy  = %5.2f %%" % (accuracy*100))
        print("Precision = %5.2f %%" % (precision*100))
        print("Recall    = %5.2f %%" % (recall*100))
        print("F-measure = %5.2f %%" % (F_measure*100))

        cMat = np.zeros((2,2), dtype=int) # confusion matrix
        cMat[0,0] = TP
        cMat[1,1] = TN
        cMat[0,1] = FN
        cMat[1,0] = FP

        result_infer = np.zeros(len(T)).reshape(len(T),1)
        index        = g3 >= 0.5
        for i in range(len(T)):
            result_infer[i] = 1 if index[i] == True else 0
        return result_infer, cMat


    def infer_Nclass(self, X, labels, num_class):
        """ 
        未知バターンの認識
            X:      学習パターン行列
            labels: 教師信号ベクトル 
        """
        cMat = np.zeros((num_class,num_class), dtype=int) # confusion matrix
        result_infer = np.zeros(len(labels)).reshape(len(labels),1)

        g2, g3 = self.predict(X) # まずNetworkに通す

        # Confusion Matrixの生成
        for estimated, label in zip(g3, labels):
            estimated_class = np.argmax(estimated) # 最大値を探す
            supervised_class = np.argmax(label)
            cMat[supervised_class, estimated_class] += 1
            
        for i in range(len(g3)):
            result_infer[i] = np.argmax(g3[i]) # 最大値を探す
            
        # Confusion Matrixの表示
        print( cMat )

        print("認識率= %6.3f %% (%d/%d)" % (np.trace(cMat) / len(X) * 100.0, np.trace(cMat), len(X)), end="")
        print(" [cross entropy = ", self.error_Nclass_cross_entropy(g3, labels), "]")

        return result_infer, cMat
            

    def error_mse(self, g3, T):
        """
        二乗誤差を計算 (回帰問題の評価関数)
        g3: 出力層データ ( g3[NUM_DATA] 入力データ数分, 出力層数は1 )
        T:  教師信号値データ ( T[NUM_DATA] 入力データ数分 )
        """
        L = ((g3 - T)**2).sum()
        L /= g3.shape[0]
        return L


    def error_2class_cross_entropy(self, g3, labels):
        """
        2クラスの交差エントロピーを計算 (2クラス問題の評価関数)
        g3:     出力層データ ( g3[NUM_DATA] 入力データ数分, 2クラス問題なので出力層数は1 )
        labels: 教師クラスラベルデータ ( labels[NUM_DATA] = 0 or 1 )
        """

        L = float(np.dot(labels.T,np.log(g3))) + float(np.dot((1-labels).T, np.log(1-g3)))
        L /= g3.shape[0]
        return -L

    def error_Nclass_cross_entropy(self, g3, labels):
        """
        Nクラスの交差エントロピーを計算 (多クラス問題の評価関数)
        g3:     出力層データ ( g3[NUM_DATA][NUM_CLASS] )
        labels: 教師クラスラベルデータ ( labels[NUM_DATA][NUM_CLASS] = 0 or 1, OneHot表現)
        """
        # print('labels.shape=', labels.shape)
        # print('g3.shape=', g3.shape)
        # print('labels=',labels)
        # print('g3=',np.round(g3, 2))
        # print('labels*log(g3)=', labels*np.log(g3))

        L = labels * np.log(g3)
        L = L.sum() / g3.shape[0]
        return -L
