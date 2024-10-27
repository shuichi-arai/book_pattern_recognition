#!/usr/bin/env python3
#coding: utf-8
#------------------------------------------------
#
# 2 class識別 (XORの学習)
#
#------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import argparse
import time
import twoLayerNeuralNetwork as nn


if __name__ == "__main__":
    """
    Main function
        -l (--loop)     <loop count>                  (default: 10000)
        -r (--rho)      <learning rate>               (default: 0.1)
        -u (--num_unit) <unit number of middle layer> (default: 3)
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', '--loop', type=int, default=10000, help='loop count')
    parser.add_argument('-r', '--rho', type=float, default=0.1, help='learning parameter rho')
    parser.add_argument('-u', '--num_unit', type=int, default=3, help='unit number of middle layer')
    args = parser.parse_args()   

    rho = args.rho
    """ XOR の学習 """
    #mlp = nn.TwoLayerPerceptron(2, args.num_unit, 1, "tanh", "sigmoid", addBias=True)
    mlp = nn.TwoLayerPerceptron(2, args.num_unit, 1, "sigmoid", "sigmoid", addBias=True)
    #print("ネットワーク構造(入力，中間，出力, 中間層活性化関数, 出力層活性化関数, Bias有): ", mlp.get_arch())

    X  = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # XOR 入力
    g3 = np.array([[0],    [1],    [1],    [0]])    # XOR 出力

    # Learning
    mlp.fit(X, g3, mlp.error_2class_cross_entropy, args.rho, args.loop)

    # Recognition
    print("=== 推定結果 ===")
    print(" 入力     出力")
    g2, g3 = mlp.predict(X)
    for i in range(4):
        print(X[i,:], "%10.7f"%(g3[i][0]))

