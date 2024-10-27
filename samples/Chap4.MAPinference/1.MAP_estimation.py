#!/usr/bin/env python3
#coding:utf-8
#
# MAP推定
#
#	クラス依存確率密度関数が与えられている際に，2クラスの事前確率によって
#	事後確率分布がどのように変化し，その結果，未知パターンに対して
#	どのようにクラス推定すれば良いのかを理解するためのサンプル．
#
#	2クラスの事前確率P(ω1), P(ω2)を変化させると，事後確率がどのように
#	変化し，その結果，未知パターンへのクラス推定結果がどのように変化するのか
#	確認する．
#
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import functools
import scipy.stats as st
import math
import sys
from matplotlib.widgets import Slider, Button, RadioButtons

# Slider, Button, RadioButton のfaceの色
axcolor = 'lightgoldenrodyellow'

class Norm:
    """
    1変量正規分布 (univariate normal distribution)
    """
    def __init__(self, mu=0.0, var=1.0):
        self.__mu    = mu
        self.__var   = var
        self.__sigma = np.sqrt(self.__var)
        self.__N     = 0

    def get_param(self):
        """ 正規分布パラメタの取得 """
        return self.__mu, self.__var

    def pdf(self, x):
        """ 与えられた分布の(x)における確率密度値を求める"""
        return st.norm.pdf(x, self.__mu, np.sqrt(self.__var))


    def this_likelihood(self, x):
        """ 与えられた1次元正規分布に対するパターンxの尤度を求める """
        prob = 1 / np.sqrt(2*np.pi*self.__var) * np.exp( -(x-self.__mu)**2 / (2*self.__var) )
        return prob

class Px:
    """
    クラス依存確率密度関数 (class conditional probability) [テキストと同じ関数]
    """
    def __init__(self, p1=1.0, mu1=0.0, sig1=1.0, p2=0.0, mu2=0.0, sig2=1.0, p3=0.0, mu3=0.0, sig3=1.0):
        self.__p1     = p1
        self.__mu1    = mu1
        self.__sigma1 = sig1
        self.__var1   = sig1*sig1
        self.__p2     = p2
        self.__mu2    = mu2
        self.__sigma2 = sig2
        self.__var2   = sig2*sig2
        self.__p3     = p3
        self.__mu3    = mu3
        self.__sigma3 = sig3
        self.__var3   = sig3*sig3
        self.__norm1 = Norm(self.__mu1, self.__var1)
        self.__norm2 = Norm(self.__mu2, self.__var2)
        self.__norm3 = Norm(self.__mu3, self.__var3)

    def pdf(self, x):
        """ 与えられた分布の(x)における確率密度値を求める"""
        p = self.__p1 * self.__norm1.pdf(x) + self.__p2 * self.__norm2.pdf(x) + self.__p3 * self.__norm3.pdf(x)
        return p



# Priori Probavility parameters for Slider
P1_min, P1_max, P1_init = 0.0, 1.0, 0.5
P2_min, P2_max, P2_init = 0.0, 1.0, 0.5
    
# Drawing range of p(ω|x)
x_min, x_max = 0,  10
y_min, y_max = 0.0, 0.6
x_step = 0.01
x = np.arange(x_min, x_max, x_step)

# Drawing area size
fig = plt.figure(figsize=(12,8))

#
# Drawing sub-area for Class conditional probability [左上]
#                Left  Btm   Wid   Hight 
axcP = plt.axes([0.05, 0.55, 0.40, 0.40], facecolor=axcolor)
axcP.grid()
axcP.set_xlim(x_min, x_max)
axcP.set_ylim(y_min, y_max)

# Class conditional probability
p_x_w1 = Px(0.45, 3.4, 0.52, 0.53, 4.6, 0.50, 0.02, 6.8, 0.60)
p_x_w2 = Px(0.57, 4.2, 0.40, 0.40, 5.5, 0.55, 0.03, 2.0, 0.95)

plt.plot(x, p_x_w1.pdf(x), lw=2, color='blue', label=r'$p(x|\omega_1)$')
plt.plot(x, p_x_w2.pdf(x), lw=2, color='red',  label=r'$p(x|\omega_2)$')
plt.legend()



#
# Slider
#                Left  Btm   Wid   Hight 
axP1 = plt.axes([0.05, 0.25, 0.40, 0.03], facecolor=axcolor)
axP2 = plt.axes([0.05, 0.20, 0.40, 0.03], facecolor=axcolor)

sP1 = Slider(axP1, r'$P(\omega_1)$', P1_min, P1_max, valinit=P1_init, valstep=0.001, valfmt='%.3f')
sP2 = Slider(axP2, r'$P(\omega_2)$', P2_min, P2_max, valinit=P2_init, valstep=0.001, valfmt='%.3f')


#
# Drawing sub-area for the numerator and denominator(evidence) of MAP fraction [右上]
#                Left  Btm   Wid   Hight 
axMAP = plt.axes([0.55, 0.55, 0.40, 0.40], facecolor=axcolor)
axMAP.grid()
axMAP.set_xlim(x_min, x_max)
axMAP.set_ylim(y_min, y_max)

line2dObj1, = plt.plot(x, sP1.val * p_x_w1.pdf(x), lw=2, color='blue', label=r'$p(x|\omega_1) P(\omega1)$')
line2dObj2, = plt.plot(x, sP2.val * p_x_w2.pdf(x), lw=2, color='red',  label=r'$p(x|\omega_2) P(\omega2)$')
line2dObj3, = plt.plot(x, sP1.val * p_x_w1.pdf(x) + sP2.val * p_x_w2.pdf(x), lw=2, color='black', label='evidence')
plt.legend()


#
# Drawing sub-area for Posterior probability [右下]
#                Left  Btm   Wid   Hight 
axPost = plt.axes([0.55, 0.05, 0.40, 0.40], facecolor=axcolor)
axPost.grid()
axPost.set_xlim(x_min, x_max)
axPost.set_ylim(y_min, 1.0)

evidence = sP1.val * p_x_w1.pdf(x) + sP2.val * p_x_w2.pdf(x)
post1 = sP1.val * p_x_w1.pdf(x) / evidence
post2 = sP2.val * p_x_w2.pdf(x) / evidence

line2dObj4, = plt.plot(x, post1, lw=2, color='blue', label=r'$P(\omega_1|x)$')
line2dObj5, = plt.plot(x, post2, lw=2, color='red',  label=r'$P(\omega_2|x)$')
plt.legend()


def cb_update1(val):
    """
        Slider Event Callback Func.
    """
    line2dObj1.set_ydata(sP1.val * p_x_w1.pdf(x))
    sP2.valinit = 1.0 - sP1.val
    sP2.reset()
    line2dObj2.set_ydata(sP2.val * p_x_w2.pdf(x))
    line2dObj3.set_ydata(sP1.val * p_x_w1.pdf(x) + sP2.val * p_x_w2.pdf(x))
    evidence = sP1.val * p_x_w1.pdf(x) + sP2.val * p_x_w2.pdf(x)
    post1 = sP1.val * p_x_w1.pdf(x) / evidence
    post2 = sP2.val * p_x_w2.pdf(x) / evidence
    line2dObj4.set_ydata(post1)
    line2dObj5.set_ydata(post2)
    fig.canvas.draw_idle()

def cb_update2(val):
    """
        Slider Event Callback Func.
    """
    line2dObj2.set_ydata(sP2.val * p_x_w2.pdf(x))
    sP1.valinit = 1.0 - sP2.val
    sP1.reset()
    line2dObj1.set_ydata(sP1.val * p_x_w1.pdf(x))
    line2dObj3.set_ydata(sP1.val * p_x_w1.pdf(x) + sP2.val * p_x_w2.pdf(x))
    evidence = sP1.val * p_x_w1.pdf(x) + sP2.val * p_x_w2.pdf(x)
    post1 = sP1.val * p_x_w1.pdf(x) / evidence
    post2 = sP2.val * p_x_w2.pdf(x) / evidence
    line2dObj4.set_ydata(post1)
    line2dObj5.set_ydata(post2)
    fig.canvas.draw_idle()

sP1.on_changed(cb_update1) # Set the Event callback func. 
sP2.on_changed(cb_update2) # Set the Event callback func. 


#
# Reset Button
#                  Left  Btm   Wid   Hight 
resetax = plt.axes([0.1, 0.05, 0.1, 0.04])
buttonR = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def cb_reset(event):
    """
        Reset Button Event Callback Func.
    """
    sP1.valinit = P1_init
    sP2.valinit = P2_init
    sP1.reset()
    sP2.reset()

buttonR.on_clicked(cb_reset) # Set the Event callback func. 


#
# Quit Button
#                 Left  Btm   Wid   Hight 
quitax = plt.axes([0.3, 0.05, 0.1, 0.04])
buttonQ = Button(quitax, 'Quit', color=axcolor, hovercolor='0.975')

def cb_quit(event):
    """
        Quit Button Event Callback Func.
    """
    sys.exit()

buttonQ.on_clicked(cb_quit) # Set the Event callback func. 

# Plot
plt.show()

