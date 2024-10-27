#!/usr/bin/env python3
#coding:utf-8
#
# Bayes updating
#
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import functools
import scipy.stats as st
import math
#import distribution as ds
import sys
from matplotlib.widgets import Slider, Button, RadioButtons
from PIL import Image

axcolor = 'lightgoldenrodyellow'

# Pattern
# correct answer: 1 --> 正
# wrong   answer: 0 --> 誤
#pattern = [1, 0, 0, 1, 1, 1] # p.84 式(4.30)
pattern = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] # p.84 式(4.30) + α

pattern_idx = 0
cur_pattern = " "


def drawBarGraph( axis, title, lambda_, draw_y_max, col ):
    """
    Bar Graph の描画
        axis       : グラフ
        title      : タイトル文字列
        lambda_    : lambda_[3]の要素数の確率
        draw_y_max : 表示するbar graph確率の最大値
    """
    xClass = np.arange(3)          # 次元数は３に固定(それ以上のグラフは書けない)
    label = ["$\omega_1$", "$\omega_2$", "$\omega_3$"]  # 横軸のラベル
    axis.cla()
    axis.set_ylim(0, draw_y_max)   # 縦軸の表示最大確率
    axis.set_title(title)          # 棒グラフのタイトル
    # Bar graph
    axis.bar(xClass,  lambda_,  tick_label=label, align="center", color=col)
    # Bar graph内に数値を書く
    for x, y in zip(xClass, lambda_): axis.text(x, y, '{:.3f}'.format(y), ha='center', va='bottom')


def drawPattern( axis, pattern ):
    """
    """
    if pattern==1: # "正"
        img_file = "./fig/True.png"
    else:
        img_file = "./fig/False.png"

    img = Image.open( img_file )
    img_array = np.asarray(img)
    axis.cla()
    axis.axis("off")
    axis.imshow(img_array)
    axis.text(-200,-200, cur_pattern, fontname='IPAPGothic')


def drawArrow( axis ):
    """
    """
    img = Image.open("./fig/arrow.png")
    img_array = np.asarray(img)
    axis.cla()
    axis.axis("off")
    axis.imshow(img_array)



###
### ここからMain
###

# 図の準備
fig = plt.figure(figsize=(8,6))

#
# Slider P(ω)
#                Left  Btm   Wid   Hight 
axP1 = plt.axes([0.07, 0.95, 0.35, 0.03])
axP2 = plt.axes([0.07, 0.90, 0.35, 0.03])
axP3 = plt.axes([0.07, 0.85, 0.35, 0.03])

P1_min, P1_max, P1_init = 0.0, 1.0, 0.5
P2_min, P2_max, P2_init = 0.0, 1.0, 0.4
P3_min, P3_max, P3_init = 0.0, 1.0, 0.1

sP1 = Slider(axP1, r'$P(\omega_1)$', P1_min, P1_max, valinit=P1_init, valstep=0.01, valfmt='%.2f')
sP2 = Slider(axP2, r'$P(\omega_2)$', P2_min, P2_max, valinit=P2_init, valstep=0.01, valfmt='%.2f')
sP3 = Slider(axP3, r'$P(\omega_3)$', P3_min, P3_max, valinit=P3_init, valstep=0.01, valfmt='%.2f')

#
# Slider P("正"|ω)
#
#                Left  Btm   Wid   Hight 
axPC1 = plt.axes([0.58, 0.95, 0.35, 0.03])
axPC2 = plt.axes([0.58, 0.90, 0.35, 0.03])
axPC3 = plt.axes([0.58, 0.85, 0.35, 0.03])

PC1_min, PC1_max, PC1_init = 0.0, 1.0, 0.3
PC2_min, PC2_max, PC2_init = 0.0, 1.0, 0.6
PC3_min, PC3_max, PC3_init = 0.0, 1.0, 0.8

sPC1 = Slider(axPC1, r'$P(CQT|\omega_1)$', PC1_min, PC1_max, valinit=PC1_init, valstep=0.01, valfmt='%.2f')
sPC2 = Slider(axPC2, r'$P(CQT|\omega_2)$', PC2_min, PC2_max, valinit=PC2_init, valstep=0.01, valfmt='%.2f')
sPC3 = Slider(axPC3, r'$P(CQT|\omega_3)$', PC3_min, PC3_max, valinit=PC3_init, valstep=0.01, valfmt='%.2f')


# color
col_Prior     = 'magenta'
col_Posterior = 'green'

#
# Prior and Posterior Bar graphs, Pattern
#                       Left  Btm   Wid   Hight 
axPrior     = plt.axes([0.05, 0.28, 0.30, 0.50])
axPosterior = plt.axes([0.65, 0.28, 0.30, 0.50])
axPattern   = plt.axes([0.40, 0.28, 0.20, 0.50])
axArrow     = plt.axes([0.18, 0.15, 0.63, 0.10])

axPattern.axis("off")
axArrow.axis("off")

# Draw Bar graph
lambda_Prior = np.zeros(3)
lambda_Posterior = np.zeros(3)
lambda_ClassConditional = np.zeros(3)
lambda_Prior = [0.5,0.4,0.1]
lambda_Posterior = [0.5,0.4,0.1]
lambda_ClassConditional = [sPC1.val, sPC2.val, sPC3.val]

bar_y_max = 1.0 
#drawBarGraph( axPrior,     "Prior",     lambda_Prior,     bar_y_max, col_Prior )
#drawBarGraph( axPosterior, "Posterior", lambda_Posterior, bar_y_max, col_Posterior )
drawArrow( axArrow )


def bayes_updating(pattern):
    """
    """
    global lambda_ClassConditional, lambda_Prior


    evidence = 0.0
    for pc, pp in zip(lambda_ClassConditional, lambda_Prior):
        if pattern==1:
            evidence += pc * pp
        else:
            evidence += (1-pc) * pp

    for n in range(3):
        if pattern==1:
            lambda_Posterior[n] = lambda_ClassConditional[n] * lambda_Prior[n] / evidence
        else:
            lambda_Posterior[n] = (1-lambda_ClassConditional[n]) * lambda_Prior[n] / evidence

    #print('(Prior):', lambda_Prior)
    #print('(Posterior):', lambda_Posterior)




def cb_updateP1(val):
    """
        Slider Event Callback Func.
    """
    global cid1, cid2, cid3, lambda_Posterior
    val1, val2, val3 = sP1.val, sP2.val, sP3.val
    p_res = 1.0 - val1
    rate = p_res / (val2 + val3)
    sP2.disconnect(cid2)    # Disconnect once not to loop calls
    sP3.disconnect(cid3)    # Disconnect once not to loop calls
    sP2.set_val(val2 * rate)
    sP3.set_val(val3 * rate)
    fig.canvas.draw_idle()
    cid2 = sP2.on_changed(cb_updateP2) # Set the Event callback func. 
    cid3 = sP3.on_changed(cb_updateP3) # Set the Event callback func. 
    lambda_Posterior = [sP1.val, sP2.val, sP3.val]

def cb_updateP2(val):
    """
        Slider Event Callback Func.
    """
    global cid1, cid2, cid3, lambda_Posterior
    val1, val2, val3 = sP1.val, sP2.val, sP3.val
    p_res = 1.0 - val2
    rate = p_res / (val1 + val3)
    sP1.disconnect(cid1)    # Disconnect once not to loop calls
    sP3.disconnect(cid3)    # Disconnect once not to loop calls
    sP1.set_val(val1 * rate)
    sP3.set_val(val3 * rate)
    fig.canvas.draw_idle()
    cid1 = sP1.on_changed(cb_updateP1) # Set the Event callback func. 
    cid3 = sP3.on_changed(cb_updateP3) # Set the Event callback func. 
    lambda_Posterior = [sP1.val, sP2.val, sP3.val]

def cb_updateP3(val):
    """
        Slider Event Callback Func.
    """
    global cid1, cid2, cid3, lambda_Posterior
    val1, val2, val3 = sP1.val, sP2.val, sP3.val
    sP1.disconnect(cid1)    # Disconnect once not to loop calls
    sP2.disconnect(cid2)    # Disconnect once not to loop calls
    p_res = 1.0 - val3
    rate = p_res / (val1 + val2)
    sP1.set_val(val1 * rate)
    sP2.set_val(val2 * rate)
    fig.canvas.draw_idle()
    cid1 = sP1.on_changed(cb_updateP1) # Set the Event callback func. 
    cid2 = sP2.on_changed(cb_updateP2) # Set the Event callback func. 
    lambda_Posterior = [sP1.val, sP2.val, sP3.val]

cid1 = sP1.on_changed(cb_updateP1) # Set the Event callback func. 
cid2 = sP2.on_changed(cb_updateP2) # Set the Event callback func. 
cid3 = sP3.on_changed(cb_updateP3) # Set the Event callback func. 


def cb_updatePC(val):
    """
        Slider Event Callback Func.
    """
    global lambda_ClassConditional
    lambda_ClassConditional = [sPC1.val, sPC2.val, sPC3.val]

sPC1.on_changed(cb_updatePC) # Set the Event callback func.
sPC2.on_changed(cb_updatePC) # Set the Event callback func.
sPC3.on_changed(cb_updatePC) # Set the Event callback func.




#
# Next Pattern Button
#                   Left  Btm   Wid   Hight 
nextax = plt.axes([0.42, 0.30, 0.15, 0.05])
button_next = Button(nextax, 'Next Pattern', color=axcolor, hovercolor='0.975')


def cb_next(event):
    """
        Next Button Event Callback Func.
    """
    global pattern_idx, lambda_Prior, lambda_Posterior, cur_pattern

    if pattern_idx < len(pattern):
        if pattern[pattern_idx] == 1:
            cur_pattern = cur_pattern + "正 "
        else:
            cur_pattern = cur_pattern + "誤 "
        print("Pattern: ", cur_pattern)

        # Draw Pattern
        drawPattern( axPattern, pattern[pattern_idx] )

        # Draw Text
        str_no = "      No. %d" % (pattern_idx+1)
        axPattern.text(0,0, str_no)

        # Posterior --> Prior
        lambda_Prior = np.copy(lambda_Posterior)

        # Bayes updating
        bayes_updating(pattern[pattern_idx])

        # Draw Bar Graph
        drawBarGraph( axPrior,     "Prior",     lambda_Prior,     bar_y_max, col_Prior )
        drawBarGraph( axPosterior, "Posterior", lambda_Posterior, bar_y_max, col_Posterior )

    
    # For next pattern
    if pattern_idx < len(pattern):
        pattern_idx += 1

    fig.canvas.draw_idle()


button_next.on_clicked(cb_next) # Event callback func. のセット


#
# Reset Button
#                   Left  Btm   Wid   Hight 
resetax = plt.axes([0.1, 0.05, 0.10, 0.05])
button_reset = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def cb_reset(event):
    """
        Reset Button Event Callback Func.
    """
    global lambda_Posterior, pattern_idx, cur_pattern
    cur_pattern = " "
    sP1.reset()
    sP2.reset()
    sP3.reset()
    sPC1.reset()
    sPC2.reset()
    sPC3.reset()
    axPrior.cla()
    axPosterior.cla()
    axPattern.cla()
    axPattern.axis("off")
    lambda_Posterior = [P1_init, P2_init, P3_init]
    pattern_idx = 0
    fig.canvas.draw_idle()

button_reset.on_clicked(cb_reset) # Event callback func. のセット

#
# Quit Button
#                  Left  Btm   Wid   Hight 
quitax = plt.axes([0.4, 0.05, 0.10, 0.05])
button_quit = Button(quitax, 'Quit', color=axcolor, hovercolor='0.975')

def cb_quit(event):
    """
        Quit Button Event Callback Func.
    """
    sys.exit()

button_quit.on_clicked(cb_quit) # Event callback func. のセット

#
# Save Button
#                  Left  Btm   Wid   Hight 
saveax = plt.axes([0.7, 0.05, 0.10, 0.05])
button_save = Button(saveax, 'Save Fig.', color=axcolor, hovercolor='0.975')

def cb_save(event):
    """
        Save Button Event Callback Func.
    """
    fig.savefig('2.BayesianUpdating.py.png', dpi=300, format='png', transparent=True)

button_save.on_clicked(cb_save) # Event callback func. のセット


plt.show()

