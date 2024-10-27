#!/usr/bin/env python3
#coding:utf-8
#
# パターンが1次元連続変数で正規分布でモデル化する最尤推定
# ユーザが指定する分布パラメタで定まる正規分布から尤度を表示
#
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import functools
import scipy.stats as st
import math
import distribution as ds
import sys
import pandas as pd
from matplotlib.widgets import Slider, Button, RadioButtons


class Likelihood:
    """
    尤度
    """
    def __init__(self):
        # Some datasets are stored in this CSV file.
        # Read the CSV file to a data frame df.
        self.df = pd.read_csv('data/score.csv')
        self.labels = self.df.columns

        # Select the first dataset as default 
        # self.label, self.num_samples, self.pattern が作成される
        self.set_pattern(self.df.columns[0])
        print('Dataset', self.df.columns[0], 'selected by Radio buttons has', self.num_samples, '[patterns]')

        #                         mu    var
        self.mu_ML, self.var_ML =  0.0,   1.0   # ML infer
        self.mu_SL, self.var_SL = 50.0, 100.0   # User selection with Slider

        #
        self.MLinfer()
        self.USRinfer()


    def set_pattern(self, label):
        """
        Select 1 dataset from some datasets stored in the 'self.df' using the dataset 'label'.
        """
        self.label = label
        # 最長のdatasetに合わせてCSV行は決まり, それ以外のdatasetの最後にはNULLが入っている．
        self.num_samples = len(self.df[label]) - self.df[label].isnull().sum()  # 実際のパターン数
        print('Dataset ', self.label, 'has', self.num_samples, '[patterns]')
        self.patterns = np.zeros(self.num_samples)
        self.patterns = self.df[self.label][:self.num_samples]


    def MLinfer(self):
        """
        選択したdatasetを用い1変量正規分布をモデルとしてパラメタ(μ,σ^2)を最尤推定
        """
        self.norm = ds.Norm() # 正規分布
        self.mu_ML, self.var_ML = self.norm.MLinfer(self.patterns) # パターンから最尤推定
        # (μ,σ2)の最尤推定値で定まる正規分布にパターンを入力した時の対数尤度値
        self.likelihood_ML = self.norm.this_log_likelihood_value(self.patterns).sum() 
        self.likelihood_ML /= self.num_samples
        print('mu_ML = %4.1f var_ML = %5.1f likelihood_ML  = %6.4e' % (self.mu_ML, self.var_ML, np.exp(self.likelihood_ML)))

    def USRinfer(self):
        """
        Sliderの値を正規分布パラメタとした分布から
        """
        self.normUSR = ds.Norm(mu=self.mu_SL, var=self.var_SL) # Sliderの(μ,σ2)から正規分布を作成
        # Sliderの(μ,σ2)で定まる正規分布にパターンを入力した時の対数尤度値
        self.likelihood_USR = self.normUSR.this_log_likelihood_value(self.patterns).sum() 
        self.likelihood_USR /= self.num_samples
        print('mu_SL = %4.1f var_SL = %5.1f likelihood_USR = %6.4e' % (self.mu_SL, self.var_SL, np.exp(self.likelihood_USR)))





class Application(Likelihood):
    def __init__(self):
        # 描画範囲と解像度
        self.rangeMu  = [40,100]
        self.eps      = 1.e-2
        self.rangeVar = 300
        self.numGrid  = 128     # Heatmap 128x128

        # 2D 分布表示用
        self.x_min, self.x_max =  self.rangeMu[0], self.rangeMu[1]  # 推定分布(Norm)表示用 (X軸)
        self.y_min, self.y_max = -0.02,   0.10                      # 推定分布(Norm)表示用 (Y軸)
        self.delta_x = (self.x_max - self.x_min) / 1000             # X方向には1000点描画

        # 推定点の色
        self.col_ML    = 'green'
        self.col_USR   = 'magenta'
        #'red' 'cyan' 'magenta'

        # Slider, Button, RadioButton のfaceの色
        self.face_color = 'lightgoldenrodyellow'

        # 図の準備
        self.fig = plt.figure(figsize=(12,8))

        # Layout
        #                               Left  Btm   Wid   Hight 
        self.axLikelihood   = plt.axes([0.06, 0.40, 0.40, 0.52])    # 尤度分布Heatmap
        self.ax2D           = plt.axes([0.56, 0.40, 0.40, 0.52])    # PatternとUser指定正規分布
        self.axRadioButtons = plt.axes([0.70, 0.02, 0.20, 0.30])    # Radio Buttons
        self.axMu           = plt.axes([0.06, 0.15, 0.50, 0.03], facecolor=self.face_color) # Slider
        self.axVar          = plt.axes([0.06, 0.10, 0.50, 0.03], facecolor=self.face_color) # Slider
        self.quitax         = plt.axes([0.06, 0.02, 0.10, 0.04])    # Quit  button
        self.saveax         = plt.axes([0.30, 0.02, 0.10, 0.04])    # Save  button
        self.resetax        = plt.axes([0.50, 0.02, 0.10, 0.04])    # Reset button
        self.valueax        = plt.axes([0.06, 0.20, 0.50, 0.20])    # Likehood Value
        #self.radio_buttons.ax.set_position([0.05,0.5,0.2,0.3])

        self.axRadioButtons.set_facecolor(self.face_color)
        self.axLikelihood.set_xlabel(r'$\mu$')
        self.axLikelihood.set_ylabel(r'$\sigma^2$')
        self.axLikelihood.set_title('Likelihood')
        self.ax2D.set_xlabel(r'$x$')
        self.ax2D.set_ylabel(r'$p(x)$')
        self.ax2D.set_title('Patterns and your Model')

        super().__init__()

        print('>> Ready to start')


    def draw_likelihood_heatmap(self, axis):
        """ Draw a Probability density of Likelihood with a heatmap """
        # 表示範囲
        axis.set_xlim(self.rangeMu[0],            self.rangeMu[1])
        axis.set_ylim(self.rangeVar/self.numGrid, self.rangeVar  )
        # (x,y)座標群
        x, y = np.meshgrid( np.linspace(self.rangeMu[0],           self.rangeMu[1], self.numGrid), 
                            np.linspace(self.rangeVar/self.numGrid,self.rangeVar,   self.numGrid))
        # 確率密度
        p = self.norm.likelihood(self.patterns, x, y)
        # 分布表示
        axis.pcolormesh(x, y, p, cmap=plt.cm.hot)

        # 尤度表示
        self.valueax.cla()
        self.valueax.axis("off")
        self.valueax.text(0.0,0.5, 'ML(Maximum Likelihood) value', fontsize='18')
        self.valueax.text(0.0,0.2, 'User select likelihood value', fontsize='18')
        str_MLlikelihood  = '= %6.4e' % np.exp(self.likelihood_ML)
        str_USRlikelihood = '= %6.4e' % np.exp(self.likelihood_USR)
        self.valueax.text(0.7,0.5, str_MLlikelihood,  fontsize='18')
        self.valueax.text(0.7,0.2, str_USRlikelihood, fontsize='18')
    

    def draw_point(self, axis, point2D, color):
        """ Draw Point on graph """
        axis.scatter(point2D[0], point2D[1], c=color, marker='o')


    def drawLikelihood(self, axis):
        """
            尤度分布の表示 (中央)
        """
        self.draw_likelihood_heatmap(axis) # Draw ML
        self.draw_point(axis, [self.mu_ML,self.var_ML], self.col_ML)
        self.draw_point(axis, [self.mu_SL,self.var_SL], self.col_USR)
        axis.legend(('ML', 'Your Choice'), loc='upper left')


    def draw2Ddistribution(self, axis):
        """
            推定したクラス依存確率密度の表示(右)
        """
        axis.set_xlim(self.x_min, self.x_max)
        axis.set_ylim(self.y_min, self.y_max)
        axis.grid()

        # 入力パターン表示
        posY = np.zeros(self.num_samples)
        axis.scatter(self.patterns, posY, c='red', marker="o") # 入力パターン表示
        x_data = np.arange(self.x_min, self.x_max, self.delta_x) # x軸用データ

        # User推定の分布
        axis.plot(x_data, self.normUSR.pdf(x_data), c=self.col_USR)


        axis.legend(('Your Choice', 'Patterns'))


    #
    # Callback functions
    #
    def cb_radio(self, label):
        """
            Callback Function of Radio Buttons
        """
    
        self.set_pattern(label)   # label(d1〜d10) から１つのdatasetを選択 
        print('Dataset', label, 'selected by Radio buttons has', self.num_samples, '[patterns] [cb_radio()]')
    
        # 新たなdatasetで最尤推定し直し，その分布の尤度を算出
        self.MLinfer()
        # UserがSliderで指定した分布の尤度を算出
        self.mu_SL  = self.sMu.val
        self.var_SL = self.sVar.val
        self.USRinfer()


        # Redraw Likelihood (Middle graph)
        self.drawLikelihood(self.axLikelihood)
        # Redraw 2D distribution (Right graph)
        self.ax2D.cla()
        self.draw2Ddistribution(self.ax2D)
        self.fig.canvas.draw()


    def cb_changeMuVar(self, val):
        """
            Callback Function of Sliders Mu, Var 
        """
    
        # UserがSliderで指定した分布の尤度を算出
        self.mu_SL  = self.sMu.val
        self.var_SL = self.sVar.val
        self.USRinfer()
    
        # Redraw Likelihood (Middle graph)
        self.drawLikelihood(self.axLikelihood)
        # Redraw 2D distribution (Right graph)
        self.ax2D.cla()
        self.draw2Ddistribution(self.ax2D)
        self.fig.canvas.draw()
    
    def cb_quit(self, event):
        """
            Quit Button Event Callback Func.
        """
        sys.exit()

    def cb_save(self, event):
        """
            Save Button Event Callback Func.
        """
        print('Save')
        self.fig.savefig('Chap5.MLinference.py.png', dpi=300, format='png', transparent=True)

    def cb_reset(self, event):
        """
            Reset Button Event Callback Func.
        """
        self.sMu.reset()
        self.sVar.reset()

    def main_proc(self):
        """
        Main Procedure
        """

        # Draw Likelihood Heatmap
        self.drawLikelihood(self.axLikelihood)

        # Draw Patterns & User specified Nomal distribution
        self.draw2Ddistribution(self.ax2D)

        # Draw Radio Buttons
        self.radio_buttons = RadioButtons(self.axRadioButtons, self.labels, active=0 )
        #self.radio_buttons.ax.set_position([0.05,0.5,0.2,0.3])
        self.axRadioButtons.text(0.5,0.5,'Select Dataset')

        self.radio_buttons.on_clicked(self.cb_radio) # Event callback func. のセット

        # Draw Slider (μ, σ^2)
        self.sMu  = Slider(self.axMu,  r'$\mu$'     , self.rangeMu[0], self.rangeMu[1], valinit=self.mu_SL)
        self.sVar = Slider(self.axVar, r'$\sigma^2$', self.eps,        self.rangeVar,   valinit=self.var_SL)

        self.sMu.on_changed(self.cb_changeMuVar)  # Event callback func. のセット
        self.sVar.on_changed(self.cb_changeMuVar) # Event callback func. のセット

        # Draw Quit Button
        self.button_quit = Button(self.quitax, 'Quit', color=self.face_color, hovercolor='0.975')

        self.button_quit.on_clicked(self.cb_quit) # Event callback func. のセット

        # Draw Save Button
        self.button_save = Button(self.saveax, 'Save', color=self.face_color, hovercolor='0.975')

        self.button_save.on_clicked(self.cb_save) # Event callback func. のセット

        # Draw Reset Button
        self.button_reset = Button(self.resetax, 'Reset', color=self.face_color, hovercolor='0.975')

        self.button_reset.on_clicked(self.cb_reset) # Event callback func. のセット



###
### ここからMain
###
if __name__ == '__main__':

	# Instanciation
	app = Application()

	# Main
	app.main_proc()

	# Plot
	plt.show()


