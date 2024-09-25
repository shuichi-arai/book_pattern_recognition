#!/usr/bin/env python3
#coding:utf-8

import matplotlib.pyplot as plt
from matplotlib import font_manager

# InstallされているTrueTypeフォント名のリストを作成
fonts = set([f.name for f in font_manager.fontManager.ttflist])

num_fonts = len(fonts) # Font数

print('InstallされているFontは', num_fonts, '個あります．')
 
# 描画領域
fig=plt.figure(figsize=(16,16))
 
# フォントの表示
num_draw = 70
for i, font in enumerate(fonts):
    print("font", i, font)

    plt.text((i//num_draw)*0.3, (i%num_draw)*3, f"日本語：{font}", fontname=font)

plt.ylim(0, len(fonts))
plt.axis("off")
    
plt.show()


