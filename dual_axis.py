# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:33:04 2021

@author: AliceChen
"""

import matplotlib.pyplot as plt
import pandas as pd
from numpy import *
import seaborn as sns
from matplotlib.widgets import Slider,RadioButtons

def cut_function(value, cut_into, _):
    if _ == 'qcut':
        return pd.qcut(value, cut_into)
    else:
        return pd.cut(value, cut_into)
    
    
N = 6
feature = 'AtoA'

data = pd.read_csv('D:/Data/Data_new.csv',header=1)

data[feature+'_cut'] = cut_function(data[feature], N, 'qcut')


x = list(data[feature+'_cut'].value_counts(sort=False).index.astype(str))


# Count
y = pd.value_counts(data[feature+'_cut']).values

# NG ratio
y1 = data.groupby(feature+'_cut').param_value.mean().values 

fig, ax1 = plt.subplots(figsize=(13,8))

ax2 = ax1.twinx()
ax1.bar(x, y)
sns.lineplot(x = x, y=y1  , marker = 'o', color = 'red', ax = ax2  )

ax1.set_xlabel('X data')
ax1.set_ylabel('Counts', color='g')
ax2.set_ylabel('NG Ratio', color='b')
fig.autofmt_xdate(rotation=-25, ha='left')


axcolor = 'lightgoldenrodyellow'  # slider的颜色

om1= plt.axes([0.25, 0, 0.65, 0.03], facecolor=axcolor) # 第一slider的位置

som1 = Slider(om1, r'$\omega_1$', 1, 10, valinit=6,valstep=1,valfmt="%i") # 产生第一slider
fig.autofmt_xdate(rotation=-25, ha='left')

lb = 'qcut'
def update(val):
    s1 = int(som1.val)
    print(s1)
    data[feature+'_cut'] = cut_function(data[feature], s1, lb)
    x = list(data[feature+'_cut'].value_counts(sort=False).index.astype(str))
    y = pd.value_counts(data[feature+'_cut']).values
    y1 = data.groupby(feature+'_cut').param_value.mean().values 
    ax1.clear()
    ax1.bar(x, y)
    ax2.clear()
    sns.lineplot(x = x, y=y1  , marker = 'o', color = 'red', ax = ax2  )
    ax1.set_xticklabels(x, rotation=-25 )
    fig.autofmt_xdate(rotation=-25, ha='left')
    fig.canvas.draw()
    # fig.canvas.draw_idle()
som1.on_changed(update)


cc = plt.axes([0.01, 0.5, 0.08, 0.08], facecolor=axcolor)
radio = RadioButtons(cc, ('qcut', 'cut'), active=0)


def colorfunc(label):
    global lb 
    lb = label
    s1 = int(som1.val)
    data[feature+'_cut'] = cut_function(data[feature], s1,  label)
    x = list(data[feature+'_cut'].value_counts(sort=False).index.astype(str))
    y = pd.value_counts(data[feature+'_cut']).values
    y1 = data.groupby(feature+'_cut').param_value.mean().values 
    ax1.clear()
    ax1.bar(x, y)
    ax2.clear()
    sns.lineplot(x = x, y=y1  , marker = 'o', color = 'red', ax = ax2  )
    ax1.set_xticklabels(x, rotation=-25 )
    fig.autofmt_xdate(rotation=-25, ha='left')
    fig.canvas.draw()   
    print(label)
    # fig.canvas.draw_idle()
radio.on_clicked(colorfunc)





plt.show()
