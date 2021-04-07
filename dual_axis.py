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
feature = 'TUCtoTUC'

data = pd.read_csv('D:/Data/OPI_Particle/CF_Data_new.csv',header=1)

data[feature+'_cut'] = cut_function(data[feature], N, 'qcut')

# sns.countplot(x='ARCOtoARCO', hue='param_value', data=data, ax = axs )


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
# 


#%%

def spv(omega1,omega2):
    x=linspace(0,20*pi,200)
    y1=sin(omega1*x)
    y2=sin(omega2*x)
    return (x,y1+y2)


fig, ax = plt.subplots()  
plt.subplots_adjust(bottom=0.2,left=0.3) #调整子图间距

x,y=spv(2,3)                             # 初始化函数
l,=plt.plot(x,y,color='red')             # 画出该条曲线


axcolor = 'lightgoldenrodyellow'  # slider的颜色
om1= plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor) # 第一slider的位置

som1 = Slider(om1, r'$\omega_1$', 1, 30.0, valinit=3) # 产生第一slider


def update(val):
    s1 = som1.val
    s2 = som2.val
    x,y=spv(s1,s2)
    l.set_ydata(y)
    l.set_xdata(x)
    fig.canvas.draw_idle()
som1.on_changed(update)
som2.on_changed(update)


cc = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(cc, ('red', 'blue', 'green'), active=0)


def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()
radio.on_clicked(colorfunc)


plt.show()


#%%
data['ARCOtoARCO'] = pd.qcut(data['ARCOtoARCO'], 13)

#%%

fig, axs = plt.subplots(figsize=(13,8))
sns.countplot(x='ARCOtoARCO', hue='param_value', data=data, ax = axs )

b = list(data['ARCOtoARCO'].value_counts(sort=False).index.astype(str))
ax2 = axs.twinx() 
sns.lineplot(x = b, y=data.groupby("ARCOtoARCO").param_value.mean().values  , marker = 'o', color = 'red', ax = ax2  )

plt.xlabel('ARCOtoARCO', size=10, labelpad=10)
plt.ylabel('Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=15)
# plt.xticks(rotation=-25, ha='left')

fig.autofmt_xdate(rotation=-25, ha='left')
plt.legend(['OK', 'NG'], loc='upper right', prop={'size': 15})
plt.title('Count of NG in {} Feature'.format('ARCOtoARCO'), size=15, y=1.05)


plt.show()
#%%
import numpy as np


sns.histplot(data['ARCOtoARCO'], bins=5)

a= pd.qcut(data['ARCOtoARCO'], 10)

pd.value_counts(a)



N=5
nn = (data['ARCOtoARCO'].max() - data['ARCOtoARCO'].min() )/N
aa = np.arange( data['ARCOtoARCO'].min() , data['ARCOtoARCO'].max() , nn )
