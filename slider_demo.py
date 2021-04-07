from numpy import *
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider,RadioButtons


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
om2 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor) # 第二个slider的位置

som1 = Slider(om1, r'$\omega_1$', 1, 30.0, valinit=3) # 产生第一slider
som2 = Slider(om2, r'$\omega_2$', 1, 30.0, valinit=5) # 产生第二slider


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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

t = np.arange(0.0, 1.0, 0.001)
a0 = 5
f0 = 3
s = a0 * np.sin(2 * np.pi * f0 * t)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
l, = plt.plot(t, s, lw=2)

slider_bkd_color = 'lightgoldenrodyellow'
ax_freq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=slider_bkd_color)
ax_amp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=slider_bkd_color)

# define the values to use for snapping
allowed_amplitudes = np.concatenate([np.linspace(.1, 5, 100), [6, 7, 8, 9]])

# create the sliders
samp = Slider(
    ax_amp, "Amp", 0.1, 9.0,
    valinit=a0, valstep=allowed_amplitudes,
    color="green"
)

sfreq = Slider(
    ax_freq, "Freq", 0, 10*np.pi,
    valinit=2*np.pi, valstep=np.pi,
    initcolor='none'  # Remove the line marking the valinit position.
)


def update(val):
    amp = samp.val
    freq = sfreq.val
    l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    fig.canvas.draw_idle()


sfreq.on_changed(update)
samp.on_changed(update)

ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(ax_reset, 'Reset', color=slider_bkd_color, hovercolor='0.975')


def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)


plt.show()