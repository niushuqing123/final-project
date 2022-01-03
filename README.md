# 太极图形课S1- final project

这是太极图形第一季的大作业

## 背景简介

做这个东西主要是因为对物理引擎比较感兴趣，正好学了sph，再加上自己以前写过的小玩意，就尝试把他们结合起来，可能物理上并不太正确，但是效果还可以接受。
对于sph部分，由于自己实现的时候遇到了许多的障碍，所以最终还是借用不少了助教sph的写法。

## 成功效果展示
这里可以展示这份作业（项目）run起来后的可视化效果，可以让其他人更直观感受到你的工作

## 整体结构

结构嘛。。。
对于python我习惯于把代码都写在一个文件里了，简单描述一下，第一大部分是sph的计算，第二大部分是耦合进来的圆形，第三大部分是弹簧质点系统，之后有不少函数是对于粒子的增删改，最后一大部分是交互的部分，由于我希望把它做的完善一些，更像个沙盒，所以交互代码写了不少
```
-README.MD
-xixi.py
```
## 操作手册

p：暂停开关

shift+左键可以选择一个圆形

鼠标左键按住：可以吸引一个被选择的圆形

鼠标右键点击：创建一个自动链接弹簧的圆形（链接距离，弹簧长度可调）

shift+E：之后单独生成的圆形弹簧的长度增大

ctrl+E：之后单独生成的圆形弹簧的长度减小

shift+D：之后单独生成的圆形弹簧的探测距离增大

ctrl+D：之后单独生成的圆形弹簧的探测距离减小

F：切换固定状态，可以将活动的圆形设置为固定，将固定的圆设置为可活动

H：生成一个由圆组成的弹性体矩形，参数可调

M+鼠标右键：生成一个构造好的弹性轮子（参数可调）

N：让上一个生成的轮子开始转动

shift+X：轮子中，圆的链接距离增大

ctrl+X：轮子中，圆的链接距离减小

shift+0：构造轮子的圆形半径增大

ctrl+0：构造轮子的圆形半径减小

shift+C：之后生成的圆形半径增大（除了轮子上的圆）

ctrl+C：之后生成的圆形半径减小

按住J+鼠标右键：在屏幕上点击两点，生成一个首位被固定的圆形链子（参数可调）



R:清除所有的圆形，包括弹性物体

O：生成一个液体矩形

ctrl+1：打开一号水龙头

按住1+鼠标右键：在屏幕上点击两点，设置水龙头姿态和流速

ctrl+2：打开二号水龙头

按住2+鼠标右键：在屏幕上点击两点，设置水龙头姿态和流速

按住L+鼠标右键：在屏幕上点击两点，画出一条固定的边界

B:所有作为边界的粒子变为流体

ctrl+B：上一个画出来的线变为流体

G:切换之后的液体为重力向下，或者反重力的液体


z：删除上一个圆

ctrl+z：开始清除流体粒子

shift+z：删除上一个用边界粒子画出来的线

shift+w：弹性矩形宽度增加

ctrl+w：弹性矩形宽度减小

shift+s：弹性矩形高度增加

ctrl+s：弹性矩形高度减小


o：生成一个液体矩形


shift+q：弹性矩形宽度增加

ctrl+q：弹性矩形宽度减小

shift+a：弹性矩形高度增加

ctrl+a：弹性矩形高度减小



## 运行方式
直接运行xixi.py即可
