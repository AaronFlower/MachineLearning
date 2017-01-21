#-*-coding:utf-8-*-
'''
绘制 classify0
'''
import kNN
import matplotlib
import matplotlib.pyplot as plt 

dataSet, labels = kNN.createDataSet()
aDataSet = dataSet[0:2]
bDataSet = dataSet[2:]

fig = plt.figure()
ax = fig.add_subplot(111)
typeA = ax.scatter(aDataSet[:, 0], aDataSet[:, 1], c='r') # 画点并标注颜色为红色
typeB = ax.scatter(bDataSet[:, 0], bDataSet[:, 1]) # 默认为 blue
ax.legend([typeA, typeB], ['Type A', 'Type B'], loc="upper left")
ax.axis([-0.2, 1.2, -0.2, 1.2]) # 指定 x,y 坐标的起始。
plt.xlabel('x --')
plt.ylabel('y --')
plt.show()
