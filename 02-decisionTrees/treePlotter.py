#coding:utf-8

import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = "sawtooth", fc="0.8")
leafNode = dict(boxstyle = 'round4', fc="0.8")
arrow_args = dict(arrowstyle = '<-')

# 在父子节点间填充路径文本信息。
def plotMidText(centerPt, parentPt, txtString):
	xMid = (parentPt[0] - centerPt[0] / 2.0 + centerPt[0])
	yMid = (parentPt[1] - centerPt[1] / 2.0 + centerPt[1])
	createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
	numLeafs = getNumLeafs(myTree)
	depth = getTreeDepth(myTree)
	firstStr = myTree.keys()[0]
	centerPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
	plotMidText(centerPt, parentPt, nodeTxt)
	plotNode(firstStr, centerPt, parentPt, decisionNode)
	secondDict = myTree[firstStr]
	plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
	for key in secondDict.keys():
		if (type(secondDict[key]).__name__ == 'dict'):
			plotTree(secondDict[key], centerPt, str(key))
		else:
			plotTree.xOff = plotTree.xOff + 1.0 /plotTree.totalW
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), centerPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), centerPt, str(key))
	plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	#matplotlib.pyplot.annotate(*args, **kwargs) Annotate the point xy with text s.
	createPlot.ax1.annotate(
		nodeTxt, 											# 注解文本
		xy = parentPt, 									# 箭头起始坐标点位置
		xycoords = 'axes fraction', 		# 坐标点小数。
		xytext= centerPt,  							# 文本坐标点位置
		textcoords = 'axes fraction', 	# 文本坐标点小数
		va = 'center', 									# vertical-align
		ha = 'left', 										# horizontal-align
		bbox = nodeType, 								# border-box
		arrowprops = arrow_args 				# 箭头样式
	)

def createPlot(inTree):
	fig = plt.figure(1, facecolor = 'cornsilk') # facecolor 背景色
	fig.clf() 																	# 清空绘图区 Clear the current figure.
	# createPlot.ax1 全局变量。 python 中所有的变量默认都是全局有效的。
	axprops = dict(xticks = [], yticks = [])
	createPlot.ax1 = plt.subplot(111, frameon = False, **axprops) # 返回 matplotlib.axes.AxesSubplot
	plotTree.totalW = float(getNumLeafs(inTree))
	plotTree.totalD = float(getTreeDepth(inTree))/2
	plotTree.xOff = -0.5/plotTree.totalW 
	plotTree.yOff = 1.0
	plotTree(inTree, (0.5, 1.0), '')
	# plotNode(U'决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
	# plotNode(U'叶节点', (0.8, 0.4), (0.7, 0.8), leafNode)
	plt.show()

# 获取树的叶子结点个数
def getNumLeafs (myTree):
	numLeafs = 0
	firstStr = myTree.keys()[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict': 
			numLeafs += getNumLeafs(secondDict[key])
		else:
			numLeafs += 1
	return numLeafs

# 判断是否是树
def isTree(tree):
	if not tree or type(tree).__name__ != 'dict':
		return False
	return True

# 判断是否是叶子节点
def isLeafNode(tree):
	if isTree(tree) and type(tree[tree.keys()[0]]).__name__ != 'dict':
		return True
	return False

# 获取所有子树
def getAllChildTrees(tree):
	if isTree(tree) and not isLeafNode(tree):
		return tree[tree.keys()[0]]
	return {}

# 获取树的深度。 确定图的高度。
def getTreeDepth(tree):
	if not isTree(tree): 
		return 0
	if isLeafNode(tree): 
		return 1
	maxDepth = 0
	allChildTrees = getAllChildTrees(tree)
	for childTreeKey in allChildTrees.keys():
		# getTreeDepth({childTreeKey: allChildTrees[childTreeKey]}) 保证是真正的子树，
		# 而getTreeDepth(allChildTrees[childTreeKey]) 就不是真正的树了。
		depth = 1 + getTreeDepth({childTreeKey: allChildTrees[childTreeKey]})
		print {childTreeKey: allChildTrees[childTreeKey]}
		print depth
		if maxDepth < depth :
			maxDepth = depth
	return maxDepth

# 获取叶子节点个数。 确定图的宽度。
def getNumLeafs(tree):
	if not isTree(tree):
		return 0
	if isLeafNode(tree):
		return 1
	numLeafs = 0
	allChildTrees = getAllChildTrees(tree)
	for childKey in allChildTrees:
		numLeafs += getNumLeafs({childKey: allChildTrees[childKey]})
	return numLeafs

# 先将树存储起来，避免每次都从数据中创建树。
def retrieveTree(i):
	listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}]
	return listOfTrees[i]