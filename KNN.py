from numpy import *
from os import listdir
import operator

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels

def classify0(inX, dataSet, labels, k):     #输入向量，数据集，标签，K近邻
	dataSetSize = dataSet.shape[0]          #行数
	diffMat = tile(inX, (dataSetSize,1))-dataSet    #复制inX dataSetSize行1列
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis = 1) #axis=0代表往down，1代表cross
	distances = sqDistances ** 0.5
	sortedDistIndicies = distances.argsort()#从小到大排序的序号
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) #从python字典读值，默认为零
	sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]

def file2matrix(file2name):
	fr = open(file2name)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = zeros((numberOfLines, 3))
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		line = line.strip()               #除去首位指定字符，默认换行符
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))#-1表示最后一列元素
		index += 1
	return returnMat,classLabelVector

def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m,1))
	normDataSet = normDataSet / tile(ranges, (m,1))
	return normDataSet, ranges, minVals

def datingClassTest():
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m * hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
				datingLabels[numTestVecs:m], 3)
		print("the classifier came back with: %d, the real answer is: %d"\
				%(classifierResult,datingLabels[i]))
		if(classifierResult != datingLabels[i]): 
			errorCount += 1.0
	print("the total error rate is: %f" %(errorCount / float(numTestVecs)))

def classifyPerson():
	resultList = ['not at all', 'in small does', 'in large does']
	percentTats = float(input("percentage of time spent playing video games"))
	ffMiles = float(input("frequent filter miles earned per year"))
	icecream = float(input("liters of icecream consumed per year"))
	datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, icecream])
	classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels,3)
	print("You will probably like this person: ",resultList[classifierResult - 1])

def img2vector(filename):
	returnVect = zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	return returnVect

def handwritingClassTest():
	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m, 1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s' %fileNameStr)
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' %fileNameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print("the classifier came back with: %d,the real answer is: %d" %(classifierResult,classNumStr))
		if(classifierResult != classNumStr): errorCount += 1.0
	print("\nthe total number of errors is: %d" %errorCount)
	print("\nthe total error rate is: %f" %(errorCount/float(mTest)))