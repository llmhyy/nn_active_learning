#TODO move the code here

import math
import random


def balancingPoint(points,gradient,length_added):
	
	outputX=[]
	iter=0
	flag=False
	while True:
		for i in range(len(points)):
			g_total = 0
			grad = 0
			for k in range(len(points[0])):
			    grad += gradient[i][k] * gradient[i][k]
			g_total = math.sqrt(grad)
			tmpList=[]
			step=random.random()*5
			for j in range(len(points[i])):
				tmpValue=points[i][j]+gradient[i][j]*(step/g_total)
				tmpList.append(tmpValue)
			outputX.append(tmpList)
			iter+=1
			if (iter==length_added):
				flag=True
				break
		if (flag==True):
			break
	
	print ("points added \n",outputX)
	return outputX

# label_0=[[1,2]]
# label_1=[[1,2],[3,4],[5,6],[7,8],[9,10]]

# gra0=[[0.1,0.2]]
# gra1=[]
# balancingPoint(label_0,label_1,gra0,gra1)