#TODO move the code here

import math
import random
import testing_function
import formula


def balancingPoint(inflag, points, gradient, length_added, formu, category):
	
	times = 100
	outputX=[]
	iter=0
	flag=False
	count = 0
	wrong = 0
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

			if category == formula.POLYHEDRON:
				pointflag = testing_function.polycircleModel(formu[0], formu[1], tmpList)
			elif category == formula.POLYNOMIAL:
				pointflag = testing_function.polynomialModel(formu[:-1],tmpList,formu[-1])
			count += 1
			if inflag==1 and pointflag:
				times += 1
				wrong += 1
				if times>100:
					flag = True
					break
				continue
			if inflag==0 and not pointflag:
				times += 1
				wrong += 1
				if times>100:
					flag = True
					break
				continue
			outputX.append(tmpList)

			iter+=1
			if (iter==length_added):
				flag=True
				break
		if (flag==True):
			break
	
	print ("points added \n",outputX)
	print ("Boundary remaining accuracy: ", float((count-wrong)/count))
	return outputX

# label_0=[[1,2]]
# label_1=[[1,2],[3,4],[5,6],[7,8],[9,10]]

# gra0=[[0.1,0.2]]
# gra1=[]
# balancingPoint(label_0,label_1,gra0,gra1)