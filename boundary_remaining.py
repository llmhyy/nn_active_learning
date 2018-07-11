#TODO move the code here


import random


def balancingPoint(label_0,label_1,label_0_gradient,label_1_gradient):
	length0=len(label_0)+0.0
	length1=len(label_1)+0.0
	outputX=[]
	
	if length0/length1<0.5:
		while length0/length1<0.5:
			for i in range(len(label_0)):
				tmpList=[]
				step=random.random()
				for j in range(len(label_0[i])):
					tmpValue=label_0[i][j]+label_0_gradient[i][j]*step
					tmpList.append(tmpValue)
				outputX.append(tmpList)

			length0=len(label_0)+len(outputX)+0.0
			length1=len(label_1)+0.0


	elif length1/length0<0.5:
		while length1/length0<0.5:
			for i in range(len(label_1)):
				tmpList=[]
				step=random.random()
				for j in range(len(label_1[i])):
					tmpValue=label_1[i][j]+label_1_gradient[i][j]*step
					tmpList.append(tmpValue)
				outputX.append(tmpList)

			length1=len(label_1)+len(outputX)+0.0
			length0=len(label_0)+0.0
	print (outputX)
	return outputX

# label_0=[[1,2]]
# label_1=[[1,2],[3,4],[5,6],[7,8],[9,10]]

# gra0=[[0.1,0.2]]
# gra1=[]
# balancingPoint(label_0,label_1,gra0,gra1)