import csv
import random

with open('train.csv', 'wb') as csvfile:
	with open('test.csv', 'wb') as csvfile2:
		train = csv.writer(csvfile)
		test = csv.writer(csvfile2)
		for i in range(1000):
			x = random.uniform(-1000,1000)
			y = random.uniform(-1000,1000)
			if i<200 :
				y = random.uniform(50-x*x, 500-x*x)
				train.writerow([0.0, x, y])
			elif i<700:
				if (x*x+y*y<500 and x*x+y*y>50):
					train.writerow([0.0,x ,y])
				else:
					train.writerow([1.0,x ,y])
			else:
				if (x*x+y*y<500 and x*x+y*y>50):
					test.writerow([0.0,x ,y])
				else:
					test.writerow([1.0,x ,y])