import csv
import random
import math

with open('train.csv', 'wb') as csvfile:
	with open('test.csv', 'wb') as csvfile2:
		train = csv.writer(csvfile)
		test = csv.writer(csvfile2)
		for i in range(1000):
			x = random.uniform(0,11)
			y = random.uniform(-10,10)
			if i<700:
				if (y>math.log(x)):
					train.writerow([0.0,x ,y])
				else:
					train.writerow([1.0,x ,y])
			else:
				if (y>math.log(x)):
					test.writerow([0.0,x ,y])
				else:
					test.writerow([1.0,x ,y])