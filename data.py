import csv
import random
import math

with open('train.csv', 'wb') as csvfile:
	with open('test.csv', 'wb') as csvfile2:
		train = csv.writer(csvfile)
		test = csv.writer(csvfile2)
		for i in range(1000):
			x = random.uniform(-25,25)
			y = random.uniform(-12.5,12.5)
			if i<700:
				if ((x-12.5)*(x-12.5)+y*y<100 or (x+12.5)*(x+12.5)+y*y<100):
					train.writerow([0.0,x ,y])
				else:
					train.writerow([1.0,x ,y])
			else:
				if ((x-12.5)*(x-12.5)+y*y<100 or (x+12.5)*(x+12.5)+y*y<100):
					test.writerow([0.0,x ,y])
				else:
					test.writerow([1.0,x ,y])