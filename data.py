import csv
import random

with open('train.csv', 'wb') as csvfile:
    with open('test.csv', 'wb') as csvfile2:
        train = csv.writer(csvfile)
        test = csv.writer(csvfile2)
        for i in range(1000):
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            flag = y > x * x * x + x * x + x
            if i < 700:
                ##if ((x-12.5)*(x-12.5)+y*y<100 or (x+12.5)*(x+12.5)+y*y<100):
                if (flag):
                    train.writerow([0.0, x, y])
                else:
                    train.writerow([1.0, x, y])
            else:
                ##if ((x-12.5)*(x-12.5)+y*y<100 or (x+12.5)*(x+12.5)+y*y<100):
                if (flag):
                    test.writerow([0.0, x, y])
                else:
                    test.writerow([1.0, x, y])
