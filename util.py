import csv
import tensorflow as tf

def calculateAccuracy(y, set_Y, b):
    test_correct = []
    test_wrong = []
    train_correct = []
    train_wrong = []
    for i in range(len(set_Y)):
        if (b):
            print(i, " predict:", y[i][0], " actual: ", set_Y[i][0])

        if y[i][0] > 0.5 and set_Y[i][0] == 1:
            test_correct.append(y[i])
        elif y[i][0] > 0.5 and set_Y[i][0] == 0:
            test_wrong.append(y[i])
        elif y[i][0] <= 0.5 and set_Y[i][0] == 0:
            test_correct.append(y[i])
        else:
            test_wrong.append(y[i])
    # print (test_correct)
    # print (test_wrong)

    print(len(test_correct) / float(len(test_correct) + len(test_wrong)))


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden fully connected layer with 256 neurons
    ##x0=tf.nn.batch_normalization(x,mean=0.01, variance=1,offset=0,scale=1,variance_epsilon=0.001)
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # layer1_out = tf.sigmoid(layer_1)

    # Hidden fully connected layer with 256 neurons
    # layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # layer2_out = tf.sigmoid(layer_2)

    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return tf.nn.sigmoid(out_layer)


def preprocess(train_set_X, train_set_Y, test_set_X, test_set_Y):
    # read training data
    with open('train.csv', 'r+') as csvfile:
        with open('train_next.csv', 'w+') as file:
            i=0
            spamreader = csv.reader(csvfile)
            writer = csv.writer(file)
            for row in spamreader:
                if(i>70):
                    break
                i+=1

                writer.writerow(row)
                # writer.writerow([1-float(row[0]),float(row[1])+0.01,float(row[2])+0.01])


        file.close()

    # read testing data
    with open('test.csv', 'r+') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            # print(row)
            l = [0, 0]
            l[0] = float(row[1])
            l[1] = float(row[2])
            # print(l)
            test_set_X.append(l)
            if (row[0] == '1.0'):
                test_set_Y.append([1])
            else:
                test_set_Y.append([0])

    # read testing data
    with open('train.csv', 'r+') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            l = [0, 0]
            l[0] = float(row[1])
            l[1] = float(row[2])
            # print(l)
            train_set_X.append(l)
            if (row[0] == '1.0'):
                train_set_Y.append([1])
            else:
                train_set_Y.append([0])
