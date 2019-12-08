import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# learning rate
alpha = 0.01
# weight[0] ->x 
# weight[1] ->y 
# weight[2] ->bias 

def sigmoid(weight, x , y):
    # (x, y) means the data point xi
    num =  weight[0]*x + weight[1]*y + weight[2] 
    try:
        result = 1/(1+math.exp(-1*num))
    except OverflowError:
        result = 0.000001
    return result


def gradient_square(weight, result, datax, datay):
    weight_after = [0,0,0]
    for i in range(len(datax)):
        sigmoid_value = sigmoid(weight, datax[i],datay[i])
        weight_after[0] += (result - sigmoid_value)*sigmoid_value*(1-sigmoid_value)*datax[i]
        weight_after[1] += (result - sigmoid_value)*sigmoid_value*(1-sigmoid_value)*datay[i]
        weight_after[2] += (result - sigmoid_value)*sigmoid_value*(1-sigmoid_value)
    return weight_after

def gradient_entropy(weight, result, datax, datay): # need to divide by n
    weight_after = [0,0,0]
    for i in range(len(datax)):
        sigmoid_value = sigmoid(weight, datax[i],datay[i])
        weight_after[0] += (result - sigmoid_value)*datax[i]
        weight_after[1] += (result - sigmoid_value)*datay[i]
        weight_after[2] += (result - sigmoid_value)
    for j in range(len(weight_after)):
        weight_after[j] /= len(datax)
    return weight_after

def load_data(filename):
    t1x = []
    t1y = []
    f = open(filename, 'r')
    temp = f.readlines()
    f.close()
    for line in range(len(temp)):
        t1x.append(float(temp[line].strip().split(',')[0]))
        t1y.append(float(temp[line].strip().split(',')[1]))
    return t1x, t1y


t1x, t1y = load_data('Logistic_data2-1.txt')
t2x, t2y = load_data('Logistic_data2-2.txt')
weight = [0,0,0]
weight_entropy = [0,0,0]

# train the weight of l2-norm
for j in range(10000):
    for i in range(len(weight)):
        weight[i] += gradient_square(weight,0, t1x,t1y)[i]*alpha
        weight[i] += gradient_square(weight,1, t2x,t2y)[i]*alpha
confusion_matrix = [0,0,0,0]
for i in range(len(t1x)):
	predict1 = sigmoid(weight, t1x[i], t1y[i])
	predict2 = sigmoid(weight, t2x[i], t2y[i])
	if(predict1 < 0.5):
		confusion_matrix[0] +=1
		plt.scatter(t1x[i], t1y[i], c="red")
	else:
		confusion_matrix[1] +=1
		plt.scatter(t1x[i], t1y[i], c="blue")
	if(predict2 > 0.5):
		confusion_matrix[3] +=1
		plt.scatter(t2x[i], t2y[i], c="blue")
	else:
		confusion_matrix[2] +=1
		plt.scatter(t2x[i], t2y[i], c="red")

print("L2-norm")
print("-----------------------")
print("The weight after using gradient decent: ", weight[0], "*x", weight[1], "*y +", weight[2])
print("Confusion matrix:")
print("                           predict 0   predict 1")
print("actual 0:                   ", confusion_matrix[0],"        ", confusion_matrix[1])
print("actual 1:                   ", confusion_matrix[2],"        ", confusion_matrix[3])
print("                  predict 0       predict 1")
print("Recall:        ",confusion_matrix[0] / (confusion_matrix[0]+confusion_matrix[1]), "  ",confusion_matrix[3]/(confusion_matrix[3]+confusion_matrix[2]) )
print("                  predict 0       predict 1")
print("Precision:      ",confusion_matrix[0] / (confusion_matrix[0]+confusion_matrix[2]), "   ",confusion_matrix[3]/(confusion_matrix[3]+confusion_matrix[1]))
print("Accuracy:       " ,(confusion_matrix[0] + confusion_matrix[3]) / (confusion_matrix[0]+confusion_matrix[3]+confusion_matrix[1]+confusion_matrix[2]) )
print("-----------------------")
print('\n')



# train the weight of cross-entropy
for j in range(10000):
    for i in range(len(weight_entropy)):
        weight_entropy[i] += gradient_entropy(weight_entropy,0, t1x,t1y)[i]*alpha
        weight_entropy[i] += gradient_entropy(weight_entropy,1, t2x,t2y)[i]*alpha
confusion_matrix = [0,0,0,0]
for i in range(len(t1x)):
	predict1 = sigmoid(weight_entropy, t1x[i], t1y[i])
	predict2 = sigmoid(weight_entropy, t2x[i], t2y[i])
	if(predict1 < 0.5):
		confusion_matrix[0] +=1
		plt.scatter(t1x[i], t1y[i], c="red")
	else:
		confusion_matrix[1] +=1
		plt.scatter(t1x[i], t1y[i], c="blue")
	if(predict2 > 0.5):
		confusion_matrix[3] +=1
		plt.scatter(t2x[i], t2y[i], c="blue")
	else:
		confusion_matrix[2] +=1
		plt.scatter(t2x[i], t2y[i], c="red")

print("Cross entropy")
print("-----------------------")
print("The weight after using gradient decent: ", weight_entropy[0], "*x", weight_entropy[1], "*y +", weight_entropy[2])
print("Confusion matrix:")
print("                           predict 0   predict 1")
print("actual 0:                   ", confusion_matrix[0],"        ", confusion_matrix[1])
print("actual 1:                   ", confusion_matrix[2],"        ", confusion_matrix[3])
print("                  predict 0       predict 1")
print("Recall:        ",confusion_matrix[0] / (confusion_matrix[0]+confusion_matrix[1]), "  ",confusion_matrix[3]/(confusion_matrix[3]+confusion_matrix[2]) )
print("                  predict 0       predict 1")
print("Precision:      ",confusion_matrix[0] / (confusion_matrix[0]+confusion_matrix[2]), "   ",confusion_matrix[3]/(confusion_matrix[3]+confusion_matrix[1]))
print("Accuracy:       " ,(confusion_matrix[0] + confusion_matrix[3]) / (confusion_matrix[0]+confusion_matrix[3]+confusion_matrix[1]+confusion_matrix[2]) )
print("-----------------------")


# data2 
a1 = np.linspace(-6, 6, 100)
a2 = ( (-weight_entropy[0] * a1) - weight_entropy[2]) / weight_entropy[1]
plt.ylim(ymin=-5, ymax=10)
plt.plot(a1, a2)
plt.show()

# data1 
# a1 = np.linspace(-6, 15, 100)
# a2 = ( (-weight[0] * a1) - weight[2]) / weight[1]
# plt.ylim(ymin=-10, ymax=15)
# plt.plot(a1, a2)
# plt.show()

# plt.scatter(t1x,t1y, c="red")
# plt.scatter(t2x,t2y, c="blue")
# plt.show()