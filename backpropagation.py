import numpy as np
import pandas as pd
from random import random
import math
import matplotlib.pyplot as plt


lrate = 0.01

def load_data(filename):
    x = []
    y = []
    ans = []
    f = open(filename, 'r')
    temp = f.readlines()
    f.close()
    for line in range(len(temp)):
        x.append(float(temp[line].strip().split(',')[0]))
        y.append(float(temp[line].strip().split(',')[1]))
        ans.append(float(temp[line].strip().split(',')[2]))
    return x, y, ans

def initialize():
	network = []
	for i in range(5):
		neuro = {}
		neuro['input'] = [0 for k in range(2)]
		neuro['weight'] = [random() for i in range(3) ]
		# neuro['gradient_sum'] = [random() for i in range(3) ]
		neuro['delta'] = [0]
		neuro['output'] = [0]
		neuro['activate'] = [0]
		network.append(neuro)
	return network

def sigmoid(z):
	try:
		result = 1/(1+math.exp(-1*z))
	except OverflowError:
		result = 0.000001
	return result


def sigmoid_deriv(z):
	try:
		result = sigmoid(z)*(1.0-sigmoid(z))
	except OverflowError:
		result = 0.000001
	return result

def activate(neuro):
	return neuro['input'][0] * neuro['weight'][0] + neuro['input'][1] * neuro['weight'][1] + neuro['weight'][2]



def forward(network, inputx, inputy ):
	network[0]['input'][0] = inputx
	network[0]['input'][1] = inputy
	network[1]['input'][0] = inputx
	network[1]['input'][1] = inputy
	for i in range(5):
		network[i]['activate'] = activate(network[i])
		network[i]['output'] = sigmoid(network[i]['activate'])
		if(i==0):
			network[2]['input'][0] = network[i]['output']
			network[3]['input'][0] = network[i]['output']
		elif(i==1):
			network[2]['input'][1] = network[i]['output']
			network[3]['input'][1] = network[i]['output']
		elif(i==2):
			network[4]['input'][0] = network[i]['output']
		elif(i==3):
			network[4]['input'][1] = network[i]['output']
	return network[4]['output']
	# may be wrong
	# if(network[-1]['output']>0.5):
	# 	network[-1]['output'] = 0
	# else:
	# 	network[-1]['output'] = 1


def train_weight(network, ans):
	error = 0
	for i in reversed(range(5)):
		if(i==4):
			# error = abs(ans - sigmoid(activate(network[i])) )
			error = abs((network[i]['output'] - ans))**2
			network[i]['delta'] = (network[i]['output'] - ans) * sigmoid_deriv(network[i]['activate'])

		elif(i==3 ):
			network[i]['delta'] = network[4]['weight'][1] * network[4]['delta'] * sigmoid_deriv(network[i]['activate'])
		elif(i==2 ):
			network[i]['delta'] = network[4]['weight'][0] * network[4]['delta'] * sigmoid_deriv(network[i]['activate'])	
		elif(i==1 ):
			network[i]['delta'] = (network[2]['weight'][1] * network[2]['delta'] + network[3]['weight'][1] * network[3]['delta'])* sigmoid_deriv(network[i]['activate'])	
		elif(i==0 ):
			network[i]['delta'] = (network[2]['weight'][0] * network[2]['delta'] + network[3]['weight'][0] * network[3]['delta'])* sigmoid_deriv(network[i]['activate'])				
		# print(network[i]['delta'])

	for i in (range(5)):	
		update_weight(network[i])
	return error

def update_weight(neuro):
	global lrate
	# neuro['gradient_sum'][-1] += neuro['delta']**2
	# lrate +=  0.01/(math.sqrt(neuro['gradient_sum'][-1])+1e-7)* neuro['delta']
	neuro['weight'][-1] -= lrate * neuro['delta'] 
	for i in range(2):
		# neuro['gradient_sum'][i] += (neuro['delta'] * neuro['input'][i])**2
		# lrate +=  0.01/(math.sqrt(neuro['gradient_sum'][i])+1e-7)* neuro['delta']* neuro['input'][i]
		neuro['weight'][i] -= lrate * neuro['delta'] * neuro['input'][i]


x,y,y_hat = load_data('data.txt')
network  = initialize()

# to print the ground truth
# for i in range(len(x)):
# 	if(y_hat[i]==1):
# 		plt.scatter(x[i], y[i], c="blue")
# 	else:
# 		plt.scatter(x[i], y[i], c="red")
# plt.show()

for j in range(10):
	error = 0.0
	for k in range(1000):
		for i in range(len(x)):
			forward(network, x[i], y[i])
			error += train_weight(network, y_hat[i])
	print('epochs %d loss: %.10f' %( (j+1)*1000 , error/1000/len(x)  ) )


for i in range(len(x)):
	forward(network, x[i], y[i])
	if(network[-1]['output'] > 0.5):
		plt.scatter(x[i], y[i], c="blue")
	else:
		plt.scatter(x[i], y[i], c="red")
plt.show()
