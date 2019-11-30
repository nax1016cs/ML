import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def matrix_mul(x,y):
  # iterate through rows of X
  try:
    temp = len(y[0])
  except TypeError:
    result = np.zeros( (len(x), 1 ) )
    for i in range(len(x)):
      for k in range(len(y)):
        result[i][0] += x[i][k] * y[k]
    return result
  else:
    result = np.zeros( (len(x), len(y[0]) ) )
    for i in range(len(x)):
      for j in range(len(y[0])):
          for k in range(len(y)):
            result[i][j] += x[i][k] * y[k][j]
    return result

def matrix_transpose(x):
  # len(x) = row(6), len(x[0]) = columns(3)
  result = [[x[j][i] for j in range(len(x))] for i in range(len(x[0]))] 
  return result

def matrix_inverse(x):
  result = np.identity(len(x),dtype = float)
  for i in range(len(x)):
    target = x[i][i]
    for j in range(len(x)):
      if(i != j):
        ratio = x[j][i] / target
        for k in range(len(x[j])):
          x[j][k] -= x[i][k]*ratio
          result[j][k] -=  result[i][k]*ratio
  for i in range(len(x)):
    division = x[i][i]
    for j in range(len(x[i])):
      result[i][j] /= division
  return result

x = []
y = []

f = open('linear_data.txt', 'r')
temp = f.readlines()
f.close()
for line in range(len(temp)):
  x.append(float(temp[line].strip().split(',')[0]))
  y.append(float(temp[line].strip().split(',')[1]))

linear1x = []
linear2x = []
for num in x:
  linear1x.append([1,float(num)])
  linear2x.append([1,float(num),float(num)**2])

part1 = matrix_inverse(matrix_mul(matrix_transpose(linear1x),linear1x))
part2 = matrix_mul(matrix_transpose(linear1x),y)   
result1 = matrix_mul(matrix_transpose(part1),part2)
# print(result1)
part3 = matrix_inverse(matrix_mul(matrix_transpose(linear2x),linear2x))
part4 = matrix_mul(matrix_transpose(linear2x),y)   
result2 = matrix_mul(matrix_transpose(part3),part4)
# print(result2)

w1 = 0
w2 = 0
for i in range(len(x)):
  w1 += (y[i]-(result1[0] + result1[1]*float(x[i])))**2
  w2 += (y[i]-(result2[0] + result2[1]*float(x[i])+ result2[2]*(float(x[i])**2) ))**2

print("Fitting line: ", result1[1][0],"X^1 +", result1[0][0])
print("Total error: ", w1[0])
# plt.(x,y)
print("Fitting line: ", result2[2][0],"X^2 +", result2[1][0],"X^1+", result2[0][0])
print("Total error: ", w2[0])


space = np.linspace(-6, 6, 100)  
F = result1[1][0]*space + result1[0][0]
Q = result2[2][0]*space**2 + result2[1][0]*space + result2[0][0] 
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x,y)
plt.title("n = 3") 
line1, = plt.plot(space, Q, color = 'red')           
plt.show()