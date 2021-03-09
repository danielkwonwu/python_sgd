import numpy as np
import sys, os
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

#basic usage of command line input
if len(sys.argv) < 3:
    sys.exit(f"Usage: {sys.argv[0]} filename y-index nrows(=all if no argument)")
if len(sys.argv) == 4:
    try:
        int(sys.argv[3])
    except:
        sys.exit(f"Error: Invalid nrow argument")
filename = sys.argv[1]
y_location = sys.argv[2]
try:
    int(y_location)
except:
    sys.exit(f"Error: Invalid y index")
if not os.path.exists(filename):
    sys.exit(f"Error: File '{sys.argv[1]}' not found")

#read from file
data = ""
if ".txt" in filename:
    if len(sys.argv) == 4:
        data = pd.read_csv(filename, sep="\t", nrows=int(sys.argv[3]))
    else:
        data = pd.read_csv(filename, sep="\t")
else:
    if len(sys.argv) == 4:
        data = pd.read_excel(filename, nrows=int(sys.argv[3]))
    else:
        data = pd.read_excel(filename)

#pad d0 with 1's 
data = pd.concat([pd.Series(1, index=data.index, name= "d0"), data], axis = 1)

#separate x and y in order to form a regression model
len_data = len(data.columns)
if int(y_location) >= len_data:
    sys.exit(f"Error: y index out of bounds")
x = data.drop(data.columns[int(y_location)+1], axis=1)
y = data[data.columns[int(y_location)+1]]


#construct a model out of small random numbers
regress = []
for _ in range(len(x.columns)):
    regress.append(random.random())
print (regress)

#calculate t(k) in a loss function with the vector

def calculate_w_d (regress, x):
    return np.dot(regress, np.transpose(x))


#specify the objective function of LSE (Least Square Error)
def calculate_objective_function(x, y, regress):
    sum = 0
    for line in range(0, len(x.index)):
        w_d = calculate_w_d (regress, x.iloc[line])
        each = (y.iloc[line] - w_d) ** 2
        sum += each
    return sum/2

print (calculate_objective_function(x,y,regress))


def gradient_descent (regress, x, y, iterations):
    iter = 0
    while iter < iterations :
        #cost = calculate_objective_function(x, y, regress)
        step = 0
        for k in range(len(regress)):
            sum = 0
            for a in range(len(x.index)):
                #gradient 
                sum += (calculate_w_d(regress, x.iloc[a]) - y.iloc[a])*x.iloc[a][k] 
            regress[k] = regress[k] - 0.001*sum
            step += 0.001*sum
        #print(regress)
        iter += 1
        if np.abs(step) < 1e-06:
                print("converged")
                break
        #print("Sum " , step)
    return regress

after = gradient_descent (regress, x, y, 10000)
print (regress)

x_plot = x
y_plot = y


#report LSEs and error for first dataset
#LSEs
print(len(x_plot.columns[0]))
print(len(y_plot))
print ("LSE :", calculate_objective_function(x,y,regress))
plt.figure()
plt.scatter(x= x_plot["hour/week"], y= y_plot, color = 'blue')
plt.plot(x, regress[0] + x*regress[1])
plt.show()