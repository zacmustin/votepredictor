#Code taken from https://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np
from numpy import array
import sys



LEARNING_RATE = 1e-4 

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def load_data(filepath):
    with open(filepath,'r') as csvfile:
        data = []
        i_list_list = []
        output_list = []
        counter = 0
        for rows in csvfile:
            if counter < 2600:
                row_data = rows.split(',')
                row_data[3] = row_data[3][:1] #removes /n at end 
                data.append(row_data)
                i_list_list.append(int(data[counter][2])) # i_list_list = FFRs
                output_list.append(data[counter][3]) #output_list = D/R
                counter+=1
        return array(i_list_list),array(output_list) 

# input dataset
X,Y = load_data('CountyFast.csv')


# CONVERSION OF INPUTS
new_X = []
for x in X:
    new_X.append([x])
X = np.array(new_X)

# CONVERSION OF OUTPUTs
new_Y = []
for y in Y:
    if y == 'R':
        new_Y.append([0])
    else:
        new_Y.append([1])
Y = np.array(new_Y)

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((1,100)) - 1
syn1 = 2*np.random.random((100,1)) - 1

for j in range(60000):



    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # how much did we miss the target value?
    l2_error = Y - l2
    
    if (j% 100) == 0:
        print("Iteration " + str(j) + " - Error:" + str(np.mean(np.abs(l2_error))))
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += LEARNING_RATE * l1.T.dot(l2_delta)
    syn0 += LEARNING_RATE * l0.T.dot(l1_delta)

#NEXT STEPS
    # ADD BATCHES
    # PLAY AROUND WITH LEARNING RATE
    # GRAPH ERROR OVER TIME
    # ADD MORE DATA