#Code taken from https://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np
from numpy import array
import sys
from pprint import pprint
import json



LEARNING_RATE = 1e-5
HL_SIZE = 100

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def load_data(filepath):
    with open(filepath,'r') as csvfile:
        data = []
        state_dict = {}
        state = []
        output_list = []
        counter = 0
        state_counter = 0
        for rows in csvfile:
            if counter < 2600:
                row_data = rows.split(',')
                row_data[4] = row_data[4][:1] #removes /n at end 
                data.append(row_data)
                #Checks if state has been mapped to dict & maps if it hasn't
                if data[counter][1] not in state_dict:
                    state_dict.update({data[counter][1] : state_counter})
                    state_counter+=1
                state.append(float(state_dict[data[counter][1]]))
                output_list.append(data[counter][4]) #output_list = D/R
                counter+=1
        return array(state),array(output_list) 

# input dataset
states,Y = load_data('CountyFast.csv')
#put in ffr per capita AND ffr

# CONVERSION OF INPUTS
new_X = []
for statesx in states:
    new_X.append([np.float(statesx)])
X = np.array(new_X)

# CONVERSION OF OUTPUTs
new_Y = []
for y in Y:
    if y == 'R':
        print()
        new_Y.append([0])
    else:
        new_Y.append([1])
Y = np.array(new_Y)

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

"""#Uncomment to load in 'brain'
with open('synapses0.txt', 'r') as weightsfile:
    syn0 = json.load(weightsfile)
with open('synapses1.txt', 'r') as weightsfile:
    syn1 = json.load(weightsfile)
print('Synapses successfully loaded')
with open('weights.txt','r') as weightsfile:
    syn0 = json.load(weightsfile)
    print(gucci)"""

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((1,HL_SIZE)) - 1
syn1 = 2*np.random.random((HL_SIZE,1)) - 1

for j in range(60000):
    correct_guesses = 0
    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # how much did we miss the target value?
    l2_error = Y - l2
        
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

    #syn0.tolist()
    #round up guess to either 0 or 1
    for i in range(len(l2)):
        if l2[i] < 0.5: 
            l2[i] = 0
        else:
            l2[i] = 1

        if l2[i] == Y[i]:
            correct_guesses+=1
        #print(l2[i],Y[i])
    if (j % 100) == 0:
        #print(weights)
        print("Iteration " + str(j) + " - Error:" + str(np.mean(np.abs(l2_error))) + " Accuracy: " + str(correct_guesses/2600))
        with open('synapses0.txt', 'w') as weightsfile:
            json.dump(syn0.tolist(), weightsfile)
        with open('synapses1.txt', 'w') as weightsfile:
            json.dump(syn1.tolist(), weightsfile)

#NEXT STEPS
    # DO CSV STUFF WITH PANDAS
    # LOOK AT STATE?
    # BENCHMARK WITH GUESSING SOLELY BASED ON STATE
    # ADD BATCHES
    # TRY WITH TESTING ERROR
    # PLAY AROUND WITH LEARNING RATE
    # GRAPH ERROR OVER TIME
    # ADD MORE DATA
