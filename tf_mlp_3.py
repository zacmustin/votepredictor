"""
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

import numpy as np
from numpy import array
import sys
from pprint import pprint
import json
import tensorflow as tf


LEARNING_RATE = 0.001
HL_SIZE = 100
INPUT_SIZE = 2

def load_data(filepath):
    with open(filepath,'r') as csvfile:
        data = []
        state = []
        ffr = []
        ffrpc = []
        output_list = []
        counter = 0
        for rows in csvfile:
            if counter < 2600:
                row_data = rows.split(',')
                row_data[4] = row_data[4][:1] #removes /n at end 
                data.append(row_data)
                ffr.append(float(data[counter][2])) # ffrpc = FFRs
                ffrpc.append(float(data[counter][3])) # ffrpc = FFRs
                output_list.append(data[counter][4]) #output_list = D/R
                counter+=1
        return array(ffr),array(ffrpc),array(output_list) 

# input dataset
ffr,ffrpc,Y = load_data('CountyFast.csv')
#put in ffr per capita AND ffr

# CONVERSION OF INPUTS
new_X = []
for ffrx,ffrpcx in zip(ffr,ffrpc):
    new_X.append([np.float(ffrx),np.float(ffrpcx)])
X = np.array(new_X)

# CONVERSION OF OUTPUTs
new_Y = []
for y in Y:
    if y == 'R':
        new_Y.append([np.float(0),np.float(1)])
    else:
        new_Y.append([np.float(1),np.float(0)])
Y = np.array(new_Y)

# tf Graph input
pX = tf.placeholder("float", [None, INPUT_SIZE])
pY = tf.placeholder("float", [None, 2])


def perceptron(x, weights, biases):
    # Hidden layer with ReLU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    #print(out_layer)
    return out_layer

# Store layers weight &amp; bias
weights = {
'h1': tf.Variable(tf.random_normal([INPUT_SIZE, HL_SIZE])),
'h2': tf.Variable(tf.random_normal([HL_SIZE, HL_SIZE])),
'out': tf.Variable(tf.random_normal([HL_SIZE, 2]))
}
biases = {
'b1': tf.Variable(tf.random_normal([HL_SIZE])),
'b2': tf.Variable(tf.random_normal([HL_SIZE])),
'out': tf.Variable(tf.random_normal([2]))
}

print(tf.random_normal([INPUT_SIZE, HL_SIZE]))
# Construct model
logits = perceptron(pX, weights, biases)


# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=pY))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()


# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

#Uncomment to load in 'brain'
"""with open('synapses0.txt', 'r') as weightsfile:
    syn0 = json.load(weightsfile)
with open('synapses1.txt', 'r') as weightsfile:
    syn1 = json.load(weightsfile)
print('Synapses successfully loaded')"""

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(60000):
        avg_cost = 0.
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([train_op, loss_op], feed_dict={pX: X, pY: Y})
        #print(train_op)
        #print(weights['h1'][0])
        # Compute average loss
        avg_cost += c
        # Display logs per epoch step
        if epoch % 1000 == 0:
            print("Epoch:", '%04d' % (epoch), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

"""
    #round up guess to either 0 or 1
    for i in range(len(l2)):
        if l2[i] < 0.5: 
            l2[i] = 0
        else:
            l2[i] = 1

        if l2[i] == Y[i]:
            correct_guesses+=1
        
    if (j % 100) == 0:
        #print(weights)
        print("Iteration " + str(j) + " - Error:" + str(np.mean(np.abs(l2_error))) + " Accuracy: " + str(correct_guesses/2600))
        with open('synapses0.txt', 'w') as weightsfile:
            json.dump(syn0.tolist(), weightsfile)
        with open('synapses1.txt', 'w') as weightsfile:
            json.dump(syn1.tolist(), weightsfile)
"""
#NEXT STEPS
    # DO CSV STUFF WITH PANDAS
    # LOOK AT STATE?
    # BENCHMARK WITH GUESSING SOLELY BASED ON STATE
    # ADD BATCHES
    # TRY WITH TESTING ERROR
    # PLAY AROUND WITH LEARNING RATE
    # GRAPH ERROR OVER TIME
    # ADD MORE DATA
