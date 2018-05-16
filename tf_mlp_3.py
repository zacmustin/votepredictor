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
INPUT_SIZE = 3

def load_data(filepath):
    with open(filepath,'r') as csvfile:
        data = []
        state = []
        ffr = []
        ffrpc = []
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
                ffr.append(float(data[counter][2])) # ffrpc = FFRs
                ffrpc.append(float(data[counter][3])) # ffrpc = FFRs

                #Checks if state has been mapped to dict & maps if it hasn't
                if data[counter][1] not in state_dict:
                    state_dict.update({data[counter][1] : state_counter})
                    state_counter+=1
                state.append(float(state_dict[data[counter][1]]))

                output_list.append(data[counter][4]) #output_list = D/R
                counter+=1
        return array(ffr),array(ffrpc),array(state),array(output_list) 

# input dataset
ffr,ffrpc,states,Y = load_data('CountyFast.csv')
#put in ffr per capita AND ffr

# CONVERSION OF INPUTS
new_X = []
for ffrx,ffrpcx,statesx in zip(ffr,ffrpc,states):
    new_X.append([np.float(ffrx),np.float(ffrpcx),(statesx)])
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
    layer_1 = tf.nn.relu(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    #print(out_layer)
    return out_layer

# Store layers weight &amp; bias
weights = {
'h1': tf.Variable(tf.random_normal([INPUT_SIZE, HL_SIZE])),
'out': tf.Variable(tf.random_normal([HL_SIZE, 2]))
}
biases = {
'b1': tf.Variable(tf.random_normal([HL_SIZE])),
'out': tf.Variable(tf.random_normal([2]))
}

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
            # Test model
            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(pY, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Epoch:", '%04d' % (epoch), "Cost={:.5f}".format(avg_cost), "Accuracy={:.2%}".format(accuracy.eval({pX: X, pY: Y})))



#NEXT STEPS
    # DO CSV STUFF WITH PANDAS
    # LOOK AT STATE?
    # BENCHMARK WITH GUESSING SOLELY BASED ON STATE
    # ADD BATCHES
    # TRY WITH TESTING ERROR
    # PLAY AROUND WITH LEARNING RATE
    # GRAPH ERROR OVER TIME
    # ADD MORE DATA
