# The MIT License (MIT)

# Copyright (c) 2014 Jake Cowton

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# A Neural Network to calculate if an RGB value is more red or blue
from perceptron import Perceptron

BLUE = 1
RED = 0
# Lowest MSE
LMSE = 0.001

def normalise(data):
    """
    MUST BE CUSTOMISED PER PROJECT
    Turn data into values between 0 and 1
    @param data list of lists of input data and output e.g.
        [
            [[0,0,255], 1],
            ...
        ]
    @returns Normalised training data
    """
    temp_list = []
    for entry in data:
        entry_list = []
        for value in entry[0]:
            # Normalise the data. 1/255 ~ 0.003921568
            entry_list.append(float(value*0.003921568))
        temp_list.append([entry_list, entry[1]])
    return temp_list

def main(data):

    # Normalise the data
    training_data = normalise(data)

    # Create the perceptron
    p = Perceptron(len(data[0][0]))

    # Number of full iterations
    epochs = 0

    # Instantiate mse for the loop
    mse =999

    while (abs(mse-LMSE) > 0.002):

        # Epoch cumulative error
        error = 0

        # For each set in the training_data
        for value in training_data:

            # Calculate the result
            output = p.result(value[0])

            # Calculate the error
            iter_error = value[1] - output

            # Add the error to the epoch error
            error += iter_error

            # Adjust the weights based on inputs and the error
            p.weight_adjustment(value[0], iter_error)

        # Calculate the MSE - epoch error / number of sets
        mse = float(error/len(training_data))

        # Print the MSE for each epoch
        print ("The MSE of %d epochs is %.10f" % (epochs, mse))

        # Every 100 epochs show the weight values
        if epochs % 100 == 0:
            print ("0: %.10f - 1: %.10f - 2: %.10f - 3: %.10f" % (p.w[0], p.w[1], p.w[2], p.w[3]))

        # Increment the epoch number
        epochs += 1
    
    print("\n" + p.recall([0.8156861440000001, 0.0, 0.192156832]))  
    print("\n" + p.recall([0,0,0.994]))

        
    return p
data = [#((R,G,B), CLASSIFICATION)
        [[0,0,255], BLUE],
        [[0, 0, 255], BLUE],
        [[0, 0, 192], BLUE],
        [[243, 80, 59], RED],
        [[255, 0, 77], RED],
        [[77, 93, 190], BLUE],
        [[255, 98, 89], RED],
        [[208, 0, 49], RED],
        [[67, 15, 210], BLUE],
        [[82, 117, 174], BLUE],
        [[168, 42, 89], RED],
        [[248, 80, 68], RED],
        [[128, 80, 255], BLUE],
        [[228, 105, 116], RED]
    ]


main(data)

#print(data[0])
#print(data[0][0])
#print(len(data[0][0]))
#
#from numpy.random import random_sample
#print(random_sample(4))


