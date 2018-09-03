
# coding: utf-8

# In[1]:


import numpy as np
import random
import sys
from scipy.special import expit as sigmoid

training_data_path = sys.argv[1]
testing_data_path = sys.argv[2]
output_path = sys.argv[3]
batch_size = int(sys.argv[4])
activation = sys.argv[5]
hidden_layers_sizes = []
for i in range(6,len(sys.argv)):
    hidden_layers_sizes.append(int(sys.argv[i]))
print("Hidden Layers = ",hidden_layers_sizes)
n0 = 2


# In[ ]:


def relu(x):
    return (x>0) * x

def tanh(x):
    return np.tanh(x)

def reluPrime(x):
    return (x>0)+0

def tanhPrime(x):
    return 1 - np.power(x,2)

def sigmoidPrime(x):
    return x * (1 - x)

def exp_normalize(x):
    b = np.amax(x,axis=1,keepdims = True)
    y = np.exp(x - b)
    return y / y.sum(axis=1,keepdims=True)


# In[ ]:


class NeuralNetwork:
    
    def __init__(self,input_size,output_size,hidden_layers_sizes, activation):
        self.weights = []
        self.biases = []
        
        if(activation == 'relu'):
            self.activation = relu
            self.activationPrime = reluPrime
        elif(activation == 'tanh'):
            self.activation = tanh
            self.activationPrime = tanhPrime
        else:
            self.activation = sigmoid
            self.activationPrime = sigmoidPrime
        
        self.input_size = input_size
        self.output_size = output_size
        self.hiddent_layers_sizes = hidden_layers_sizes
        
        prev_layer_count = input_size
        
        for i in range(len(hidden_layers_sizes) + 1):
            if i==len(hidden_layers_sizes):
                self.weights.append(np.random.rand(prev_layer_count, output_size)/100)
                self.biases.append(np.random.rand(1, output_size)/100)        
            else:
                hidden_layer_count = hidden_layers_sizes[i]
                self.weights.append(np.random.rand(prev_layer_count, hidden_layer_count)/100)
                self.biases.append(np.random.rand(1, hidden_layer_count)/100)
                prev_layer_count = hidden_layer_count
        
    def train(self,inpX,inpY,batch_size,n0,max_iterations):
        max_examples = inpX.shape[0]
        max_possible_iterations = int(0.5 + max_examples / batch_size)
        num_hidden_layers = len(self.weights) - 1
        
        for n in range(max_iterations):
            # Forming Mini Batches
            i_eff = n%max_possible_iterations
            
            outputs = []
            
            if i_eff != max_possible_iterations - 1:
                X = inpX[i_eff*batch_size: (i_eff+1)*batch_size]
                Y = inpY[i_eff*batch_size: (i_eff+1)*batch_size]
            else:
                X = inpX[i_eff*batch_size:]
                Y = inpY[i_eff*batch_size:]
            
            # Updating Learning Rate
            lr = n0 / np.sqrt(n+1) 
                
            # Neural Network Forward Propagation
            outputs.append(X)
            prev_layer_output = X
            for i in range(num_hidden_layers + 1):
                weight = self.weights[i]
                bias = self.biases[i]
                if i == num_hidden_layers:
                    prev_layer_output = sigmoid(prev_layer_output.dot(weight) + bias)
                else:
                    prev_layer_output = self.activation(prev_layer_output.dot(weight) + bias)
                outputs.append(prev_layer_output)
            
            # Backpropagation
            dWs = []
            dbs = []
            
            y_onehot = np.zeros((Y.shape[0],self.output_size))
            y_onehot[range(Y.shape[0]),Y] = 1
            
            for i in range(num_hidden_layers + 1,0,-1):
                if i == num_hidden_layers + 1:
                    delta = (outputs[i] - y_onehot).dot(2/Y.shape[0]) * sigmoidPrime(outputs[i])
                else:
                    delta = delta.dot(self.weights[i].T) * self.activationPrime(outputs[i])
                dW = (outputs[i-1].T).dot(delta)
                dWs.append(dW)
                dbs.append(np.sum(delta,axis=0,keepdims=True))
                
            if (n%100 == 0):
                loss = np.sum(np.power(outputs[-1] - y_onehot,2) )/Y.shape[0]
                labels = np.argmax(outputs[-1],axis = 1)
                accuracy = 100 * np.sum(labels == Y)/Y.shape[0]
                print("Iteration ",n,"\tLoss = ",loss,"\tAccuracy = ",accuracy,"%")
                
            dWs.reverse()
            dbs.reverse()

            # Gradient Descent Parameter Update
            for i in range(len(dWs)):
                self.weights[i] += dWs[i].dot(-1 * lr)
                self.biases[i] += dbs[i].dot(-1 * lr)

    def predict(self,X):
        return self.forward_run(X)
        
    def forward_run(self,X):
        prev_layer_output = X
        num_hidden_layers = len(self.weights) - 1
        for i in range(num_hidden_layers + 1):
            weight = self.weights[i]
            bias = self.biases[i]
            if i == num_hidden_layers:
                probabilities = sigmoid(prev_layer_output.dot(weight) + bias)
                labels = np.argmax(probabilities,axis = 1)
                return labels
            else:
                prev_layer_output = self.activation(prev_layer_output.dot(weight) + bias)


# In[ ]:


def load_data(path,avg,std):
    if avg is None:
        input_data = np.loadtxt(open(path, "rb"), delimiter=",")
        Y = input_data[:,0].copy()
        X = input_data[:,1:].copy()
        avg = np.average(X,axis=0)
        X = X - avg
        std = np.std(X,axis=0)
        std[(std == 0)] = 1
        X = X / std
        return X,Y,avg,std
    else:
        input_data = np.loadtxt(open(path, "rb"), delimiter=",")
        X = input_data[:,1:].copy()
        X = (X - avg)/std
        return X


# In[ ]:


inpX,Y,avg,std = load_data(training_data_path,None,None)


# In[ ]:


X = inpX.copy()

input_size = X.shape[1]
output_size = int(np.amax(Y))+1
num_examples = X.shape[0]
max_iterations = int(4*(num_examples/batch_size))

network = NeuralNetwork(input_size,output_size,hidden_layers_sizes,activation)
network.train(X,Y.astype(int),batch_size,n0,max_iterations)


# In[ ]:


predictions = network.predict(X.copy())
print("Accuraccy on Training Data = ",100 * np.sum(predictions == Y)/Y.shape[0])
print("Average of predictions on Training Data = ",np.average(predictions))


# In[ ]:


testX = load_data(testing_data_path,avg,std)


# In[ ]:


predictions = network.predict(testX)
np.savetxt(output_path,predictions,fmt="%i")

