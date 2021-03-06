
# coding: utf-8

# In[25]:


import numpy as np
import random
import sys
from scipy.special import expit as sigmoid

training_data_path = sys.argv[1]
testing_data_path = sys.argv[2]
output_path = sys.argv[3]
batch_size = int(sys.argv[4])
n0 = float(sys.argv[5])
activation = sys.argv[6]
hidden_layers_sizes = []
for i in range(7,len(sys.argv)):
    hidden_layers_sizes.append(int(sys.argv[i]))

# training_data_path = "../data/devnagri_train.csv"
# testing_data_path = "../data/devnagri_test_public.csv"
# output_path = "../data/nn/a/cs1160328.txt"
# batch_size = 128
# n0 = 5
# activation = 'sigmoid'
# hidden_layers_sizes = [50,50]


# In[2]:


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


# In[28]:


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
        
        count = 0
            
        lr = n0
        totLoss = 0
        prevAvgLoss = sys.float_info.max
        epoch = 0
        
        for n in range(max_iterations):
            # Forming Mini Batches
            i_eff = n%max_possible_iterations
            
            # Updating Learning Rate
            if (i_eff == 0 and n!=0):
                avgLoss = totLoss/max_possible_iterations
                
                if(np.absolute(avgLoss - prevAvgLoss) < 0.0001 * prevAvgLoss):
                    stopCount += 1
                    if stopCount > 1:
                        break
                else:
                    stopCount = 0
                if(avgLoss >= prevAvgLoss):
                    count += 1
                    lr = n0 / np.sqrt(count+1)
                print("Epoch = ",epoch," Average Loss = ",avgLoss," New Learning Rate = ",lr)
                epoch += 1
                prevAvgLoss = avgLoss
                totLoss = 0
            
            outputs = []
            
            if i_eff != max_possible_iterations - 1:
                X = inpX[i_eff*batch_size: (i_eff+1)*batch_size]
                Y = inpY[i_eff*batch_size: (i_eff+1)*batch_size]
            else:
                X = inpX[i_eff*batch_size:]
                Y = inpY[i_eff*batch_size:]
                
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
                loss_ = np.sum(np.power(outputs[-1] - y_onehot,2) )/Y.shape[0]
                labels_ = np.argmax(outputs[-1],axis = 1)
                accuracy_ = 100 * np.sum(labels_ == Y)/Y.shape[0]
                print("Iteration ",n,"\tLoss = ",loss_,"\tAccuracy = ",accuracy_,"%")
                
            dWs.reverse()
            dbs.reverse()

            # Gradient Descent Parameter Update
            for i in range(len(dWs)):
                self.weights[i] += dWs[i].dot(-1 * lr)
                self.biases[i] += dbs[i].dot(-1 * lr)

            loss = np.sum(np.power(outputs[-1] - y_onehot,2) )/Y.shape[0]
            totLoss += loss
                
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


# In[4]:


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


# In[5]:


inpX,Y,avg,std = load_data(training_data_path,None,None)


# In[ ]:


X = inpX.copy()

input_size = X.shape[1]
output_size = int(np.amax(Y))+1
num_examples = X.shape[0]
max_iterations = int(40*(num_examples/batch_size))

network = NeuralNetwork(input_size,output_size,hidden_layers_sizes,activation)
network.train(X,Y.astype(int),batch_size,n0,max_iterations)


# In[27]:


predictions = network.predict(X.copy())
print("Accuraccy on Training Data = ",100 * np.sum(predictions == Y)/Y.shape[0])
# print("Average of predictions on Training Data = ",np.average(predictions))


# In[8]:


testX = load_data(testing_data_path,avg,std)


# In[9]:


predictions = network.predict(testX)
np.savetxt(output_path,predictions,fmt="%i")

