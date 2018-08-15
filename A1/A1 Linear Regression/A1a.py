
import numpy as np
import sys

training_data_path = sys.argv[1]
testing_data_path = sys.argv[2]
output_path = sys.argv[3]

def train_model(X,Y):
    X_t = np.transpose(X)
    XT_X = np.matmul(X_t,X)
    XT_X_inverse = np.linalg.inv(XT_X)
    W = (XT_X_inverse).dot(X_t).dot(Y)
    return W
    
def predict_using_model(W,test_data):
    return (np.matmul(test_data,W))
    
def test_model(W,test_data,Y):
    Predictions = predict_using_model(W,test_data)
    return np.sum(np.square(Y - Predictions)) 

input_data = np.loadtxt(open(training_data_path, "rb"), delimiter=",")

Y = input_data[:,0].copy()

X = input_data.copy()
X[:,0] = 1


W = train_model(X,Y)

test_data = np.loadtxt(open(testing_data_path, "rb"), delimiter=",")

Y_test = test_data[:,0].copy()
X_test = test_data.copy()
X_test[:,0] = 1

Result = predict_using_model(W,X_test)

np.savetxt(output_path,Result)

