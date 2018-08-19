from sklearn import linear_model
import numpy as np
import sys

training_data_path = sys.argv[1]
testing_data_path = sys.argv[2]
output_path = sys.argv[3]

def get_error(true_vals,predicted_vals):
    min_value = np.min(true_vals)
    error = np.sum(np.square(true_vals-predicted_vals))/np.sum(np.square(true_vals-min_value))
    return error
def normalize(X):
    X_normalized = (X - np.min(X,axis=0)) / (np.max(X,axis=0) - np.min(X,axis=0))
    return X_normalized
def remove_useless_columns(X,Y):
    X_new = np.zeros((X.shape[0],X.shape[1]))
    X_normalized = normalize(X)
    j=0
    columns = []
    for i in range(0,X.shape[1]):
        correlation = np.correlate(Y,X_normalized[:,i]) / (2*Y.shape[0] + 1)
        if (correlation[0] >= 50.0):
            columns.append(i)
    X_new = X_new[:,columns]
    return X_new,columns

def get_with_new_features(X):
    X_new = np.zeros((X.shape[0],X.shape[1] * 4))
    X_new[:,:X.shape[1]] = X
    for i in range(0,X.shape[1]):
        new_feature = X[:,i]*X[:,i]*X[:,i]
        X_new[:,X.shape[1] + i] = new_feature
    for i in range(0,X.shape[1]):
        new_feature = X[:,i]*X[:,i]
        X_new[:,2 * X.shape[1] + i] = new_feature
    for i in range(0,X.shape[1]):
        new_feature = X[:,i]*X[:,i]*X[:,i]*X[:,i]
        X_new[:,3 * X.shape[1] + i] = new_feature
    return X_new

input_data = np.loadtxt(open(training_data_path, "rb"), delimiter=",")

Y = input_data[:,0].copy()
X = input_data[:,1:].copy()

X_prime = get_with_new_features(X)
reg = linear_model.LassoLars(alpha=0.00005)
reg.fit(X_prime,Y)

test_data = np.loadtxt(open(testing_data_path, "rb"), delimiter=",")

Y_test = test_data[:,0].copy()
X_test = test_data[:,1:].copy()

X_test_prime = get_with_new_features(X_test)

Predictions = reg.predict(X_test_prime)

np.savetxt(output_path,Predictions)

