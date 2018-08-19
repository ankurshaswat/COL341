import numpy as np
import sys

training_data_path = sys.argv[1]
testing_data_path = sys.argv[2]
output_path = sys.argv[3]

folds = 10
lambdas = [0,0.01,0.05,0.1,1,10,100,1000]

def train_model(X,Y,lam = 0):
    n = X.shape[0]
    X_t = np.transpose(X)
    XT_X = np.matmul(X_t,X)
    XT_X_inverse = np.linalg.inv(XT_X + (lam * np.identity(X.shape[1])))
    W = (XT_X_inverse).dot(X_t).dot(Y)
    return W

def predict_using_model(W,test_data):
    return (np.matmul(test_data,W))
    
def test_model(W,test_data,Y):
    Predictions = predict_using_model(W,test_data)
    return np.sum(np.square(Y - Predictions)) 

def kFold_cross_validation(X,Y,lambdas,folds):
    
    fold_size = int(X.shape[0]/folds)
    
    sums = []
    
    CV_test_X, CV_train_X = np.split(X.copy(), [fold_size], axis=0)
    CV_test_Y, CV_train_Y = np.split(Y.copy(), [fold_size], axis=0)

    for lam in lambdas:
        W= train_model(CV_train_X,CV_train_Y,lam)
        sums.append(test_model(W,CV_test_X,CV_test_Y))

    for i in range (0,folds-1):
        if i==folds-2:
            CV_train_X[i*fold_size:(i+1)*fold_size],CV_test_X = CV_test_X, CV_train_X[i*fold_size:]
            CV_train_Y[i*fold_size:(i+1)*fold_size],CV_test_Y = CV_test_Y, CV_train_Y[i*fold_size:]
        else:
            CV_train_X[i*fold_size:(i+1)*fold_size],CV_test_X = CV_test_X, CV_train_X[i*fold_size:(i+1)*fold_size]
            CV_train_Y[i*fold_size:(i+1)*fold_size],CV_test_Y = CV_test_Y, CV_train_Y[i*fold_size:(i+1)*fold_size]

        for i in range(0, len(lambdas)):
            W= train_model(CV_train_X,CV_train_Y,lam)
            sums[i] += test_model(W,CV_test_X,CV_test_Y)
    
    for i in range(0,len(sums)):
        sums[i] /= folds
    return sums

input_data = np.loadtxt(open(training_data_path, "rb"), delimiter=",")

Y = input_data[:,0].copy()
X = input_data.copy()
X[:,0] = 1



errors = kFold_cross_validation(X,Y,lambdas,folds)

min = errors[0]
lam = lambdas[0]

for i in range (0,len(lambdas)):
    if (errors[i] < min):
        lam = lambdas[i]
        min = errors[i]
        
W = train_model(X,Y,lam)

test_data = np.loadtxt(open(testing_data_path, "rb"), delimiter=",")

Y_test = test_data[:,0].copy()
X_test = test_data.copy()
X_test[:,0] = 1

Result = predict_using_model(W,X_test)

np.savetxt(output_path,Result)

