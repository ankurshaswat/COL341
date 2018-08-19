
# coding: utf-8

# In[24]:


import numpy as np
import sys
import csv
import re
import scipy.sparse as sp

method = int(float(sys.argv[1]))
learning_rate = float(sys.argv[2])
iterations = int(float(sys.argv[3]))
batch_size = int(float(sys.argv[4]))
training_data_path = sys.argv[5]
vocabulary_path = sys.argv[6]
testing_data_path = sys.argv[7]
output_path = sys.argv[8]

lambdas = [0.1,1,10,100,1000]
folds = 10



# In[32]:


def load(data_path,vocab_map,num_features):
    Y = []
    S = []
#     regex = re.compile('[^a-zA-Z ]')
    with open(data_path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        spamreader = list(spamreader)
        n = len(spamreader)
        S = sp.dok_matrix((n,num_features), dtype=np.int64)
        #X = np.zeros((n,num_features))
        Y = np.zeros((n,1))
        i = 0
        for row in spamreader:
#             cleaned_text = regex.sub('',row[1])
            words = row[1].split(' ')
            S[i,0] = 1
            for word in words:
                if word in vocab_map:
                    S[i,vocab_map[word]] += 1
            Y[i] = int(row[0])
            i += 1
    return S.tocsr(),Y

def load_data(data_path,vocab_map,num_features):
    X_sparse,Y = load(data_path,vocab_map,num_features)
    return X_sparse,Y

def load_vocab(data_path):
    vocab_map = {}
    with open(data_path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')        
        i = 0
        for row in spamreader:
            if row[0] not in vocab_map:
                vocab_map[row[0]] = i + 1
                i += 1
    return vocab_map, i+1

def sigmoid(t):
    return 1.0 / (1.0 + np.exp(t * -1))

def get_log_likelihood(W,X,Y):
    X_W = X.dot(W)
    Predictions = sigmoid(X_W)
    return np.sum(Y.T.dot(np.log(Predictions)) + (1-Y.T).dot(np.log(1-Predictions)))/Y.shape[0]

def get_predictions(W,X):
    X_W = X.dot(W)
    return np.rint(sigmoid(X_W))

def get_accuracy(W,X,Y):
    Predictions = get_predictions(W,X)
    return 100*np.sum(Predictions == Y)/Y.shape[0]

def f_dash_gamma(X_W,Xd,Y,gamma):
    return (sigmoid(X_W + Xd.dot(gamma)) - Y).T.dot(Xd)

def get_optimal_learning_rate(W,X,Y):
    X_W = X.dot(W)
    d = X.T.dot(Y - sigmoid(X_W))
    d = d / np.linalg.norm(d)
    Xd = X.dot(d)
    
    gamma1 = 0.0
    gamma2 = 1.0
#     print(np.absolute(gamma1-gamma2) > 0.001)
    while(np.absolute(gamma1-gamma2) > 0.001):
        mid = (gamma1 + gamma2)/2
        f_dash_val = f_dash_gamma(X_W,Xd,Y,mid)
#         print(gamma1,gamma2,f_dash_val)
        if (f_dash_val < 0):
            gamma1 = mid
        else:
            gamma2 = mid
    
    return (gamma1 + gamma2) / 2

def train_model(inpX,inpY,iterations,learning_rate,lam,batch_size,learning_rate_mode):
    m = inpX.shape[1]
    n= inpX.shape[0]
    num_iters_max = int(0.5 + n/batch_size)
    W = np.zeros((m,1))
#     XT = X.transpose()
    for i in range(iterations):
        
        i_eff = i%num_iters_max
        if i_eff != num_iters_max - 1:
            X = inpX[i_eff*batch_size: (i_eff+1)*batch_size]
            Y = inpY[i_eff*batch_size: (i_eff+1)*batch_size]
        else:
            X = inpX[i_eff*batch_size:]
            Y = inpY[i_eff*batch_size:]
        XT = X.transpose()
        
        g_val = sigmoid(X.dot(W))
#        print('Accuracy = ',get_accuracy(W,X,Y))
#         print('Log Likeli = ',get_log_likelihood(W,X,Y))
        if (learning_rate_mode == 1):
            W = W + (XT.dot(Y-g_val) - W.dot(lam)).dot(learning_rate).dot(1/X.shape[0])
        elif (learning_rate_mode == 2):
            W = W + (XT.dot(Y-g_val) - W.dot(lam)).dot(learning_rate / np.sqrt(i+1)).dot(1/X.shape[0])
        else:
            lr = get_optimal_learning_rate(W,X,Y)
#             print('Best Learning Rate ..= ',lr)
            W = W + (XT.dot(Y-g_val) - W.dot(lam)).dot(lr).dot(1/X.shape[0])
    return W

def kFold_cross_validation(X,Y,lambdas,folds,iterations,learning_rate,batch_size,learning_rate_mode):
    
    fold_size = int(X.shape[0]/folds)
    
    sums = []
    for lam in lambdas:
        sums.append(0.0)
        
    for i in range(folds):
        if i < folds - 1 :
            X_test = X[i*fold_size:(i+1)*fold_size]
            X_train = sp.vstack((X[:i*fold_size],X[(i+1)*fold_size:]))
            Y_test = Y[i*fold_size:(i+1)*fold_size]
            Y_train = np.vstack((Y[:i*fold_size],Y[(i+1)*fold_size:]))
        else:
            X_test = X[i*fold_size:]
            X_train = X[:i*fold_size]
            Y_test = Y[i*fold_size:]
            Y_train = Y[:i*fold_size]
        for i in range(len(lambdas)):
            W= train_model(X_train,Y_train,iterations,learning_rate,lambdas[i],batch_size,learning_rate_mode)
#             sums[i] += get_log_likelihood(W,X_test,Y_test)
            sums[i] += get_accuracy(W,X_test,Y_test)
#        print(sums)
    for i in range(0,len(sums)):
        sums[i] /= folds
    return sums


# In[3]:


vocab_map, m = load_vocab(vocabulary_path)


# In[4]:


# print(vocab_map)
# print(m)


# In[5]:


X,Y = load_data(training_data_path,vocab_map,m)


# In[6]:


#print(X_train_sparse)
#print(Y_train)


# In[7]:


# print(X.shape)


# In[8]:


# print(Y.shape)


# In[9]:


# W = train_model(X,Y,iterations,learning_rate,0.01,2)


# In[25]:


# W = train_model(X,Y,iterations,learning_rate,0.01)

# W = train_model(X,Y,iterations,learning_rate,0.01,1)
accuracy = kFold_cross_validation(X,Y,lambdas,folds,iterations,learning_rate,batch_size,method)


# In[26]:


max = accuracy[0]
lam = lambdas[0]

for i in range (0,len(lambdas)):
    if (accuracy[i] > max):
        lam = lambdas[i]
        max = accuracy[i]
        
#print(lam)


# In[33]:


W = train_model(X,Y,iterations,learning_rate,lam,batch_size,method)


# In[17]:


X_test,Y_test = load_data(testing_data_path,vocab_map,m)


# In[28]:


#print(get_accuracy(W,X_test,Y_test))


# In[19]:


# print(X_test.shape)


# In[29]:


Predictions = get_predictions(W,X_test)
np.savetxt(output_path,Predictions,fmt="%i")

