
# coding: utf-8

# In[9]:


import csv
from math import log
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import sys

training_data_path = sys.argv[1]
testing_data_path = sys.argv[2]
output_path = sys.argv[3]

# training_data_path = "../data/amazon_train.csv"
# testing_data_path = "../data/amazon_test_public.csv"
# output_path = "../data/cs1160328.txt"

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


# In[2]:


def get_list_of_words(sentence):
    word_tokens = word_tokenize(sentence)
    words = [w for w in word_tokens if not w in stop_words]
    result = []
    for word in words:
        result.append(ps.stem(word))
    return result

def load_train_data(path):
    total_reviews = 0
    num_reviews = [0,0,0,0,0]
    dictionary = {1:{},2:{},3:{},4:{},5:{}}
    vocab = {}
    total_words_in_vocab = 0
    with open(path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            total_reviews += 1
            rating = int(float(row[0]))
            num_reviews[rating - 1] += 1
            words = get_list_of_words(row[1])
            dict = {}
            for word in words:
                if word not in dict:
                    dict[word] = 1
                    if word not in dictionary[rating]:
                        dictionary[rating][word] = 1
                        if word not in vocab:
                            vocab[word] = 1
                            total_words_in_vocab += 1
                    else:
                        dictionary[rating][word] += 1
    return num_reviews,total_reviews,dictionary,total_words_in_vocab

def load_test_data(path):
    data = []
    with open(path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            data.append(get_list_of_words(row[1]))
    return data

def predict(testing_examples,dictionary,num_reviews,total,vocab_len):
    ratings = []
    for list_of_words in testing_examples:
        probabs = []
        for rating in range(1,6):
            probab = log(num_reviews[rating-1]/total)
            for word in list_of_words:
                if word in dictionary[rating]:
                    probab += log((dictionary[rating][word] + 1)/ (num_reviews[rating-1] + vocab_len + 1))
                else:
                    probab += log(1/ (num_reviews[rating-1] + vocab_len + 1))
            probabs.append(probab)
        best_rating  = probabs.index(max(probabs))+1
        ratings.append(best_rating)
    return ratings


# In[3]:


num_reviews, tot, dictionary,vocab_len = load_train_data(training_data_path)


# In[4]:


test_data = load_test_data(testing_data_path)
predictions = predict(test_data,dictionary,num_reviews,tot,vocab_len)
x = {1:0,2:0,3:0,4:0,5:0}
for pr in predictions:
    x[pr] += 1
print(x)


# In[6]:


with open(output_path, 'w') as f:
    for item in predictions:
        f.write("%s\n" % item)

