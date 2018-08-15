import sys
import numpy as np

def read_data(fname):
    with open(fname, 'r', encoding="utf-8") as fp:
        temp = fp.readlines()
    temp = [item.split(",")[0] for item in temp]
    return np.array(temp).astype(np.int)

def compute_accuracy(true_labels, predicted_labels):
    num_instances = true_labels.size
    return np.sum(true_labels==predicted_labels)/num_instances

targets = read_data(sys.argv[1])
predicted = np.genfromtxt(sys.argv[2], dtype=np.int) 
print(compute_accuracy(targets, predicted))
