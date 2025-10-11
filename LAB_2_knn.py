#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import argparse
from math import *

class Examples:
    """
    a batch of examples:
    One example is 
    - a BOW vector represented as a python dictionary, for features with non-null values only
    - a gold class

    dict_vectors = list of dictionary BOW vector
    gold_classes = list of gold classes
    """
    def __init__(self):
        self.gold_classes = [] # the correct classes (labels)
        self.dict_vectors = [] # list of {word: value} dictionaries

class KNN:
    """
    K-NN for document classification (multiclass)

    members = 

    X_train = matrix of training example vectors
    Y_train = list of corresponding gold classes

    K = maximum number of neighbors to consider

    """
    def __init__(self, X, Y, K=3, weight_neighbors=False, verbose=False):
        self.X_train = X   # for storing training matrix (nbexamples, d)
        self.Y_train = np.array(Y)   # list of corresponding gold classes (nbexamples,)
        self.K = K # nb neighbors to consider (look for 2/3/5/... nearest neighbors in order to identify the class)
        # if True, the nb of neighbors will be weighted by their similarity to the example to classify
        self.weight_neighbors = weight_neighbors
        self.verbose = verbose
        self.train_norms = np.linalg.norm(self.X_train, axis=1) # sqrt(sum(features**2)) for computing cosine similarity
    
    def cosine_similarity_matrix(self, X_test):
        """ A function to compute the cosine similarity between two vectors for every training and text example """
        product = np.dot(X_test, self.X_train.T) # a matrix of dot product between test and training examples
        test_norms = np.linalg.norm(X_test, axis=1) # since we know there are no null values, we can divide by the norm without checking for zero
        denom = np.outer(test_norms, self.train_norms) # outer product of test_norms and train_norms to change the shape to (n_test, n_train). before np.outer we have a matrix of (||test1||, ||test2|| ...) x (||train1||, ||train2|| ...) and we need a matrix of (test_norms[i] * train_norms[j]) for every i,j
        return product / denom # returns a matrix where rows are test examples and columns are training examples, so we compute the similarity for everything. we calculate everything at the same time to avoid loops (and, pehaps, this approach is more efficient since we just need to calculate everything once, not every loop?)

def read_examples(infile):
    """ Reads a .examples file and returns an Examples instance.    """
    stream = open(infile)
    examples = Examples()
    dict_vector = None
    while 1:
        line = stream.readline()
        if not line:
            break
        line = line[0:-1]
        if line.startswith("EXAMPLE_NB"): # this are the lines with gold classes
            if dict_vector != None:
                examples.dict_vectors.append(dict_vector)
            dict_vector = {} # when the new example starts, create a fresh new dictionary
            cols = line.split('\t')
            gold_class = cols[3] # we split tabs by \t and take the 4th column containing the gold class (cocoa, grain...)
            examples.gold_classes.append(gold_class) # for storing gold classes in a list
        elif line:# and dict_vector != None:
            (wordform, val) = line.split('\t')
            dict_vector[wordform] = float(val)    # here we store the weights (0.00537 for 'limited'...)
    if dict_vector != None:
        examples.dict_vectors.append(dict_vector)
    return examples # here we store examples and their gold classes

def build_matrices(examples, w2i):
    # TODO
    # list of dictionaries -> matrix
    nb_examples = len(examples.dict_vectors) # number of examples = number of rows in the matrix
    vocab_size = len(w2i) # number of unique words or distinct features = number of columns in the matrix
    X = np.zeros([nb_examples, vocab_size]) # initializing the matrix with zeros
    y = np.array(examples.gold_classes) # list of labels -> array
    for i, doc in enumerate(examples.dict_vectors): # we fill the matrix here: for each example(row) and each word in (word, val), if the word is in w2i we put its val (weight from the examples) to the matrix
        for word, val in doc.items():
            if word in w2i:
                X[i, w2i[word]] = val
    return (X, y) # y = classes, X =[j, i] value of word i in example j, e.g. y=['cocoa', 'grain'...]
# X:                 (limited)     (now)    (currently)
#          (cocoa)    0.00395     0.00197    0.00197
#          (grain)    0.00211        0          0

usage = """ DOCUMENT CLASSIFIER using K-NN algorithm

  prog [options] TRAIN_FILE TEST_FILE

  In TRAIN_FILE and TEST_FILE , each example starts with a line such as:
EXAMPLE_NB	1	GOLD_CLASS	earn

and continue providing the non-null feature values, e.g.:
declared	0.00917431192661
stake	0.00917431192661
reserve	0.00917431192661
...

"""

parser = argparse.ArgumentParser(usage = usage)
parser.add_argument('train_file', help='Examples\' file, used as neighbors', default=None)
parser.add_argument('test_file', help='Examples\' file, used for evaluation', default=None)
parser.add_argument("-k", '--k', default=1, type=int, help='Maximum number of nearest neighbors to consider (all values between 1 and K will be tested). Default=1')
parser.add_argument('-v', '--verbose',action="store_true",default=False,help="If set, triggers a verbose mode. Default=False")
parser.add_argument('-w', '--weight_neighbors', action="store_true", default=False,help="If set, neighbors will be weighted when scoring classes. Default=False")
# we run the program by specifying train and test files paths and the number of neighbors
args = parser.parse_args()

#------------------------------------------------------------
# Reading training and test examples :

train_examples = read_examples(args.train_file)
test_examples = read_examples(args.test_file)
# examples files consist of lines with words and their example IDs, the words are grouped by the classes (cocoa, grain etc.)
#------------------------------------------------------------
# Building indices for vocabulary in TRAINING examples

#TODO
# For every new word, assign a number (for the first word it's 0).
# If word already exists in w2i, pass
w2i = {}
for doc in train_examples.dict_vectors:
    for word in doc.keys():
        if word not in w2i:
            w2i[word] = len(w2i)

#------------------------------------------------------------
# Organize the data into two matrices for document vectors
#                   and two lists for the gold classes
(X_train, Y_train) = build_matrices(train_examples, w2i)
(X_test, Y_test) = build_matrices(test_examples, w2i)
print(f"Training matrix has shape {X_train.shape}")
print(f" Testing matrix has shape {X_test.shape}")
print("Evaluating on test...")
#------------------------------------------------------------
# creating KNN classifier and computing cosine similarities matrix
myclassifier = KNN(X = X_train,
                   Y = Y_train,
                   K = args.k,
                   weight_neighbors = args.weight_neighbors,
                   verbose=args.verbose)
sim_matrix = myclassifier.cosine_similarity_matrix(X_test)
# for each test example, we find k most similar training examples (with shortest cosine distance), collect their classes, pick the class with most votes (weighted if needed)
# then we compare the chosen class with the gold class and compute accuracy
for k_val in range(1, args.k + 1): # for k=5, it's in range(1, 6)
    predictions = [] # predicted classes for all test examples
    for sim in sim_matrix: # each row of the sim_matrix is similarity between one test example and all training examples and we iterate through the rows(test examples), so sim = each row.
        # sim_matrix example:           (train1)   (train2)   (train3)
        #                     (test 1)   0.1234     0.5678     0.9876
        #                     (test 2)   0.1111     0.2222     0.3333
        top_k_idx = np.argsort(sim)[-k_val:][::-1] # looking at the current row's similiarities with all the other train examples, take k most similar indices (largest cosine similarity according to our big matrix), so take 5 indices if k=5
        class_scores = {} # here we store information about classes and their votes: for example, if k=5 and the closest neighbors are 'wolf', 'chocolate', 'dog', 'fox' and 'cake', then the class_scores will be {'animals': 3, 'sweets': 2}. animals wins by the majority vote.
        for idx in top_k_idx: # for every closest neighbor out of top-k...
            label = myclassifier.Y_train[idx] # get the neighbor's gold_class at index idx
            score = sim[idx] if myclassifier.weight_neighbors else 1 # we do not use the weight in this assignment, but if needed, the vote may be multiplied by the weight if it is more/less important than others. default value is 1, so everyone is equal
            class_scores[label] = class_scores.get(label,0) + score # accumulating the votes for each label, the score is += 1 if no special weight was assigned
        best_class = sorted(class_scores.items(), key=lambda x: (-x[1], x[0]))[0][0] # sorting for getting the highest first; in case of ties, sort alphabetically
        predictions.append(best_class) # store all the predictions to compare them with gold classes later
    correct = sum(p == y for p, y in zip(predictions, Y_test)) # computing the sum of predictions == true labels
    accuracy = (correct / len(Y_test)) * 100 # accuracy in percentage
    print(f"ACCURACY FOR K = {k_val:2d} = {accuracy:6.2f} ({correct} / {len(Y_test)}) (use_weight = {args.weight_neighbors})")

# usage example: output for k=5 is
# ACCURACY FOR K =  1 =  78.50 (157 / 200) (use_weight = False)
# ACCURACY FOR K =  2 =  77.00 (154 / 200) (use_weight = False)
# ACCURACY FOR K =  3 =  78.00 (156 / 200) (use_weight = False)
# ACCURACY FOR K =  4 =  82.00 (164 / 200) (use_weight = False)
# ACCURACY FOR K =  5 =  80.00 (160 / 200) (use_weight = False)
# so the algorithm performs best with 4 neighbors to consider 
# the worst result: k=2