#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import sys
import numpy as np
import argparse
from math import *
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
 
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
    def __init__(self, X, Y, K=5, weight_neighbors=False, verbose=False, perform_tuning=False):
        self.X_train = X   # for storing training matrix (nbexamples, d)
        self.Y_train = np.array(Y)   # list of corresponding gold classes (nbexamples,)
        self.K = K # nb neighbors to consider (look for 2/3/5/... nearest neighbors in order to identify the class)
        # if True, the nb of neighbors will be weighted by their similarity to the example to classify
        self.weight_neighbors = weight_neighbors
        self.verbose = verbose
        self.train_norms = np.linalg.norm(self.X_train, axis=1) # sqrt(sum(features**2)) for computing cosine similarity
        self.perform_tuning = perform_tuning # this is a boolean value, True for dev set, False for train set
 
    def cosine_similarity_matrix(self, X_test):
        """ A function to compute the cosine similarity between two vectors for every training and text example """
        product = np.dot(X_test, self.X_train.T) # a matrix of dot product between test and training examples
        test_norms = np.linalg.norm(X_test, axis=1) # since we know there are no null values, we can divide by the norm without checking for zero
        denom = np.outer(test_norms, self.train_norms) # outer product of test_norms and train_norms to change the shape to (n_test, n_train). before np.outer we have a matrix of (||test1||, ||test2|| ...) x (||train1||, ||train2|| ...) and we need a matrix of (test_norms[i] * train_norms[j]) for every i,j
        return product / denom # returns a matrix where rows are test examples and columns are training examples, so we compute the similarity for everything. we calculate everything at the same time to avoid loops (and, pehaps, this approach is more efficient since we just need to calculate everything once, not every loop?)
    

class Hyperparameters:
    def __init__(self, K=2, cos_or_dist=False, use_weight=False, use_tfidf=False):
        self.K = K # min 2, max unlimited
        self.cos_or_dist=cos_or_dist # False for cos similiarity, True for cos distance
        self.use_weight=use_weight # False for not weighting the neighbors, True for weighting
        self.use_tfidf=use_tfidf # False for keeping the original TF, True for using TF.IDF values

    # I decided to turn computing accuracy into a new function since I need it to choose best params
    def compute_accuracy(self, knn_model, X_test, Y_test, k=None, cos=None, weight=None, tfidf=None): # all hyperparams are optional
        # if no new parameters, use the default ones
        k = self.K if k is None else k
        cos = self.cos_or_dist if cos is None else cos
        weight = self.use_weight if weight is None else weight
        tfidf = self.use_tfidf if tfidf is None else tfidf

        X_train = knn_model.X_train
        X_eval = X_test
        
        if tfidf: # apply tfidf if tdfidf was passed True
            X_train_tfidf, X_eval_tfidf = apply_tfidf(X_train, X_eval)
            X_train = X_train_tfidf
            X_eval = X_eval_tfidf
       
        # repeat the code from lab 2
        sim_matrix = np.dot(X_eval, X_train.T)
        test_norms = np.linalg.norm(X_eval, axis=1)
        train_norms = np.linalg.norm(X_train, axis=1)
        denom = np.outer(test_norms, train_norms)
        sim_matrix = sim_matrix / denom
        if cos: # to calculate cosine distance, compute 1 - cosine similarity and choose the nearest neighbors by the smallest distance
            sim_matrix = 1 - sim_matrix
        predictions = []
        #for k_val in range(1, args.k + 1): # for k=5, it's in range(1, 6)
            #predictions = [] # predicted classes for all test examples
        for sim in sim_matrix: # each row of the sim_matrix is similarity between one test example and all training examples and we iterate through the rows(test examples), so sim = each row.
                                   # sim_matrix example:           (train1)   (train2)   (train3)
                                   #                     (test 1)   0.1234     0.5678     0.9876
                                   #                     (test 2)   0.1111     0.2222     0.3333
            # cos_distance = ascending order; cos_similarity = descending order
            order = np.argsort(sim)
            top_k_idx = order[:k] if cos else order[-k:][::-1] # looking at the current row's similiarities with all the other train examples, take k most similar indices (largest cosine similarity according to our big matrix), so take 5 indices if k=5
            class_scores = {} # here we store information about classes and their votes: for example, if k=5 and the closest neighbors are 'wolf', 'chocolate', 'dog', 'fox' and 'cake', then the class_scores will be {'animals': 3, 'sweets': 2}. animals wins by the majority vote.
            for idx in top_k_idx: # for every closest neighbor out of top-k...
                label = knn_model.Y_train[idx] # get the neighbor's gold_class at index idx
                score = sim[idx] if weight else 1 # we do not use the weight in this assignment, but if needed, the vote may be multiplied by the weight if it is more/less important than others. default value is 1, so everyone is equal
                class_scores[label] = class_scores.get(label,0) + score # accumulating the votes for each label, the score is += 1 if no special weight was assigned
            best_class = sorted(class_scores.items(), key=lambda x: (-x[1], x[0]))[0][0] # sorting for getting the highest first; in case of ties, sort alphabetically
            predictions.append(best_class) # store all the predictions to compare them with gold classes later
        correct = sum(p == y for p, y in zip(predictions, Y_test)) # computing the sum of predictions == true labels
        accuracy = (correct / len(Y_test)) * 100 # accuracy in percentage
        return(accuracy)

    def compare_accuracy(self, knn_model, X_dev, Y_dev, k, parameter, cos, weight, tfidf):
        # takes five arguments: k, first variant, second variant, two other hyperparameters (false by default)
        accuracy1 = self.compute_accuracy(knn_model, X_dev, Y_dev, k=k, cos=cos if parameter != 'cos' else False,
                                          weight=weight if parameter != 'weight' else False, tfidf=tfidf if parameter != 'tfidf' else False) #parameter_to_test=False, other_parameter1=False, other_parameter2=False)
        accuracy2 = self.compute_accuracy(knn_model, X_dev, Y_dev, k=k, cos=cos if parameter != 'cos' else True,
                                          weight=weight if parameter != 'weight' else True, tfidf=tfidf if parameter != 'tfidf' else True) # parameter_to_test=True, other_parameter1=False, other_parameter2=False)
        if accuracy1 > accuracy2: # simply choose the better accuracy
            return(False)
        else:
            return(True)
        
    def grid_search(self, knn_model, X_dev, Y_dev):
        # this function is not used in the code anymore since I had to rewrite for the DataFrames to work but I keep it here for legacy, it still works for searching best params
        # it prints out every accuracy one by one, collect_all_accuracies prints only the best combination
        # the logic here is as follows: first take k=1 and test which cos is better (true or false) by using compare_accuracy function (simply calculate which accuracy is higher)
        # choose the best cos and then test the best weight (true or false)
        # do the same for tfidf, using best results from two tests before
        # then increase k by 1 and repeat the same procedure
        # if new accuracy is better than the best so far, replace the new accuracy and store the new hyperparameters
        # this way, only the best combination of hyperparameters is returned
        best_hyperparams = {}
        best_accuracy = 0 
        K = self.K
        for K in range(1, 300):
            best_cos = self.compare_accuracy(knn_model, X_dev, Y_dev, K, parameter='cos', cos=False, weight=False, tfidf=False)
            best_weight = self.compare_accuracy(knn_model, X_dev, Y_dev, K, parameter='weight', cos=best_cos, weight=False, tfidf=False)
            best_tfidf = self.compare_accuracy(knn_model, X_dev, Y_dev, K, parameter='tfidf', cos=best_cos, weight=best_weight, tfidf=False)
 
            new_accuracy = self.compute_accuracy(knn_model, X_dev, Y_dev, k=K, cos=best_cos, weight=best_weight, tfidf=best_tfidf)

            if new_accuracy > best_accuracy:
                best_accuracy = new_accuracy
                new_hyperparams = {'K': K, 'cos_or_dist': best_cos, 'use_weight': best_weight, 'use_tfidf': best_tfidf}
                best_hyperparams.update(new_hyperparams)
                print(f"[Tuning] K={K}, cos={best_cos}, weight={best_weight}, tfidf={best_tfidf}, acc={new_accuracy:.2f}")
            else:
                print(f"![Decrease] with K={K}, acc={new_accuracy:.2f}")
        return best_hyperparams
    
    def collect_all_accuracies(self, knn_model, X_dev, Y_dev, max_k=300):
        """ Evaluate all combinations of hyperparameters for K in [1, max_k], returns a list of dicts with results. """
        results = []
        for K in range(1, max_k + 1):
            for cos_val, weight_val, tfidf_val in product([False, True], repeat=3):
                acc = self.compute_accuracy(
                    knn_model, X_dev, Y_dev,
                    k=K, cos=cos_val, weight=weight_val, tfidf=tfidf_val
                )
                results.append({
                    'K': K,
                    'cos_or_dist': cos_val,
                    'use_weight': weight_val,
                    'use_tfidf': tfidf_val,
                    'accuracy': acc
                })
        return results
 
def read_examples(infile):
    """ Reads a .examples file and returns an Examples instance. """
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

def apply_tfidf(X_train, X_test):
    TF = X_train / np.sum(X_train, axis=1, keepdims=True) # term frequency
    IDF = np.log(X_train.shape[0] / np.sum(X_train > 0, axis=0)) # inverse document frequency
    X_train_tfidf = TF * IDF
    X_test_tfidf = (X_test / X_test.sum(axis=1, keepdims=True)) * IDF
    return X_train_tfidf, X_test_tfidf

usage = """ DOCUMENT CLASSIFIER using K-NN algorithm
 
  prog [options] TRAIN_FILE TEST_FILE
 
  In TRAIN_FILE and TEST_FILE , each example starts with a line such as:
EXAMPLE_NB  1   GOLD_CLASS  earn
 
and continue providing the non-null feature values, e.g.:
declared    0.00917431192661
stake   0.00917431192661
reserve 0.00917431192661
...
 
"""
 
parser = argparse.ArgumentParser(usage = usage)
parser.add_argument('train_file', help='Examples\' file, used as neighbors', default=None)
parser.add_argument('test_file', help='Examples\' file, used for evaluation', default=None)
parser.add_argument("-k", '--k', default=1, type=int, help='Maximum number of nearest neighbors to consider (all values between 1 and K will be tested). Default=1')
parser.add_argument('--dev_file', help='Examples\ file used for tuning hyperparameters', default=None)
parser.add_argument('-v', '--verbose',action="store_true",default=False,help="If set, triggers a verbose mode. Default=False")
parser.add_argument('-w', '--weight_neighbors', action="store_true", default=False,help="If set, neighbors will be weighted when scoring classes. Default=False")
parser.add_argument('-p', '--perform_tuning', action="store_true", default=False,help="If set, grid-search will look for the best hyperparameters on dev set and perform the algorithm on test set using the best hyperparameters. Default=False")
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
                   verbose=args.verbose,
                   perform_tuning=args.perform_tuning)

if args.perform_tuning: # this is the new way of calculating best params and storing them for the graph later, imitating the grid_search function
    dev_examples = read_examples(args.dev_file)
    (X_dev, Y_dev) = build_matrices(dev_examples, w2i)
    print("Running grid-search on dev set...")
    hyperparams = Hyperparameters() #K=2, cos_or_dist=False, use_weight=False, use_tfidf=False)
    all_results = hyperparams.collect_all_accuracies(myclassifier, X_dev, Y_dev, max_k=300)
    df = pd.DataFrame(all_results)
    df['config'] = (
        'cos=' + df['cos_or_dist'].astype(str) + ', ' +
        'weight=' + df['use_weight'].astype(str) + ', ' +
        'tfidf=' + df['use_tfidf'].astype(str)
    )
    pivot_df = df.pivot(index='K', columns='config', values='accuracy')
    ax = pivot_df.plot(
        kind='line',
        figsize=(14, 8),
        title='K-NN Hyperparameter Tuning Results',
        xlabel='Number of Neighbors (K)',
        ylabel='Accuracy (%)',
        grid=True,
        marker='o',
        linewidth=2
    )
    ax.legend(title='Hyperparameter Configurations', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('knn_hyperparameter_tuning_results20.png', dpi=150)

    best_row = df.loc[df['accuracy'].idxmax()]
    best_hyperparams = {
        'K': int(best_row['K']),
        'cos_or_dist': bool(best_row['cos_or_dist']),
        'use_weight': bool(best_row['use_weight']),
        'use_tfidf': bool(best_row['use_tfidf'])
    }

    print(f"Best hyperparameters found: K={best_hyperparams['K']}, "
          f"cos_or_dist={best_hyperparams['cos_or_dist']}, "
          f"use_weight={best_hyperparams['use_weight']}, "
          f"use_tfidf={best_hyperparams['use_tfidf']} "
          f"(accuracy={best_row['accuracy']:.2f}%)")

    print("Evaluating on TEST set with best hyperparameters...")
    final_acc = hyperparams.compute_accuracy(
        myclassifier, X_test, Y_test,
        k=best_hyperparams['K'],
        cos=best_hyperparams['cos_or_dist'],
        weight=best_hyperparams['use_weight'],
        tfidf=best_hyperparams['use_tfidf']
    )
    print(f"FINAL ACCURACY ON TEST SET = {final_acc:6.2f}")

    #best_hyperparams = hyperparams.grid_search(myclassifier, X_dev, Y_dev)
    #print(f"Best hyperparameters found: K={best_hyperparams['K']}, cos_or_dist={best_hyperparams['cos_or_dist']}, use_weight={best_hyperparams['use_weight']}, use_tfidf={best_hyperparams['use_tfidf']}")
    #print("Evaluating on test set with the best hyperparameters...")
    #accuracy = hyperparams.compute_accuracy(myclassifier, X_test, Y_test, k=best_hyperparams['K'], cos=best_hyperparams['cos_or_dist'], weight=best_hyperparams['use_weight'], tfidf=best_hyperparams['use_tfidf'])
    #print(f"FINAL ACCURACY ON TEST SET = {accuracy:6.2f}")

else:
    print('Running with default parameters without tuning...')
    hyperparams = Hyperparameters(K=args.k, cos_or_dist=False, use_weight=args.weight_neighbors, use_tfidf=False)
    accuracy = hyperparams.compute_accuracy(myclassifier, X_test, Y_test, k=args.k, cos=False, weight=args.weight_neighbors, tfidf=False)
    print(f"ACCURACY ON TEST SET = {accuracy:6.2f}")

# with tuning: 89.28 (on dev_set: K=12, cos=False, weight=True, tfidf=False)
# without tuning: 85.18