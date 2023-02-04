# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import math
from tqdm import tqdm
from collections import Counter
import reader
import nltk
"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
load_data calls the provided utility to load in the dataset.
You can modify the default values for stemming and lowercase, to improve performance when
    we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=True, lowercase=True, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=1.6, pos_prior=0.95,silently=False):
    
    print_paramter_vals(laplace,pos_prior)
    stopDic = {'if': 1, 'we':1, 'he':1, 'she':1, 'they':1, 'but':1, 'a':1, 'the':1, 'is':1, 'or':1, 'do':1, 'did':1, 'it':1, 'its':1, 'all':1, }
    #training phase: Making bag of words model
    bag = {} #key: word, value: [possum, negsum]
    ncount = 0 #number of neg words in training data
    nTypeC = 0
    pcount = 0 #number of positive words in training data
    pTypeC = 0
    #**^ is the count supposed be for number of words or type of words
    for j, rv in enumerate(train_set):
        i = 0
        while i < len(rv):
            currW = rv[i]
            if currW in stopDic:
                i+=1
                continue
            if currW not in bag:#if this is a new word
                if train_labels[j] == 1: #if it is a positive word
                    bag[currW] = [1,0]
                    pcount += 1
                    nTypeC += 1
                else: #if neg word
                    bag[currW] = [0,1]
                    ncount += 1
                    pTypeC += 1
            else: #if this word has been seen
                if train_labels[j] == 1:
                    bag[currW][0] += 1
                    pcount += 1
                else:
                    bag[currW][1] += 1
                    ncount += 1
            i+=1
   
    #laplace smoothing:
    nTypes = len(bag) #number of word types seen in TRAINING data **? Should this value be class specific? (split btw pos and neg)
    
    #development phase:
    yhats = []
    for doc in tqdm(dev_set,disable=silently):
        pPGW = math.log(pos_prior,2)
        pNGW = math.log((1-pos_prior),2)
        for word in doc:
            if word in stopDic:
                continue
            if word in bag: #known word
                #positive
                pWordGivenPos = (bag[word][0] + laplace) / (pcount + laplace * (nTypes + 1))
                pPGW += math.log(pWordGivenPos,2)
                #negative
                pWordGivenNeg = (bag[word][1] + laplace) / (ncount + laplace * (nTypes + 1))
                pNGW += math.log(pWordGivenNeg, 2)
            else:
                pUnkGivenPos = laplace / (pcount + laplace * (nTypes + 1))
                pPGW += math.log(pUnkGivenPos,2)
                pUnkGivenNeg = laplace / (ncount + laplace * (nTypes + 1))
                pNGW += math.log(pUnkGivenNeg, 2)
                    
        if pPGW > pNGW:
            yhats.append(1)
        else:
            if pPGW == pNGW:
                print("WTF")
            yhats.append(0)
    return yhats





def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=1.550, bigram_laplace=0.5, bigram_lambda=0.01,pos_prior=0.95, silently=False):
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    
    ##-------------------UNIGRAM------------------
    bag = {} #key: word, value: [possum, negsum]
    ncount = 0 #number of neg words in training data
    pcount = 0 #number of positive words in training data
    #**^ is the count supposed be for number of words or type of words
    
    for j, rv in enumerate(train_set):
        i = 0
        while i < len(rv):
            currW = rv[i]
          
            if currW not in bag:#if this is a new word
                if train_labels[j] == 1: #if it is a positive word
                    bag[currW] = [1,0]
                    pcount += 1
                else: #if neg word
                    bag[currW] = [0,1]
                    ncount += 1
            else: #if this word has been seen
                if train_labels[j] == 1:
                    bag[currW][0] += 1
                    pcount += 1
                else:
                    bag[currW][1] += 1
                    ncount += 1
            i+=1
            
    ##-------------------BIGRAM------------------
    bibag = {} #key: word, value: [possum, negsum]
    nPairCount = 0 #number of neg words in training data
    pPairCount = 0 #number of positive words in training data
    #**^ is the count supposed be for number of words or type of words
    for j, rv in enumerate(train_set):
        i = 0
        while i < len(rv)-1:
            currW = rv[i]
         
            currPair = rv[i] + " " + rv[i+1]
            if currPair not in bibag:#if this is a new pair
                if train_labels[j] == 1: #if it is a positive pair
                    bibag[currPair] = [1,0]
                    pPairCount += 1
                else: #if neg pair
                    bibag[currPair] = [0,1]
                    nPairCount += 1
            else: #if this word has been seen
                if train_labels[j] == 1:
                    bibag[currPair][0] += 1
                    pPairCount += 1
                else:
                    bibag[currPair][1] += 1
                    nPairCount += 1
            i+=1
            
            
    #laplace smoothing:
    nWordTypes = len(bag) #number of word types seen in TRAINING data **? Should this value be class specific? (split btw pos and neg)
    nPairTypes = len(bibag)
    print('normal: ', nWordTypes, ' bi: ', nPairTypes)
    
    #development phase:
    yhats = []
    for doc in tqdm(dev_set,disable=silently):
        pPGW = math.log(pos_prior,2)
        pNGW = math.log((1-pos_prior),2)
        pPGP = math.log(pos_prior,2)
        pNGP = math.log((1-pos_prior),2)
        #finding unigram probability
        for word in doc:
  
            if word in bag: #known word
                #positive
                pWordGivenPos = (bag[word][0] + unigram_laplace) / (pcount + unigram_laplace * (nWordTypes + 1))
                pPGW += math.log(pWordGivenPos,2)
                #negative
                pWordGivenNeg = (bag[word][1] + unigram_laplace) / (ncount + unigram_laplace * (nWordTypes + 1))
                pNGW += math.log(pWordGivenNeg, 2)
            else:
                pUnkGivenPos = unigram_laplace / (pcount + unigram_laplace * (nWordTypes + 1))
                pPGW += math.log(pUnkGivenPos,2)
                pUnkGivenNeg = unigram_laplace / (ncount + unigram_laplace * (nWordTypes + 1))
                pNGW += math.log(pUnkGivenNeg, 2)

        #finding bigram probability
        i = 0
        trueCount = 0
        while i < len(doc) - 1:
            currW = doc[i]
 
            currPair = doc[i] + " " + doc[i+1]
            if currPair in bibag:
                trueCount += 1
                #positive
                pPairGivenPos = (bibag[currPair][0] + bigram_laplace) / (pPairCount + bigram_laplace * (nPairTypes + 1))
                pPGP += math.log(pPairGivenPos,2)
                #negative
                pPairGivenNeg = (bibag[currPair][1] + bigram_laplace) / (nPairCount + bigram_laplace * (nPairTypes + 1))
                pNGP += math.log(pPairGivenNeg, 2)
            else:
                pUnkGivenPos = unigram_laplace / (pcount + unigram_laplace * (nWordTypes + 1))
                pPGP += math.log(pUnkGivenPos,2)
                pUnkGivenNeg = unigram_laplace / (ncount + unigram_laplace * (nWordTypes + 1))
                pNGP += math.log(pUnkGivenNeg, 2)

            i += 1
            
        totalpp = (1-bigram_lambda) * pPGW + bigram_lambda * pPGP
        totalnp = (1-bigram_lambda) * pNGW + bigram_lambda * pNGP
        
        if totalpp > totalnp:
            yhats.append(1)
        else:
            if totalpp == totalnp:
                print("WTF")
            yhats.append(0)
    print(trueCount)
    return yhats

