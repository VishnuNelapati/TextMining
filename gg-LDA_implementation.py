"""
In this exercise, you will implement a one-epoch LDA training process
with a simple example. Please complete the functions below.

Note that in step 5, your answer may
differ from time to time depending on some random factors 
"""

import numpy as np
import string
import keras
from keras.preprocessing.sequence import pad_sequences

### step 1: backgrounds
# the document is x, there are 4 sentences
x = [["i","love","apple"],
     ["i","love","sport"],
     ["tennis","is","sport"],
     ["apple","is","a","fruit"]]
# suppose the number of topics is 3
nT = 3 

### step 2: defining the modules
def buildVocab(x):
    '''
    method:
        generate the vocabulary
    args:
        x: a list of word list
    return:
        vocab: a dictionary mapping from serial index to word
    expected output:
        {'love': 0, 'apple': 1, 'tennis': 2, 'is': 3, 
         'sport': 4, 'i': 5, 'a': 6, 'fruit': 7}
    '''
    ### Your Codes Here ### 
    # flatten list of lists to a list
    flat_list = [item for sublist in x for item in sublist]

    # remove duplicates
    ordered_set = {i:0 for i in flat_list}.keys()

    # create required output
    vocab = {v:i for i, v in enumerate(ordered_set)}

    return vocab

def translate(x,vocab):
    '''
    method:
        use the vocab dictionary to convert every word in x into index
    args:
        x: a list of word list
        vocab: a dictionary mapping from index to word, 
        i.e., {'love': 0, 'apple': 1, 'tennis': 2, 'is': 3, 
               'sport': 4, 'i': 5, 'a': 6, 'fruit': 7}
    return:
        the translation of the dataset x (still a list) 
        but in terms of the word index
    expectd output:
        [[5, 0, 1], [5, 0, 4], [2, 3, 4], [1, 3, 6, 7]]
    '''
    ### Your Codes Here ###
    flat_list = [item for sublist in x for item in sublist]
    # remove punctuation
    flat_list = [i for i in flat_list if i not in string.punctuation]
    # remove duplicates
    ordered_set = {i:0 for i in flat_list}.keys()
    # create required output
    translated = [[vocab[i] for i in sublist] for sublist in x]
    return translated

    

def padding(x,val=-1):
    '''
    method:
        use val to pad all sentences in x to the same length
        by default we padd with -1
    args:
        x: the translated x, i.e., the output of the translate function
        val: value used to pad the sequence
    return:
        padded x
    expected output: 
        [[ 5  0  1 -1]
         [ 5  0  4 -1]
         [ 2  3  4 -1]
         [ 1  3  6  7]]
    '''
    # pad to the end -> padding = "post"
    padded = pad_sequences(x,value=val,padding = "post")
    return padded

### step 3: initialization
vocab = buildVocab(x)
translated = translate(x,vocab)
padded = padding(translated)
# initialize the topic distribtion matrix tD randomly
# the shape of it should be (nD, nW)
# nD is the number of documents in x (i.e., 4)
# nW is the number of words in each padded document (i.e., 4)
tD = np.random.randint(0,nT,size = padded.shape) 

def updateTopicSummary(tD,padded,vocab,nT):
    '''
    method:
        update the topic summary matrix given the tD matrix
    args:
        tD: current topic distribution matrix, of shape (nD, nW)
            nD is the number of documents in x (i.e., 4)
            nW is the number of word in each padded document (i.e., 4)
        padded: padded documents
        vocab: a dictionary mapping from index to word
        nT: number of topics
    return:
        topic summary table of shape (nT, nV), nV is the vocab length
    '''
    nV = len(vocab)
    topicSummary = np.zeros((nT,nV))
    for i in range(topicSummary.shape[0]):
        for j in range(topicSummary.shape[1]):
            ### Your Codes Here ### 
            
            topicSummary[i,j] = np.sum(topicSummary[i,:]==j)/topicSummary.shape[1] # the prob that word j in vocab is about topic i
            
    return topicSummary

# calculate the initial topic summary matrix
topicSummary = updateTopicSummary(tD,padded,vocab,nT)

### step 4: training for one epoch
def trainOneEpoch(tD,topicSummary,padded,vocab,nT):
    
    for i in range(tD.shape[0]):
        for j in range(tD.shape[1]):
            wIdx = padded[i,j]
            prob = []
            for t in range(nT):
                probDocTopic = np.sum(tD[i,:]==t)/tD.shape[1]
                probWordTopic = topicSummary[t,wIdx]/(np.sum(topicSummary[:,wIdx]))
                prob.append(probDocTopic*probWordTopic)
            ### Your Codes Here ### 
            tD[i,j] = np.argmax(prob)
            # reassign the topic for document i, word j
            
    ### Your Codes Here ### 
    topicSummary = updateTopicSummary(tD, wIdx, vocab, nT)# update the topic summary table based on the new tD
    
    return topicSummary, tD


### step 5: check the updated topic distribution matrix and topic summary matrix
if __name__=="__main__":
    topicSummary,tD = trainOneEpoch(tD,topicSummary,padded,vocab,nT)
    print("topic summary matrix after training for one epoch")
    print(topicSummary)
    print("topic distribution matrix after training for one epoch")
    print(tD)