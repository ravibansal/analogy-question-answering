
# coding: utf-8

# Deep Learning Programming Assignment 2
# --------------------------------------
# Name: Ravi Bansal
# Roll No.: 13CS30026
# 
# Submission Instructions:
# 1. Fill your name and roll no in the space provided above.
# 2. Name your folder in format <Roll No>_<First Name>.
#     For example 12CS10001_Rohan
# 3. Submit a zipped format of the file (.zip only).
# 4. Submit all your codes. But do not submit any of your datafiles
# 5. From output files submit only the following 3 files. simOutput.csv, simSummary.csv, analogySolution.csv
# 6. Place the three files in a folder "output", inside the zip.

# In[59]:

import gzip
import os
import Analogy_data
import Analogy_train
import csv
import numpy as np
import tensorflow as tf
## paths to files. Do not change this
simInputFile = "Q1/word-similarity-dataset"
analogyInputFile = "Q1/word-analogy-dataset"
vectorgzipFile = "Q1/glove.6B.300d.txt.gz"
vectorTxtFile = "Q1/glove.6B.300d.txt"   # If you extract and use the gz file, use this.
analogyTrainPath = "Q1/wordRep/"
simOutputFile = "Q1/simOutput.csv"
simSummaryFile = "Q1/simSummary.csv"
anaSoln = "Q1/analogySolution.csv"
Q4List = "Q4/wordList.csv"




# In[ ]:

# Similarity Dataset
simDataset = [item.split(" | ") for item in open(simInputFile).read().splitlines()]
# Analogy dataset
analogyDataset = [[stuff.strip() for stuff in item.strip('\n').split('\n')] for item in open(analogyInputFile).read().split('\n\n')]

# In[ ]:

# Dictionary of training pairs for the analogy task
trainDict = dict()
for subDirs in os.listdir(analogyTrainPath):
    for files in os.listdir(analogyTrainPath+subDirs+'/'):
        f = open(analogyTrainPath+subDirs+'/'+files).read().splitlines()
        trainDict[files] = f
print len(trainDict.keys())

def vectorExtract(simD = simDataset, anaD = analogyDataset, vect = vectorgzipFile):
    simList_ = [stuff.lower() for item in simD for stuff in item]
    simList = [stuff.strip(' ') for stuff in simList_]
    analogyList = [thing.lower() for item in anaD for stuff in item[0:6] for thing in stuff.split()] 
    simList.extend(analogyList)
    wordRepList = []
    for keys in trainDict:
        for line in trainDict[keys]:
            wv = line.split("\t")
            if len(wv) == 2:
                wordRepList.append(wv[0].lower())
                wordRepList.append(wv[1].lower())
    simList.extend(wordRepList)
    wordList = set(simList)
    print len(wordList)
    wordDict = dict()
    
    vectorFile = gzip.open(vect, 'r')
    for line in vectorFile:
        if line.split()[0].strip() in wordList:
            wordDict[line.split()[0].strip()] = line.split()[1:]
    
    
    vectorFile.close()
    print 'retrieved', len(wordDict.keys())
    return wordDict

# Extracting Vectors from Analogy and Similarity Dataset
validateVectors = vectorExtract()

# In[58]:


def similarityTask(inputDS = simDataset, outputFile = simOutputFile, summaryFile=simSummaryFile, vectors=validateVectors):
    # print 'hello world'

    """
    Output simSummary.csv in the following format
    Distance Metric, Number of questions which are correct, Total questions evalauted, MRR
    C, 37, 40, 0.61
    """

    """
    Output a CSV file titled "simOutput.csv" with the following columns

    file_line-number, query word, option word i, distance metric(C/E/M), similarity score 

    For the line "rusty | corroded | black | dirty | painted", the outptut will be

    1,rusty,corroded,C,0.7654
    1,rusty,dirty,C,0.8764
    1,rusty,black,C,0.6543


    The order in which rows are entered does not matter and so do Row header names. Please follow the order of columns though.
    """
    C = 0
    CMRR = 0.0
    E = 0
    EMRR = 0.0
    M = 0
    MMRR = 0.0
    f_output = open(outputFile, 'wb')
    output_writer = csv.writer(f_output, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    f_summary = open(summaryFile, 'wb')
    summary_writer = csv.writer(f_summary, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    count = 0
    for vec in simDataset:
        flag = 0
        for word in vec:
            if word.lower().strip(' ') not in validateVectors:
                flag = 1
                break
        if flag == 1:
            continue
        rootword = vec[0].lower().strip(' ')
        cosRank = []
        eucRank = []
        manRank = []
        count += 1
        for j in xrange(1,len(vec)):
            a = np.asarray(validateVectors[rootword.lower().strip(' ')],dtype=float)
            b = np.asarray(validateVectors[vec[j].lower().strip(' ')],dtype=float)
            # print a
            cosSim = np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b) )
            eucDis = np.linalg.norm(a-b)
            manDis = np.sum(np.absolute(a-b))
            
            output_writer.writerow([count,rootword.lower().strip(' '), vec[j].lower().strip(' '), 'C',cosSim])
            output_writer.writerow([count,rootword.lower().strip(' '), vec[j].lower().strip(' '),'E',eucDis])
            output_writer.writerow([count,rootword.lower().strip(' '), vec[j].lower().strip(' '),'M',manDis])

            cosRank.append((cosSim,j))
            eucRank.append((eucDis,j))
            manRank.append((manDis,j))
        cosRank.sort()
        cosRank.reverse()
        eucRank.sort()
        manRank.sort()
        
        for x in xrange(0,len(cosRank)):
            if cosRank[x][1] == 1:
                CMRR += 1.0/(x+1)
        
        for x in xrange(0,len(eucRank)):
            if eucRank[x][1] == 1:
                EMRR += 1.0/(x+1)
        
        for x in xrange(0,len(manRank)):
            if manRank[x][1] == 1:
                MMRR += 1.0/(x+1)
        
        if cosRank[0][1] == 1:
            C += 1
        
        if eucRank[0][1] == 1:
            E += 1
        
        if manRank[0][1] == 1:
            M += 1

    summary_writer.writerow(['C',C,count,CMRR/count])
    summary_writer.writerow(['E',E,count,EMRR/count])
    summary_writer.writerow(['M',M,count,MMRR/count])
    
# In[ ]:

def analogyTask(inputDS=analogyDataset,outputFile = anaSoln ): # add more arguments if required
    
    """
    Output a file, analogySolution.csv with the following entris
    Query word pair, Correct option, predicted option    
    """
    accuracy = Analogy_data.dataset_init(inputDS, trainDict, validateVectors, outputFile)
    
    return accuracy #return the accuracy of your model after 5 fold cross validation



# In[60]:

def derivedWordTask(inputFile = Q4List):
    # print 'hello world'
    
    """
    Output vectors of 3 files:
    1)AnsFastText.txt - fastText vectors of derived words in wordList.csv
    2)AnsLzaridou.txt - Lazaridou vectors of the derived words in wordList.csv
    3)AnsModel.txt - Vectors for derived words as provided by the model
    
    For all the three files, each line should contain a derived word and its vector, exactly like 
    the format followed in "glove.6B.300d.txt"
    
    word<space>dim1<space>dim2........<space>dimN
    charitably 256.238 0.875 ...... 1.234
    
    """
    
    """
    The function should return 2 values
    1) Averaged cosine similarity between the corresponding words from output files 1 and 3, as well as 2 and 3.
    
        - if there are 3 derived words in wordList.csv, say word1, word2, word3
        then find the cosine similiryt between word1 in AnsFastText.txt and word1 in AnsModel.txt.
        - Repeat the same for word2 and word3.
        - Average the 3 cosine similarity values
        - DO the same for word1 to word3 between the files AnsLzaridou.txt and AnsModel.txt 
        and average the cosine simialities for valuse so obtained
        
    """
    fastDict = dict()
    with open('Q4/fastText_vectors.txt','r') as f:
        for line in f:
            fastDict[line.split()[0].strip()] = line.split()[1:]

    lzaDict = dict()
    with open('Q4/vector_lazaridou.txt', 'r') as f:
        for line in f:
            line = line.replace(" ","").strip()
            vec = line.split(',')
            word = vec[0].split('[')[0]
            vec[0] = vec[0].split('[')[1]
            vec[-1] = vec[-1].split(']')[0] 
            lzaDict[word] = vec

    affixDict = dict()
    with open(Q4List,'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if row[0]=='':
                continue
            if row[1] not in affixDict:
                affixDict[row[1]] = []
            affixDict[row[1]].append((row[3],row[2]))

    f1 = open('Q4/AnsFastText.txt','w')
    f2 = open('Q4/AnsLzaridou.txt','w')
    f3 = open('Q4/AnsModel_FastText.txt','w')
    f4 = open('Q4/AnsModel_Lzaridou.txt','w')

    cosVal1 = 0.0
    count1 = 0
    sess = tf.InteractiveSession()
    inputDim = len(fastDict[fastDict.keys()[0]])
    hiddenDim = 100
    outputDim = len(fastDict[fastDict.keys()[0]])

    X = tf.placeholder(tf.float32,[None,inputDim])
    Y = tf.placeholder(tf.float32,[None,outputDim])
    W1 = tf.Variable(tf.random_uniform([inputDim,hiddenDim],-(1.0/inputDim)**(1.0/2),(1.0/inputDim)**(1.0/2),tf.float32))
    b1 = tf.Variable(tf.zeros([1,hiddenDim]))
    W2 = tf.Variable(tf.random_uniform([hiddenDim,outputDim],-(1.0/hiddenDim)**(1.0/2),(1.0/hiddenDim)**(1.0/2),tf.float32))
    b2 = tf.Variable(tf.zeros([1,outputDim]))

    Z = tf.matmul(X,W1) + b1
    Y_ = tf.matmul(tf.nn.relu(Z),W2) + b2

    loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(Y, Y_)))
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

    #fastText
    for key in affixDict:
        # print key,len(affixDict[key])
        sess.run(tf.global_variables_initializer())

        dataX = []
        dataY = []
        words = []
        for word in affixDict[key]:
            if word[0] in fastDict and word[1] in fastDict:
                dataX.append(fastDict[word[0]])
                dataY.append(fastDict[word[1]])
                words.append(word[1])
        dataX = np.asarray(dataX,dtype=float)
        dataY = np.asarray(dataY,dtype=float)
        
        total_size = dataX.shape[0]
        if total_size < 5:
            continue

        for i in xrange(0,total_size,total_size/5):
            end = i + total_size/5
            if i + 2*total_size/5 > total_size:
                end = total_size
            
            testX = dataX[i:end]
            testY = dataY[i:end]
            wordY = words[i:end]
            trainX = np.concatenate((dataX[0:i],dataX[end:total_size]))
            trainY = np.concatenate((dataY[0:i],dataY[end:total_size]))
            
            batch_size = 50
            
            for k in xrange(50):
                perm = np.random.permutation(trainX.shape[0])
                trainX = trainX[perm]
                trainY = trainY[perm]
                for j in xrange(0,trainX.shape[0],batch_size):
                    end_temp = j + batch_size
                    if end_temp > trainX.shape[0]:
                        end_temp = trainX.shape[0]
                    train_step.run(feed_dict={X:trainX[j:end_temp].reshape(-1,inputDim),Y:trainY[j:end_temp].reshape(-1,outputDim)})

            outputY = sess.run(Y_, feed_dict={X:testX.reshape(-1,inputDim)})
            
            outputY = np.asarray(outputY,dtype=float)

            for ct in xrange(0,testY.shape[0]):
                f1.write(wordY[ct] + ' ')
                f1.write("".join(" ".join(map(str, testY[ct]))))
                f1.write('\n')

            for ct in xrange(0,testY.shape[0]):
                f3.write(wordY[ct] + ' ')
                f3.write("".join(" ".join(map(str, outputY[ct]))))
                f3.write('\n')

            row_sums_1 = np.sum(np.abs(outputY)**2,axis=-1)**(1./2)
            outputY = outputY / row_sums_1[:, np.newaxis]


            row_sums_2 = np.sum(np.abs(testY)**2,axis=-1)**(1./2)
            testY = testY / row_sums_2[:, np.newaxis]

            simMat = np.dot(outputY,testY.T)
            cosVal1 += np.trace(simMat)
            count1 += (end-i)
        # print cosVal1/count1
    cosVal1 = cosVal1/count1
    print 'cosVal1: ' + str(cosVal1)
    sess.close()

    sess2 = tf.InteractiveSession()
    inputDim = len(lzaDict[lzaDict.keys()[0]])
    hiddenDim = 400
    outputDim = len(lzaDict[lzaDict.keys()[0]])

    X = tf.placeholder(tf.float32,[None,inputDim])
    Y = tf.placeholder(tf.float32,[None,outputDim])
    W1 = tf.Variable(tf.random_uniform([inputDim,hiddenDim],-(1.0/inputDim)**(1.0/2),(1.0/inputDim)**(1.0/2),tf.float32))
    b1 = tf.Variable(tf.zeros([1,hiddenDim]))
    W2 = tf.Variable(tf.random_uniform([hiddenDim,outputDim],-(1.0/hiddenDim)**(1.0/2),(1.0/hiddenDim)**(1.0/2),tf.float32))
    b2 = tf.Variable(tf.zeros([1,outputDim]))

    Z = tf.matmul(X,W1) + b1
    Y_ = tf.matmul(tf.nn.relu6(Z),W2) + b2

    loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(Y, Y_)))

    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

    count2 = 0
    cosVal2 = 0.0
    for key in affixDict:
        # print key,len(affixDict[key])
        sess2.run(tf.global_variables_initializer())

        dataX = []
        dataY = []
        words = []
        for word in affixDict[key]:
            if word[0] in lzaDict and word[1] in lzaDict:
                dataX.append(lzaDict[word[0]])
                dataY.append(lzaDict[word[1]])
                words.append(word[1])
        dataX = np.asarray(dataX,dtype=float)
        dataY = np.asarray(dataY,dtype=float)
        
        total_size = dataX.shape[0]
        if total_size < 5:
            continue

        for i in xrange(0,total_size,total_size/5):
            end = i + total_size/5
            if i + 2*total_size/5 > total_size:
                end = total_size
            
            testX = dataX[i:end]
            testY = dataY[i:end]
            wordY = words[i:end]
            trainX = np.concatenate((dataX[0:i],dataX[end:total_size]))
            trainY = np.concatenate((dataY[0:i],dataY[end:total_size]))
            
            batch_size = 50
            
            for k in xrange(100):
                perm = np.random.permutation(trainX.shape[0])
                trainX = trainX[perm]
                trainY = trainY[perm]
                for j in xrange(0,trainX.shape[0],batch_size):
                    end_temp = j + batch_size
                    if end_temp > trainX.shape[0]:
                        end_temp = trainX.shape[0]
                    train_step.run(feed_dict={X:trainX[j:end_temp].reshape(-1,inputDim),Y:trainY[j:end_temp].reshape(-1,outputDim)})

            outputY = sess2.run(Y_, feed_dict={X:testX.reshape(-1,inputDim)})
            
            outputY = np.asarray(outputY,dtype=float)

            for ct in xrange(0,testY.shape[0]):
                f2.write(wordY[ct] + ' ')
                f2.write("".join(" ".join(map(str, testY[ct]))))
                f2.write('\n')

            for ct in xrange(0,testY.shape[0]):
                f4.write(wordY[ct] + ' ')
                f4.write("".join(" ".join(map(str, outputY[ct]))))
                f4.write('\n')

            row_sums_1 = np.sum(np.abs(outputY)**2,axis=-1)**(1./2)
            outputY = outputY / row_sums_1[:, np.newaxis]


            row_sums_2 = np.sum(np.abs(testY)**2,axis=-1)**(1./2)
            testY = testY / row_sums_2[:, np.newaxis]

            simMat = np.dot(outputY,testY.T)
            cosVal2 += np.trace(simMat)
            count2 += (end-i)
        # print cosVal2/count2
    cosVal2 = cosVal2/count2
    print 'cosVal2: ' + str(cosVal2)
    sess2.close()
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    return cosVal1,cosVal2
    


# In[ ]:

def main():
    similarityTask()
    anaSim = analogyTask()
    print anaSim
    derCos1,derCos2 = derivedWordTask()

if __name__ == '__main__':
    main()
