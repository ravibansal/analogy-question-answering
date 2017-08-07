import numpy as np
import os
import Analogy_train
import csv

def load_data(inputDS, trainDict, wordDict):
	dict={}
	dataX_word = []
	for row in inputDS:
		ex = []
		count = 0
		for line in row: 
			arr = line.split(" ")
			arr = filter(None,arr)
			if len(arr)==1 and arr[0]!='':
				# dataY.append(int(ord(arr[0])-ord('a')))
				dataX_word.append(ex)
				temp_swap = dataX_word[count][int(ord(arr[0])-ord('a'))+1]
				dataX_word[count][int(ord(arr[0])-ord('a'))+1] = dataX_word[count][1]
				dataX_word[count][1] = temp_swap
				count += 1
				ex=[]
			elif len(arr)==2:
				arr[0] = arr[0].lower()
				arr[1] = arr[1].lower()
				ex.append([arr[0],arr[1]])

	list_words = []
	for keys in trainDict:
		one_file = []
		for line in trainDict[keys]:
			wv = line.split("\t")
			wv[-1] = wv[-1].rstrip()
			wv = filter(None,wv)
			if len(wv)==2:
				wv[0] = wv[0].lower()
				wv[1] = wv[1].lower()
				one_file.append(wv)
		list_words.append(one_file)
	# print 'Number of word pairs in each file: '
	dataX=[]
	dataX_save = []
	count=0
	for i in dataX_word:
		ex=[]
		flag=0
		for j in i:
			if j[0] not in wordDict or j[1] not in wordDict:
				flag=1
				break
			ex.append(wordDict[j[1]] + wordDict[j[0]])
			# ex.append([a-b for a,b in zip(dict[j[1]],dict[j[0]])])
		if flag==0:
			ex_ct = []
			for ii in xrange(1,len(ex)):
				ex_ct.append(ex[0] + ex[ii])
			dataX.append(ex_ct)
			dataX_save.append(i)
			count=count+1
	
	listX=[]
	for i in list_words:
		one_list = []
		for j in i:
			if j[0].lower() in wordDict and j[1].lower() in wordDict:
				one_list.append(wordDict[j[1].lower()] + wordDict[j[0].lower()])
		if len(one_list) > 0:
			listX.append(one_list)

	return np.asarray(dataX),listX, np.asarray(dataX_save)


def dataset_init(inputDS, trainDict, wordDict, outputFile='output/analogySolution.csv'):
	dataX,listX,dataX_save = load_data(inputDS, trainDict, wordDict)
	num_data = dataX.shape[0]
	# print 'dataX Shape: ' + str(dataX.shape)
	# print 'Listx Shape ' + str(len(listX)) + ', ' + str(len(listX[0])) + ', ' + str(len(listX[0][0])) 
	trainX = []
	for i in listX:
		for j in i:
			one_data = []
			one_data.append( j + i[np.random.choice(len(i),1)[0]])
			count = 0
			while count < 4:
				k = listX[np.random.choice(int(len(listX)),1)[0]]
				if k != i:
					one_data.append( j + k[np.random.choice(int(len(k)),1)[0]])
					count = count + 1
			trainX.append(one_data)
	trainX = np.array(trainX)
	perm = np.random.permutation(trainX.shape[0])
	trainX = trainX[perm]
	# print 'trainX Shape: ' + str(trainX.shape)
	total_size = trainX.shape[0]
	avg_accuracy = 0.0
	iteration = 0
	for i in xrange(0,total_size,total_size/5):
		end = i + total_size/5
		if end > total_size:
			break
		if (i + 2*total_size/5) > total_size:
			end = total_size
		testX = trainX[i:end]
		# trainX_set = np.empty([0,trainX.shape[1],trainX.shape[2]])
		trainX_set = np.concatenate((trainX[0:i],trainX[end:total_size]))
		Analogy_train.train(trainX_set)
		accuracy_temp = Analogy_train.test(testX)
		print "\nAccuracy for iteration %d in 5-fold cross validation: %lf%%" %((iteration+1) ,accuracy_temp)
		avg_accuracy += accuracy_temp
		iteration += 1
	print "\nAnalogy Cross Validation accuracy: %lf%%" % (avg_accuracy/5)
	Analogy_train.train(trainX)
	accuracy, labels = Analogy_train.test(dataX,1)
	print "\nAnalogy Validation set accuracy: %lf%%" % accuracy
	if not os.path.exists('output'):
		os.makedirs('output')
	with open(outputFile, 'wb') as csvfile:
		wordwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting = csv.QUOTE_MINIMAL)
		for i in xrange(0,dataX_save.shape[0]):
			wordwriter.writerow([str(dataX_save[i][0][0]) + ':' + str(dataX_save[i][0][1]), str(dataX_save[i][1][0]) + ':' + str(dataX_save[i][1][1]), str(dataX_save[i][labels[i][0]+1][0]) + ':' + str(dataX_save[i][labels[i][0]+1][1])])
	return (avg_accuracy/5)