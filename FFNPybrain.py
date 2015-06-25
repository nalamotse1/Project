from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
import time 
import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules import SoftmaxLayer
import matplotlib.pyplot as pl



class ProcessData():

	def __init__(self):
		pass

	def readData(self, filename):

		return np.genfromtxt('myfile.txt', delimiter = ' ', dtype = str)

	def displayTemp(self, array):

		#ready = open('data.txt', 'w')

		for j in range(len(array)):
			print array[j][1]
			#ready.write(array[j][1])
			#("\n")
		print array.shape

	def slidingWindow(self, data, start, end):

		window = data[start:end]
		if (end-start) == len(window):
			return data[start:end]
		else:
			return 

	''' activation functions'''
	def sigmoid(self,x):
		return 1.0/(1.0 + np.exp(-x))

	def sigmoid_prime(self,x):
		return (self.sigmoid(x)*(1.0) -self.sigmoid(x))

	def tanh(self,x):
		return np.tanh(x)
	
	def tanh_prime(x):
		return 1.0 - x**2

	def linear(self,x):
		return x

	def delta_linear(self,x):
		return numpy.ones(x.shape, dtype=numpy.float)
	 
	def sigmoid(self,x):
		return 1.0/(1.0+numpy.exp(-x))
		
		
	def softplus(self,x): 
		return np.log(1 + np.exp(x))

	def delta_softplus(self,x):
		return sigmoid(x) 

def main():
	p = ProcessData()

	filename = open('myfile.txt', 'r')
	start_time = time.time()
	data = p.readData(filename)
	k = 1
	windowsize = 4
	end = windowsize+1

	net = buildNetwork(4, 3, 1, bias=True, outclass=SoftmaxLayer)

	while end<=10:
		input4 = p.slidingWindow(data, k, end)
		#x = np.linspace(-7, 7, 5)
		#input = [3,5,6,9,1]
		given = np.array(input4)

		target = data[end][1]
		print 'target is'
		print target

		target11 =np.array([[data[end][1], data[end][1], data[end][1], data[end][1]]], dtype='|S8')

		achieve = target11.astype(np.float)
		size = len(given)
		ds = ClassificationDataSet(size, 1, nb_classes=3)
		print size, len(target)

		my_list = []
		for j in range(len(given)):
			my_list.append(float(given[j][1]))
			#ds.addSample(my_list[j],float(data[end][1]))
			ds.addSample(my_list[j],float(target))



		trndata, partdata = ds.splitWithProportion(0.60)
		tstdata, validata = partdata.splitWithProportion(0.50)

		print 'input'
		print ds['input']
		print 'target set'
		print ds['target']

		print ' trndata'
		print len(trndata)
		print ' partdata'
		print len(partdata)
		print 'tstdata'
		print len(tstdata)
		print 'validata'
		print len(validata)


		 
		input = np.array(my_list)
		print 'input is'
		print input
		print 'achieve is'
		print achieve

		

		inp = input.reshape(size,1)
		tar = achieve.reshape(size,1)

		#print net.activate(input)
		#ds = SupervisedDataSet(4, 1)
		print trndata.indim, trndata.outdim, tstdata.indim, tstdata.outdim
		

		
			

		trainer = BackpropTrainer(net, dataset = trndata, verbose=True, learningrate = 10, momentum = 0.99)
		trnerr, valerr = trainer.trainUntilConvergence(dataset=trndata, maxEpochs=50)
		
		pl.plot(trnerr, 'b', valerr, 'r')

		k += 1
		end += 1

	
main()