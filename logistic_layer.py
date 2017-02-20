import numpy as np

import theano
import theano.tensor as T

class logistic_layer(object):

	def __init__ (self, input, input_size, output_size):
		#print input_size
		#print output_size
		self.weights = theano.shared(np.ones(shape=(input_size,output_size), dtype = theano.config.floatX), name = 'log-weights', borrow= True)
		self.biases = theano.shared(np.zeros(shape =(output_size,), dtype = theano.config.floatX),name = 'log-biases', borrow = True)
		self.params = [self.weights,self.biases]
		#The probability of any output  given the input is computed by the softmax function. 
		#https://en.wikipedia.org/wiki/Logistic_regression#Model_fitting
		self.likelihood = T.nnet.softmax(T.dot(input,self.weights)+self.biases)
		#The prediction of the logistic layer is the output node with the highest probability. 
		self.prediction = T.argmax(self.likelihood,axis = 1)

	#Compute cost as the negative log likelihood
	def cost(self,y):
		#Return the negative average log of the likelihood of each point in the batch having the label given as a parameter.
		return -T.mean(T.log(self.likelihood)[T.arange(y.shape[0]),y])

	def accuracy(self, y):
		return T.mean(T.eq(self.prediction,y))
	def save_params(self):
		return [i.get_value() for i in self.params]

	def load_params(self,values):
		for param, value in zip(self.params, values):
			param.set_value(value)
