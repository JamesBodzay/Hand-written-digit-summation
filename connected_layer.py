import numpy as np

import theano
import theano.tensor as T
class connected_layer(object):
	
	def __init__(self, input,input_size, output_size):

		self.weights = theano.shared(
			np.asarray(
				np.random.standard_normal(size = (input_size,output_size)), dtype=theano.config.floatX),
			borrow = True,
			name = 'Weights')

		self.biases = theano.shared(
			 np.zeros((output_size,), dtype=theano.config.floatX),
			 borrow = True,
			 name = 'biases')

		self.params = [self.weights,self.biases]
		self.output = T.tanh(T.dot(input,self.weights)+self.biases)


	def cost(self, labels):
		return T.sum(self.output)
	def save_params(self):
		return [i.get_value() for i in self.params]

	def load_params(self,values):
		for param, value in zip(self.params, values):
			param.set_value(value)


