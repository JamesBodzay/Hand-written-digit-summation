import theano
import numpy as np
import numpy.random
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d

#A wrapper for the conv2d and maxpool functions from theano
#all parameters for this layer are the same as the inputs for conv2d. i.e.
#image_size = (batch_size, num_input_maps ,height, width)
#filter_size = (filter_size, num_input_maps. height,width)
#downscale is how much smaller the output is. I.e downscale = (2,2) means the height and width are halved 
#stride_size = how much do the pools overlap. default to no overlap.
#Padding add zeros around input image before downsampling.
class layer(object):


	
	#create a convolutional layer
	def __init__(self, input, image_size, filter_size, downscale,stride_size = None,  padding = (0,0)):
	#Weights are assigned randomly according to the standard normal distribution
	#We want a shallow copy of the weights so we can update them
		self.weights = theano.shared(
			np.asarray(
				np.random.standard_normal(filter_size), 
				dtype=theano.config.floatX
				),
			borrow = True,
			name = 'Weights-Convolution')
		self.input = input
		#Create biases , one bias per kernel
		self.biases = theano.shared(
			np.ones(
				filter_size[0],
				 dtype=theano.config.floatX
				 )
			)
		self.params = [self.weights,self.biases]

		conv_output = conv2d(
			input = input,
			filters = self.weights,
			input_shape =image_size,
			filter_shape = filter_size
			)


		#link the convolutional layer to the max pooling layer
		pooled = pool_2d(
			input = conv_output,
			ds = downscale,
			st = stride_size, 
			padding = padding,
			ignore_border = True
			)

		#We want the bias to be associated with each kernel and not be dependant on the input image or position in feature map. 
		self.output = T.tanh(pooled +self.biases.dimshuffle('x',0,'x','x'))
		#output result of max pooling layer

	def save_params(self):
		return [i.get_value() for i in self.params]

	def load_params(self,values):
		for param, value in zip(self.params, values):
			param.set_value(value)