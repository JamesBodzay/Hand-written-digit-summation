import cPickle as pickle
import theano
import numpy as np
import numpy.random
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d
import math
import layer
import connected_layer
import logistic_layer
import matplotlib.pyplot as plt
import csv
import image_proc as proc

def load_images(filename):
	return pickle.load(file(filename,'r'))
#Returns a validation set equal to 1/folds the size of the total set
def create_validation_sets(training_set,training_labels,  folds):
	labeled_set = zip(training_set,training_labels)
	num_samples = len(training_set)
	samples_per_set = int(np.ceil(num_samples / float(folds)))
	np.random.shuffle(labeled_set)

	#sets = [labeled_set[a:b] for a,b in zip(range(0,num_samples,samples_per_set),range(samples_per_set,num_samples,samples_per_set)+[num_samples])]
	sets = [labeled_set[0:samples_per_set], labeled_set[samples_per_set:]]
	return sets


def cast_data(data,asint):
	shared = theano.shared(numpy.asarray(data,dtype = theano.config.floatX),borrow=True)
	if asint:
		shared = shared.flatten()
		shared = T.cast(shared,'int32')

	return shared


def test_network(features,labels,mb_size,learn_rate, max_iterations, test_features = None):
	#Create the features and all
	sets = create_validation_sets(features,labels,10)


	training_features = [i[0] for i in sets[1]]
	training_labels = np.asarray([i[1] for i in sets[1]])
	training_labels.resize(len(training_labels),)
	validation_features = [i[0] for i in sets[0]]
	validation_labels = np.asarray([i[1] for i in sets[1]])
	validation_labels.resize(len(validation_labels),)


	#print validation_labels
	training_set_size = len(training_features)
	validation_set_size = len(validation_features)
	test_set_size = len(test_features)
	#Cast to shared variables
	training_features = cast_data(training_features,False)
	training_labels = cast_data(training_labels,True)
	validation_features = cast_data(validation_features,False)
	validation_labels = cast_data(validation_labels,True)
	test_features = cast_data(test_features,False)

	# print training_labels.type
	# print training_features[0].type
	# print validation_labels.ndim
	#print training_features[0*int(mb_size):(0+1)*int(mb_size)].shape

	x = T.tensor3('x')
	y =T.ivector('y')
	input_layer = x.reshape((mb_size,1,60,60))
	
	#First layer reduces from 60*60 to 4 channells of 56/2 * 56/2 = 28*28/2.
	# 4 5*5 kernels. With a downscale of half.
	num_kernels1 = 256
	layer1 = layer.layer(input = input_layer, image_size = (mb_size,1,60,60), filter_size = (num_kernels1,1,5,5), downscale = (2,2))
	#layer1.load_params(pickle.load(file('layer_1_small.pkl','r')))
	
	num_kernels2 = 128
	layer2 = layer.layer(input = layer1.output,image_size=(mb_size,num_kernels1,28,28), filter_size = (num_kernels2,num_kernels1,5,5), downscale = (2,2))
	#layer2.load_params(pickle.load(file('layer_2_small.pkl','r')))

	# num_kernels3 = 12
	# layer3 = layer.layer(input = layer2.output,image_size=(mb_size,num_kernels2,12,12), filter_size = (num_kernels3,num_kernels2,5,5), downscale = (2,2))
	# layer3.load_params(pickle.load(file('layer_3_small.pkl','r')))
	flattened = layer2.output.flatten(2)
	#depending on time constrain, might want to increase downsampling or add an extra layer if have more than enough time.
	#num inputs = 
	layer4 = connected_layer.connected_layer(input = flattened, input_size = (num_kernels2 * 12 * 12),output_size = 100)
	#layer4.load_params(pickle.load(file('layer_4_small.pkl','r')))
	layer5 = logistic_layer.logistic_layer(input=layer4.output, input_size = 100, output_size = 19)
	#layer5.load_params(pickle.load(file('layer_5_small.pkl','r')))
	cost = layer5.cost(y)
	#Params to be updated:
	full_params = layer5.params + layer4.params + layer2.params + layer1.params

	#Create a function that cooresponds to the gradients for each parameter
	gradients = T.grad(cost,full_params)

	#From these gradients we get different updates for each parameter
	update_list = [(p,p - learn_rate * d) for p,d in zip(full_params,gradients) ]

	batch_number = T.scalar('bn', dtype='int32')

	#This function updates based on a batch of features and labels. 
	training_function = theano.function(
		[batch_number], 
		cost,
		updates=update_list,
		givens={ x: training_features[batch_number*int(mb_size):(batch_number+1)*int(mb_size)],
				 y: training_labels[batch_number*int(mb_size):(batch_number+1)*int(mb_size)]}
		)
	training_accuracy_function = theano.function(
		[batch_number],
		layer5.accuracy(y),
		givens = {x: training_features[batch_number*int(mb_size):(batch_number+1)*int(mb_size)],
				 y: training_labels[batch_number*int(mb_size):(batch_number+1)*int(mb_size)]}
		)
	validation_function = theano.function(
		[batch_number],
		layer5.accuracy(y),
		givens = {x: validation_features[batch_number*int(mb_size):(batch_number+1)*int(mb_size)],
				  y: validation_labels[batch_number*int(mb_size):(batch_number+1)*int(mb_size)]}
		)


	test_function = theano.function(
		[batch_number],
		layer5.prediction,
		givens = {x: test_features[batch_number*int(mb_size):(batch_number+1)*int(mb_size)]}
		)
	#Keep track of the training err

	training_acc = []
	
	#Keep track of the validation err

	validation_acc = []

	#PERFORM THE LEARNING OF THE MODEL
	num_batches = training_set_size / mb_size
	num_validate_batches = validation_set_size / mb_size
	num_test_batches = test_set_size/mb_size
	for iter in range(0, max_iterations):
		#Perform the batch updates
		for batch in range(0, num_batches):
			#print "LEN" + str((batch+1)* mb_size)
			training_function(batch)

		#print "Trained"
		if iter%1 == 0:
			#for batch in range(0, num_batches):
			t_acc = np.mean([training_accuracy_function(batch) for batch in range(0,num_batches)])
			v_acc = np.mean([validation_function(batch) for batch in range(0,num_validate_batches)])
			
			training_acc.append(t_acc)
			validation_acc.append(v_acc)
			print "ITERATION "+ str(iter) + " TRAIN ACC " + str(t_acc) + " VAL ACC " + str(v_acc)

			final_predictions = []
			for batch in range(0,num_test_batches):
				final_predictions.append( test_function(batch))

			pickle.dump(final_predictions,file('predictions.pkl','w'))
			pickle.dump(training_acc,file('training.pkl','w'))
			pickle.dump(validation_acc,file('validation.pkl','w'))

		if v_acc > 0.9 :
			print "COMPLETED AFTER " + str(iter) + "ITERATIONS "
			break


	
	t_acc = np.mean([training_accuracy_function(batch) for batch in range(0,num_batches)])
	v_acc = np.mean([validation_function(batch) for batch in range(0,num_validate_batches)])

	training_acc.append(t_acc)
	validation_acc.append(v_acc)
	print " TRAIN ACC " + str(t_acc) + " VAL ACC " + str(v_acc)
	
	final_predictions = []
	for batch in range(0,num_test_batches):
		final_predictions.append( test_function(batch))

	#print final_predictions
	pickle.dump(layer1.save_params(),file('layer_1.pkl','w'))
	pickle.dump(layer2.save_params(),file('layer_2.pkl','w'))
	#pickle.dump(layer3.save_params(),file('layer_3.pkl','w'))
	pickle.dump(layer4.save_params(),file('layer_4.pkl','w'))
	pickle.dump(layer5.save_params(),file('layer_5.pkl','w'))
	pickle.dump(training_acc,file('training.pkl','w'))
	pickle.dump(validation_acc,file('validation.pkl','w'))
	pickle.dump(final_predictions,file('predictions.pkl','w'))

	return final_predictions

def print_results(predictions,outfile):
	l = len(predictions)
	outfile = open(outfile,'w')
	outfile.write('Id,Prediction\n')
	for i in range(0,l):
		for j in range(0,len(predictions[i])):
			outfile.write('{0},{1}\n'.format((i*len(predictions[i]))+j,predictions[i][j]))

def load_results(filename):
	y_in = [];
	i=0;
	with open(filename, 'rb') as csvfile:#train_out010000
	    reader = csv.reader(csvfile, delimiter=',')
	    for col in reader:
	    	if (i!=0):
	    		y_in.append(int(col[1]));
	    	i+=1;
	return y_in

if __name__=="__main__":
	#features = load_images("processed.pkl")
	imgs = proc.load_images('../train_x.bin',100000)
	subset = imgs[:]
	for i in range(0,len(subset)):
		subset[i] = proc.process_image(subset[i])
	features = subset
	
	test = proc.load_images('../test_x.bin',20000)
	sub_test = test[:]
	for i in range(0,len(sub_test)):
		sub_test[i] = proc.process_image(sub_test[i])
	test_features = sub_test

	labels = load_results("../train_y.csv")
	labels = labels[:]
	print "BUILDING MODEL"
	predictions = test_network(features,labels,10000, 0.05, 10,test_features)
	print_results(predictions, 'test_y.csv')
