import numpy as np
import time
from mnist import MNIST
import json
np.set_printoptions(suppress=True)

class Restricted_Boltzman():
	"""
	Python implementation of a Restricted Boltzman Machine
	"""
	def __init__(self, learning_rate = 0.1, input_layer_size=5, hidden_layer_size=4):
		np.random.seed(1)
		self.learning_rate = learning_rate
		self.input_layer_size = input_layer_size
		self.hidden_layer_size = hidden_layer_size
		# create a numpy array with each row corresponding to an input neuron and each column corresponding to a hidden neuron
		# initialize weights with random numbers
		self.weights = np.array([np.random.random() for i in range(self.input_layer_size*self.hidden_layer_size)]).reshape(self.input_layer_size, self.hidden_layer_size)
		
	def activation_function(self, x):
		"""
		The activation function of this network
		In this case a sigmoid function
		"""
		return 1/(1+np.exp(-1 * x))
		
	def train(self, inputs, max_count = 1):
		"""
		Train the network using inputs and repeating max count times
		params:
			inputs: numpy array of input data organized in rows
			max_count: amount of times the training process should be repeated

		"""
		# print the training parameters
		print(f"input neurons: {self.input_layer_size}\nhidden neurons: {self.hidden_layer_size}\nlearning rate: {self.learning_rate}\ninput size: {len(inputs)}\nrepetitions: {max_count}")
		# initialize counter 
		ct = 0	
		# repeat the training process max_count times
		for count in range(max_count):
			# repeat for each row in the dataset
			for input_group in inputs:
				# forward activation
				# calculate the input for each neuron in the hidden layer
				# each row in weights corresponds to one input neuron and each column to one hidden neuron
				# so multiplying the input group with each column multiplies the input from each input neuron with the 
				# weight corresponding to this input neuron and the hidden neuron the current column corresponds to
				hidden_layer_in = np.dot(input_group, self.weights)
				# calculate the output of the hidden layer by apllying the activation function to the input of each neuron
				hidden_layer_out = self.activation_function(hidden_layer_in)
				
				#reconstruction
				# reconstruct the input using the same principle as above but with the roles of hidden and input neurons reversed
				output_layer_in = np.dot(self.weights, hidden_layer_out.T)
				output_layer_out = self.activation_function(output_layer_in.T)

				# contrastive divergence
				# reshape the output of the hidden layer for later use
				hl = hidden_layer_out.reshape(1, self.hidden_layer_size)
				# calculate the difference between the input and the reconstructed input
				diff = (input_group - output_layer_out).reshape(self.input_layer_size, 1)
				# multiplying the difference vector with the hidden layer output vector results in a matrix of the same shape as weights
				# with each cell (i, j) containing the difference between input and reconstructed input for the input neuron i
				# multiplied with the output of the j'th hidden neuron multiplied by the learning rate 
				# η(hj * vi0 − hj * vi1) with η = learning rate, vi0 = input from the ith input neuron, vi1 = reconstructed input from the ith input neuron
				# and hj = the output of the jth hidden neuron
				error = self.learning_rate * (diff @ hl) 
				
				#adjust the weights using the values calculated above
				# Wji^(t+1) = Wji^(t) + η(hj * vi0 − hj * vi1)
				self.weights = self.weights + error
					
				# increment the counter 
				ct+=1
				# print the percentage of the training process that has already been completed
				print("%.02f%%" % (ct/(len(inputs) * max_count)*100))			
		


	def identify(self, input_group, output=False):
		"""
		Identification method meant for use with the mnist letters dataset 
		calculates the output of the network for a given input and returns 
		the index of the largest element in output[:784] which in the case
		of mnist letters with 10 extra neurons for the label would be the
		identified number
		
		params:
			input_group: the input to be identified
			output: boolean defining wether the reconstructed input and some other information is printed
		"""
		if len(input_group) < self.input_layer_size:
			input_group = np.append(input_group, np.zeros(self.input_layer_size - len(input_group)))
		# forward activation
		hidden_layer_in = np.dot(input_group, self.weights)
		hidden_layer_out = self.activation_function(hidden_layer_in)
		# input reconstruction
		output_layer_in = np.dot(self.weights, hidden_layer_out.T)
		output_layer_out = self.activation_function(output_layer_in.T)
		# get the label part
		labels = output_layer_out[784:]
		# if the output flag is set print the reconstructed input the output value for the identified label
		# the identified label and the output values for all labels
		if output:
			print(MNIST.display(output_layer_out[:784] * 255))
			print(labels.max())
			print(labels.argmax())
			print(labels)
		# return the identified label
		return labels.argmax()


	def test(self):
		"""
		Test method designed for use with the mnist letters dataset
		loads the testing data for this dataset, constructs inputs without the label,
		has the network identify the inputs and compares these to the actual inputs
		"""
		# load the testing data
		images, labels = mndata.load_testing()
		# construct inputs with all label neurons deactivated
		test_amount = len(images)
		#inputs = np.hstack((np.array(images[:test_amount]), np.zeros((test_amount,10))))
		inputs = np.array(images[:test_amount])
		# have the network identify all inputs
		test_data = [x.identify(inputs[i]) for i in range(test_amount)]
		# compute and return the percentage of correctly identified inputs
		return f"percentage {1-np.mean(np.array(labels[:test_amount]) != np.array(test_data))}"

	def load_and_train(self, training_amount=10000, max_count=1):
		mndata = MNIST(r"samples")
		images, labels = mndata.load_training()

		# reduce the amount of images and labels in the training set to the amount specified above
		images = images[:training_amount]
		labels = labels[:training_amount]

		#convert images to a numpy array and replacing all non-zero values with 1 
		images = [[1 if el != 0 else 0 for el in image] for image in images]

		# each label is a number from 0 to 9 to represent that in the inputs for the network each number i is converted to 10 numbers with all except the ith being 0
		label_input = np.array([[0 if i != label else 1 for i in range(10)] for label in labels]).reshape(training_amount, 10)	

		# construct inputs by appending the label inputs to the images
		input_group = np.hstack((np.array(images), label_input))

		self.train(input_group, max_count=max_count)

if __name__ == "__main__":
	input_layer_size = 794
	hidden_layer_size = 794
	training_amount = 10000
	
	mndata = MNIST(r"samples")

	
	images, labels = mndata.load_training()
	
	# reduce the amount of images and labels in the training set to the amount specified above
	images = images[:training_amount]
	labels = labels[:training_amount]

	#convert images to a numpy array and replacing all non-zero values with 1 
	images = [[1 if el != 0 else 0 for el in image] for image in images]

	# each label is a number from 0 to 9 to represent that in the inputs for the network each number i is converted to 10 numbers with all except the ith being 0
	label_input = np.array([[0 if i != label else 1 for i in range(10)] for label in labels]).reshape(training_amount, 10)	
	
	
	x = Restricted_Boltzman(input_layer_size=input_layer_size, hidden_layer_size=hidden_layer_size, learning_rate=0.1)
	# construct inputs by appending the label inputs to the images
	input_group = np.hstack((np.array(images), label_input))
	
	# train and test
	print(x.train(input_group, max_count=1))
	print(x.test())