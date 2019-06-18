import sys
import os
import math
import numpy as np
import scipy.io as spio
from copy import deepcopy
from collections import defaultdict
import matplotlib.pyplot as plt
import random

np.random.seed(0)
random.seed(0)



# x - (784,m)

# W1 - (30,784)

# b1 - (30,1) (Please read about implementing broadcasting in Numpy here )

# z1 - (30,m)

# h1 - (30,m)

# W2 - (10,30)

# b2 - (10,1) (Please read about implementing broadcasting in Numpy here)

# z2 - (10,m)

# y^ - (10,m)




def normalize_vals(X):

	a = -1

	X = X.astype(float)
	for column in X.T:
		for k, j in enumerate(column):
			column[k] = (a + (2.*float(((j - 0.0))/(255.0-0.0))))

	return X
	
def one_hot_encoded(Y):
	# Get the correct number of samples to determine dimensions
	num_samples = Y.shape[1]
	matrix = np.zeros(shape=(10, (num_samples)))

	# Generate new labels, placing a 1 where the class label defines the sample
	for i, val in enumerate(Y[0]):
		matrix[val][i] = 1

	return matrix

def initialize_weights(Xtrain_normalized):

	cols = Xtrain_normalized.shape[0]
	W1 = 0.0001*np.random.randn(30, cols)
	W2 = 0.0001*np.random.randn(10, 30)

	return W1, W2

def initialize_biases():
	b1 = np.zeros(shape=(30, 1))
	b2 = np.zeros(shape=(10, 1))

	return b1, b2

def soft_max(z_2, Xtrain_normalized):
	# so z_2 is 10 x 256 and x_train is 256 x 784
	# shape of passed in Xtrain_normalized is (784, 256)
	num_samples = Xtrain_normalized.shape[1]
	# input size should be (784, num_samples)
	y_hat = np.zeros(shape=(10, num_samples))

	denom = 0
	# calculate the maximum value across the rows and take the transpose to get for each sample
	max_val = np.max(z_2, axis=0)
	dim = max_val.shape
	max_val = max_val.reshape(dim[0], 1)
	numerator = np.exp(z_2 - max_val.T)

	# Sum over the values minus the max
	denom = np.sum(np.exp(z_2 - max_val.T), axis=0)

	denom = denom.reshape(denom.shape[0], 1)

	# return softmax activation function
	y_hat =  numerator/denom.T

	return y_hat

def forward_pass(W1, W2, b1, b2, num_samples, Xtrain_normalized):
	# (30, 784) X (784, 256)
	# shape of Xtrain_normalized passed in is (784, 256)
	# h1 = np.zeros(shape=(30, num_samples))

	# Matrix multiply weights and samples
	temp_z_1 = np.matmul(W1, Xtrain_normalized)

	z_1 = np.empty_like(temp_z_1)

	# Add offset to samples
	z_1 = temp_z_1 + b1

	h_1 = deepcopy(z_1)

	# Take the ReLu of z_1
	h_1[h_1 <= 0.0] = 0.0 

	# Matrix multiply the weights and h_1
	temp_z_2 = np.matmul(W2, h_1)

	# Add offset to samples
	z_2 = temp_z_2 + b2

	# Apply the soft_max function in order to normalize vector components
	y_hat = soft_max(z_2, Xtrain_normalized)


	return y_hat, h_1, z_1

def cross_entropy_loss(y_hat, one_hot_labels):

	sum_across_labels = 0

	# calculate cross entropy loss across y_hat (y_predicted) and one_hot_labels (y_true)
	loss = (float(np.sum(np.multiply(one_hot_labels, np.log(y_hat)))) / y_hat.shape[1]) * -1.0

	return loss



def backward_pass(y_hat, one_hot_labels, h_1, W1, W2, b1, b2, z_1, Xtrain_normalized, lr):

	# Go through and implement backward pass
	
	delta2 = np.divide((y_hat - one_hot_labels.T),y_hat.shape[1])

	loss_derivative_weight2 = np.matmul(delta2, h_1.T)
	loss_derivative_offset2 = delta2

	loss_derivative_offset2 = np.sum(loss_derivative_offset2, axis=1)
	loss_derivative_offset2 = loss_derivative_offset2.reshape(10, 1)

	derivative_ReLu = deepcopy(h_1) 

	# Take the derivative of h_1
	derivative_ReLu[derivative_ReLu > 0.0] = 1.0
	derivative_ReLu[derivative_ReLu <= 0.0] = 0.0

	delta1 = np.multiply(np.matmul(W2.T, delta2), derivative_ReLu)
	

	loss_derivative_weight1 = np.matmul(delta1, Xtrain_normalized.T)
	loss_derivative_offset1 = delta1

	loss_derivative_offset1 = np.sum(loss_derivative_offset1, axis=1)
	loss_derivative_offset1 = loss_derivative_offset1.reshape(30, 1)


	# update weights according to the learning rates
	W1 = W1 - (lr*loss_derivative_weight1)
	W2 = W2 - (lr*loss_derivative_weight2)
	
	# update biases with the new learning rate
	update = lr*loss_derivative_offset1
	b1 = b1 - update


	update = lr*loss_derivative_offset2
	b2 = b2 - update

	return W1, W2, b1, b2


def calc_accuracy(y_hat, one_hot_labels):

	max_predicted_index = 0
	max_true_index = 0
	total_correct = 0

	# Go through each of the classes and see if the max value in which each index is found is the same
	for i, item in enumerate(y_hat.T):
		max_predicted_index = np.argmax(item)
		max_true_index = np.argmax(one_hot_labels.T[i])
		# If the indices the same then the class labels for the prediction match the true labels
		if max_predicted_index == max_true_index:
			total_correct += 1
	return total_correct / y_hat.shape[1]


def early_stopping(temp_list2, temp_list_labels2, XVal_normalized, yVal, Xtrain_normalized, XTest_normalized, yTest, yTrain):

	training_loss = []
	training_accuracy = []
	val_accuracy = []
	num_samples = Xtrain_normalized.shape[1]

	learning_rate = 0.1
	prev_val_loss = 100
	curr_val_loss = 0
	repeated_decrease = 10
	epsilon = .0009 # between  and .0015
	optimal_epoch = 0
	val_loss = []
	val_loss.append(100)
	continuous_decrease = False
	print("training with optimal learning rate of ", learning_rate)
	W1, W2 = initialize_weights(Xtrain_normalized)
	b1, b2 = initialize_biases()
	for i in range(100):
		# train for one hundred epochs and calculate 
		print("current epoch is ", i)
		for j, item in enumerate(temp_list2):
			y_hat, h_1, z_1 = forward_pass(W1, W2, b1, b2, num_samples, item.T)
			W1, W2, b1, b2 = backward_pass(y_hat, temp_list_labels2[j], h_1, W1, W2, b1, b2, z_1, item.T, learning_rate)

		result = forward_pass(W1, W2, b1, b2, num_samples, deepcopy(Xtrain_normalized))
		result1 = forward_pass(W1, W2, b1, b2, num_samples, deepcopy(XVal_normalized))

		val_loss.append(cross_entropy_loss(result1[0], one_hot_encoded(yVal)))
		training_loss.append(cross_entropy_loss(result[0], one_hot_encoded(yTrain)))
		val_accuracy.append(calc_accuracy(result1[0], one_hot_encoded(yVal)))
		training_accuracy.append(calc_accuracy(result[0], one_hot_encoded(yTrain)))

		curr_val_loss = val_loss[-1]
		prev_val_loss = val_loss[-2]

		# check the values of the current and prev loss & if decreasing for at least 10 epochs return the optimal epoch
		# not learning any new information
		if(abs(curr_val_loss - prev_val_loss) < epsilon):
			if continuous_decrease == False:
				repeated_decrease = 10
				optimal_epoch = i
				optimal_W1, optimal_W2, optimal_b1, optimal_b2 = W1, W2, b1, b2
			

			if repeated_decrease == 0:
				break
			
			repeated_decrease -= 1
			continuous_decrease = True

		else:
			continuous_decrease = False
			repeated_decrease = 10


	

	epochs = []
	[epochs.append(i) for i in range(optimal_epoch)]

	# Graph validation & training loss of early stopping

	line1, = plt.plot(epochs, [training_loss[1:][i] for i in range(optimal_epoch)] , 'r', label='Train')
	line2, = plt.plot(epochs, [val_loss[1:][i] for i in range(optimal_epoch)], 'b', label='Validation')
	plt.legend(handles=[line1, line2], loc=1)
	plt.xlabel("Epoch")
	plt.ylabel("Cross Entropy Loss")
	plt.title("Learning Rate " + str(learning_rate))
	plt.show()


	# Graph validation and training accuracy of early stopping


	line1, = plt.plot(epochs, [training_accuracy[1:][i] for i in range(optimal_epoch)] , 'r', label='Train')
	line2, = plt.plot(epochs, [val_accuracy[1:][i] for i in range(optimal_epoch)], 'b', label='Validation')
	plt.legend(handles=[line1, line2], loc=1)
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.title("Learning Rate " + str(learning_rate))
	plt.show()



	print("the early stopping epoch is ", optimal_epoch)
	result = forward_pass(optimal_W1, optimal_W2, optimal_b1, optimal_b2, num_samples, deepcopy(XTest_normalized))
	test_loss = cross_entropy_loss(result[0], one_hot_encoded(yTest))
	accuracy = calc_accuracy(result[0], one_hot_encoded(yTest))
	print("training loss is ", training_loss[optimal_epoch])
	print("valdiation loss is ", val_loss[optimal_epoch])
	print("training accuracy is ", training_accuracy[optimal_epoch])
	print("validation accuracy is ", val_accuracy[optimal_epoch])
	print("test loss is ", test_loss)
	print("test accuracy is ", accuracy)

def train_and_val(temp_list1, temp_list_labels1, Xtrain_normalized, yTrain, XVal_normalized, yVal, one_hot_labels):
	training_loss = []
	training_accuracy = []
	validation_loss = []
	validation_accuracy = []
	training_all_losses = defaultdict(list)
	validation_all_losses = defaultdict(list)
	training_all_accuracies = defaultdict(list)
	validation_all_accuracies = defaultdict(list)
	num_samples = Xtrain_normalized.shape[1]

	lr = [0.001, 0.01, 0.1, 1.0, 10.0]
	for learning_rate in lr:
		print ("current learning rate is ", learning_rate)
		W1, W2 = initialize_weights(Xtrain_normalized)
		b1, b2 = initialize_biases()
		for i in range(100): 
			# train for 100 epochs and calculate forward and backward pass for mini batches
			print ("current epoch is ", i)
			for j, item in enumerate(temp_list1):

				y_hat, h_1, z_1 = forward_pass(W1, W2, b1, b2, num_samples, item.T)


				W1, W2, b1, b2 = backward_pass(y_hat, temp_list_labels1[j], h_1, W1, W2, b1, b2, z_1, item.T, learning_rate)
			
			# complete forward pass with updated weights for full train and validation data set
			result = forward_pass(W1, W2, b1, b2, num_samples, deepcopy(Xtrain_normalized))
			result1 = forward_pass(W1, W2, b1, b2, num_samples, deepcopy(XVal_normalized))

			training_loss.append(cross_entropy_loss(result[0], one_hot_labels))
			validation_loss.append(cross_entropy_loss(result1[0], one_hot_encoded(deepcopy(yVal))))
			training_accuracy.append(calc_accuracy(result[0], one_hot_labels))
			validation_accuracy.append(calc_accuracy(result1[0], one_hot_encoded(deepcopy(yVal))))

		training_all_losses[learning_rate] = training_loss
		validation_all_losses[learning_rate] = validation_loss
		training_all_accuracies[learning_rate] = training_accuracy
		validation_all_accuracies[learning_rate] = validation_accuracy


		training_loss = []
		validation_loss = []
		training_accuracy = []
		validation_accuracy = []

	epochs = 100
	plot_loss(lr, training_all_losses, validation_all_losses, epochs)

	plot_accuracy(lr, training_all_accuracies, validation_all_accuracies, epochs)


def plot_loss(lr, training_all_losses, validation_all_losses, epochs):
	epochs = []
	[epochs.append(i) for i in range(epochs)]

	for learn_rate in lr:
		line1, = plt.plot(epochs, training_all_losses[learn_rate], 'r', label='Train')
		line2, = plt.plot(epochs, validation_all_losses[learn_rate], 'b', label='Validation')
		plt.legend(handles=[line1, line2], loc=1)
		plt.xlabel("Epoch")
		plt.ylabel("Cross Entropy Loss")
		plt.title("Learning Rate " + str(learn_rate))
		plt.show()


def plot_accuracy(lr, training_all_accuracies, validation_all_accuracies, epochs):
	epochs = []
	[epochs.append(i) for i in range(epochs)]

	for learn_rate in lr:
		line1, = plt.plot(epochs, training_all_accuracies[learn_rate], 'r', label='Train')
		line2, = plt.plot(epochs, validation_all_accuracies[learn_rate], 'b', label='Validation')
		plt.legend(handles=[line1, line2], loc=1)
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy")
		plt.title("Learning Rate " + str(learn_rate))
		plt.show()


def main():

	data = spio.loadmat('mnistReduced.mat')

	# print(data)

	xTrain = data['images_train']  # 784*30000 (features x number of examples) ~ (d, n)
	yTrain = data['labels_train']  # 1*30000 (1, n)

	xVal = data['images_val']  # 784*3000
	yVal = data['labels_val'] # 1*3000

	xTest = data['images_test'] # 784*3000
	yTest = data['labels_test'] # 1*3000

	# proceed to normalize xTrain, xVal, xTest and then converting yTrain, yVal, yTest to one-hot encoding


	one_hot_labels = one_hot_encoded(yTrain)


	# -------- check to see if the class labels are balanced -------- #

	bin_count = np.bincount(yTrain[0])
	labels = np.nonzero(bin_count)[0]

	class_labels = zip(labels, bin_count[labels])
	class_labels = np.vstack(data).T

	# print(class_labels)

	# --------------------------------------------------------------- #


	Xtrain_normalized = deepcopy(xTrain)
	XVal_normalized = deepcopy(xVal)
	XTest_normalized = deepcopy(xTest)
	

	# normalize the training data between -1 and 1
	Xtrain_normalized = normalize_vals(Xtrain_normalized)
	XVal_normalized = normalize_vals(XVal_normalized)
	XTest_normalized = normalize_vals(XTest_normalized)

	W1, W2 = initialize_weights(Xtrain_normalized)

	b1, b2 = initialize_biases()

	# ---------------- Group Data into mini-batches of 256 ------------ #

	index = 0
	temp_list = []
	temp_list_labels = []
	
	for i, val in enumerate(Xtrain_normalized.T[::256]):
		temp_list.append((Xtrain_normalized.T[index:index + 256]))
		index = index + 256

	# separate the one_hot_encoded labels as well
	index = 0
	for i, val in enumerate(one_hot_labels.T[::256]):
		temp_list_labels.append(one_hot_labels.T[index:index + 256])
		index = index + 256


	# ----------------------------------------------------------------- #


	# ------------------------- Train and Val -------------------------------- #

	temp_list1 = deepcopy(temp_list)
	temp_list_labels1 = deepcopy(temp_list_labels)

	# train_and_val(temp_list1, temp_list_labels1, Xtrain_normalized, yTrain, XVal_normalized, yVal, one_hot_labels)
	# ------------------------------------------------------------------------ #


	# ----------------------- Implement Early Stopping and Loss/Accuracy on Test Data --------------------- #


	temp_list2 = deepcopy(temp_list)
	temp_list_labels2 = deepcopy(temp_list_labels)

	early_stopping(temp_list2, temp_list_labels2, XVal_normalized, yVal, Xtrain_normalized, XTest_normalized, yTest, yTrain)

	# ---------------------------------------------------------------------------------------------------- #

		



if __name__ == '__main__':
	main()