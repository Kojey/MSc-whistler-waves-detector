from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
	np.random.permutation(california_housing_dataframe.index))

def preprocess_features(california_housing_dataframe):
	""" Prepares input features frim California housing data set.

	Args:
			california_housing_dataframe: A Pandas DataFrame expected to contain data from the California housing data set
	Returns:
			A dataframe that contains the features to be used for the model, including synthetic features.
	"""
	selected_features = california_housing_dataframe[
		['latitude',
		'longitude',
		'housing_median_age',
		'total_rooms',
		'total_bedrooms',
		'population',
		'households',
		'median_income']]
	processed_features = selected_features.copy()

	# create a synthetic feature
	processed_features['rooms_per_person'] = (
		california_housing_dataframe['total_rooms']/
		california_housing_dataframe['population'])
	return processed_features

def preprocess_targets(california_housing_dataframe):
	""" Prepares target features (i.e., labels) from Clifornia housing data set.
	Args:
			california_housing_dataframe: A Pandas DataFrame expected to contain data from the California housing data set
	Returns:
			A dataframe that contains the target features.
	"""		
	output_targets = pd.DataFrame()
	# scale the target to be in units of thousands of dollards.
	output_targets['median_house_value_is_high'] = (
		california_housing_dataframe['median_house_value']>265000).astype(float)
	return output_targets

def construct_feature_columns(input_features):
	"""Construct the TensorFlow Feature Columns.
	Returns: 
		A set of feature columns
	"""
	return set([tf.feature_column.numeric_column(my_feature)
		for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
	"""Trains a linear regression model.

	Args:
		features: pandas DataFrame of features
		targets: pandas DataFrame of targets
		batch_size: Size of hte batches to be passed to the model
		shuffle: True or False. Whether to shuffle the data.
		num_epochs: Number of epochs for which data should be repeated. None = repeat indefiniely.
	Returns:
		Tuple of (features, labels) for next data batch
	"""

	# Convert pandas data into a dict of np arrays.
	features = {key:np.array(value) for key,value in dict(features).items()}

	# Construct a dataset, and configure batching/repeating
	ds = Dataset.from_tensor_slices((features, targets)) # warning 2GB limit
	ds = ds.batch(batch_size).repeat(num_epochs)

	# Shuffle the data, if specified
	if shuffle:
		ds = ds.shuffle(10000)
	# Return the next batch of data
	features, labels = ds.make_one_shot_iterator().get_next()
	return features, labels

training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
# display.display(training_examples.describe())


def train_linear_classifier_model(
	learning_rate,
	steps,
	batch_size,
	training_examples,
	training_targets,
	validation_examples,
	validation_targets):
	"""Trains a linear classifier model of multiple features.

	In addition to training, this function also prints training progress information,
	as well as a plot of the trainig and validation loss over time.

	Args:
		learning_rate: A 'float', the learning_rate.
		steps: 	A non-zero 'int', the total number of training steps. 
				A traning step consists of a forward and backward pass using a single batch.
		batch_size: A non-zero 'int', the batch size.
		training_examples: A 'DataFrame' containing one or more columns from 
							'california_housing_dataframe' to use as input features for training.
		training_targets: A 'DataFrame' containing the excatly one column from 
							'california_housing_dataframe' to use as target for training.
		validation_examples: A 'DataFrame' containing one or more columns from 
							'california_housing_dataframe' to use as input features for validation.
		validation_targets: A 'DataFrame' containing the excatly one column from 
							'california_housing_dataframe' to use as target for validation.			
	Returns:
		A 'LinearRegressor' object trained on the training data.
	"""

	periods = 10
	steps_per_period = steps/periods

	# Create a linear regressor object.
	my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
	linear_classifier = tf.estimator.LinearClassifier(
		feature_columns=construct_feature_columns(training_examples),
		optimizer=my_optimizer
	)

	# Create input functions.
	training_input_fn = lambda:my_input_fn(training_examples,
											training_targets['median_house_value_is_high'],
											batch_size=batch_size)
	predict_training_input_fn = lambda:my_input_fn(training_examples,
													training_targets['median_house_value_is_high'],
													num_epochs=1,
													shuffle=False)
	predict_validation_input_fn = lambda:my_input_fn(validation_examples,
													validation_targets['median_house_value_is_high'],
													num_epochs=1,
													shuffle=False)
	# Train the model, but do so inside a loop so that we can periodically assess
	# loss metrics.
	print('Training model...')
	print('LogLoss (on training data):')
	training_log_losses = []
	validation_log_losses = []
	for period in range(0, periods):
		# Train the model, starting from the prior state.
		linear_classifier.train(
			input_fn=training_input_fn,
			steps=steps_per_period
		)
		# Take a break and compute probabilities.
		training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
		training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

		validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
		validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

		# Compute training and validation loss
		training_log_loss = metrics.log_loss(training_targets, training_probabilities)
		validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)

		# Occasionally print the current loss.
		print(' period %02d : %0.2f' % (period, training_log_loss))
		# Add the loss metrics from this period to our list.
		training_log_losses.append(training_log_loss)
		validation_log_losses.append(validation_log_loss)
	print('Model training finished.')

	# Output a graph of loss metrics over periods
	plt.ylabel('LogLoss')
	plt.xlabel('Periods')
	plt.title('LogLoss vs Periods')
	plt.tight_layout()
	plt.plot(training_log_losses, label='training')
	plt.plot(validation_log_losses, label='Validation')
	plt.legend()

	evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)
	print('AUC on the validation set: %0.2f' % evaluation_metrics['auc'])
	print('Accuracy on the validation set: %0.2f' % evaluation_metrics['accuracy'])

	validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
	validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])

	false_positive_rate, true_positive_rate, threholds = metrics.roc_curve(
		validation_targets, validation_probabilities)
	plt.figure()
	plt.plot(false_positive_rate, true_positive_rate, label='our model')
	plt.plot([0,1],[0,1], label='random classifier')
	plt.legend(loc=2)


	return linear_classifier

linear_classifier = train_linear_classifier_model(
	learning_rate=0.000005,
	steps=25000,
	batch_size=500,
	training_examples=training_examples,
	training_targets=training_targets,
	validation_examples=validation_examples,
	validation_targets=validation_targets)



plt.show()
