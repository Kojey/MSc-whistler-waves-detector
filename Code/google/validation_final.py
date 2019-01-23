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
	processed_features['rooms_per_person'] = (california_housing_dataframe['total_rooms']/california_housing_dataframe['population'])
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
	output_targets['median_house_value'] = (california_housing_dataframe['median_house_value']/1000.0)
	return output_targets


training_examples = preprocess_features(california_housing_dataframe.head(12000))
# print(training_examples.describe())
training_targets = preprocess_targets(california_housing_dataframe.head(12000))
# print(training_targets.describe())
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
# print(validation_examples.describe())
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
# print(validation_targets.describe())

print(california_housing_dataframe.describe())

plt.figure(figsize=(13,8))

ax = plt.subplot(1,4,1)
ax.set_title('Validation Data')

ax.set_autoscaley_on(False)
ax.set_ylim([32,43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126,-112])

plt.scatter(validation_examples['longitude'],
			validation_examples['latitude'],
			cmap='coolwarm',
			c=validation_targets['median_house_value']/validation_targets['median_house_value'].max())


ax = plt.subplot(1,4,2)
ax.set_title('Training Data')

ax.set_autoscaley_on(False)
ax.set_ylim([32,43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126,-112])

plt.scatter(training_examples['longitude'],
			training_examples['latitude'],
			cmap='coolwarm',
			c=training_targets['median_house_value']/training_targets['median_house_value'].max())

ax = plt.subplot(1,4,3)
ax.set_title('Data')

ax.set_autoscaley_on(False)
ax.set_ylim([32,43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126,-112])

plt.scatter(california_housing_dataframe['longitude'],
			california_housing_dataframe['latitude'],
			cmap='coolwarm',
			c=california_housing_dataframe['median_house_value']/california_housing_dataframe['median_house_value'].max())


_ = plt.plot()

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
	""" Trains a linear regression model of one feature

		Args:
			features: pandas DataFrame of features
			targets: pandas DataFrame of targets
			batch_size: Size of batches to be passed to the model
			suffle: True or False. Wheter to shuffle the data
			num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
		Return:
			Tuple of (features, label) for next data batch
	"""

	# convert pandas data into a dict of np array
	features = {key:np.array(value) for key,value in dict(features).items()}

	# construct a dataset and configure batching/repeating
	ds = Dataset.from_tensor_slices((features,targets)) # warnings: 2GB max
	ds = ds.batch(batch_size).repeat(num_epochs)

	# shuffle the data if specified
	if shuffle:
		ds = ds.shuffle(buffer_size=10000)

	# return the next batch of data
	features, labels = ds.make_one_shot_iterator().get_next()
	return features, labels

def construct_feature_columns(input_features):
	"""Constructy the TensorFlow Feature Columns.
	Args:
		input_features: The names of the numerical input features to use.
	Returns:
		A set of feature columns
	"""
	return set(tf.feature_column.numeric_column(my_feature) for my_feature in input_features)

def train_model(
	learning_rate,
	steps,
	batch_size,
	training_examples,
	training_targets,
	validation_examples,
	validation_targets):
	"""Trains a linear regression model of multiple features.

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
	linear_regressor = tf.estimator.LinearRegressor(
		feature_columns=construct_feature_columns(training_examples),
		optimizer=my_optimizer
	)

	# 1. Create input functions
	training_input_fn = lambda:my_input_fn(
		training_examples, 
		training_targets['median_house_value'],
		batch_size=batch_size)
	predict_training_input_fn = lambda:my_input_fn(
		training_examples, 
		training_targets['median_house_value'],	
		num_epochs=1, 
		shuffle=False)
	predict_validation_input_fn = lambda:my_input_fn(
		validation_examples, 
		validation_targets['median_house_value'], 
		num_epochs=1, 
		shuffle=False)

	# Train the model, but do so inside a loop so that we can periodically assess loss metrics.
	print('Training model...')
	print('RMSE (on training data):')
	training_rmse = []
	validation_rsme = []
	for period in range(0,periods):
		# Train the model, starting from the prior state.
		linear_regressor.train(
			input_fn=training_input_fn,
			steps=steps_per_period
		)
		# 2. Take a braek and compute predictions
		training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
		training_predictions = np.array([item['predictions'][0] for item in training_predictions])
		validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
		validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

		# Compute training and validation loss
		training_root_mean_squared_error = math.sqrt(
			metrics.mean_squared_error(training_predictions, training_targets))
		validation_root_mean_squared_error = math.sqrt(
			metrics.mean_squared_error(validation_predictions, validation_targets))

		# Occasionally print the current loss
		print('	period %02d : %0.2f'% (period, training_root_mean_squared_error))
		# Add the loss metrics from this period to our list.
		training_rmse.append(training_root_mean_squared_error)
		validation_rsme.append(validation_root_mean_squared_error)
	print('Model training finished.')

	# Output a graph of loss metrics over periods
	plt.subplot(1,4,4)
	plt.ylabel('RMSE')
	plt.xlabel('Periods')
	plt.title('Root Mean Squared Error vs Periods')
	plt.tight_layout()
	plt.plot(training_rmse, label='training')
	plt.plot(validation_rsme, label='Validation')
	plt.legend()

	return linear_regressor

linear_regressor = train_model(
	learning_rate=0.01,
	steps=100,
	batch_size=250,
	training_examples=training_examples,
	training_targets=training_targets,
	validation_examples=validation_examples,
	validation_targets=validation_targets)

california_housing_test_data = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv", sep=",")

test_examples = preprocess_features(california_housing_test_data)
test_targets = preprocess_targets(california_housing_test_data)

predict_test_input_fn = lambda:my_input_fn(
	test_examples,
	test_targets['median_house_value'],
	num_epochs=1,
	shuffle=False)

test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
test_predictions = np.array([item['predictions'][0] for item in test_predictions])

root_mean_squared_error = math.sqrt(
	metrics.mean_squared_error(test_predictions, test_targets))

print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)
plt.show()