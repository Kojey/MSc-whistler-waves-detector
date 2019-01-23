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
california_housing_dataframe['median_house_value'] /= 1000.0


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


def train_model(learning_rate, steps, batch_size, input_feature, input_target='median_house_value'):
	"""Trains a linear regression model of one feature.
	Args:
	learning_rate: A `float`, the learning rate.
	steps: A non-zero `int`, the total number of training steps. A training step
	consists of a forward and backward pass using a single batch.
	batch_size: A non-zero `int`, the batch size.
	input_feature: A `string` specifying a column from `california_housing_dataframe`
	to use as input feature.
	"""
	periods = 10
	steps_per_period = steps/periods

	my_feature = input_feature
	my_label = input_target
	
	my_feature_data = california_housing_dataframe[[my_feature]].astype('float32')
	targets = california_housing_dataframe[my_label].astype('float32')

	# create feature columns
	feature_columns = [tf.feature_column.numeric_column(my_feature)]

	# create input functions
	training_input_fn = lambda:my_input_fn(my_feature_data, targets, batch_size=batch_size)
	prediction_input_fn = lambda:my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

	# create a linear regressor object
	my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
	linear_regressor = tf.estimator.LinearRegressor(
		feature_columns=feature_columns,
		optimizer=my_optimizer
		)

	# set up to the plot the state of our model's line each period
	plt.figure(figsize=(15,6))
	plt.subplot(1,2,1)
	plt.title('Learned Line by Period')
	plt.ylabel(my_label)
	plt.xlabel(my_feature)
	sample = california_housing_dataframe.sample(n=300)
	plt.scatter(sample[my_feature], sample[my_label])
	colors = [cm.coolwarm(x) for x in np.linspace(-1,1,periods)]

	# train the model, but do so inside a loop so that we can periodically assess loss metrics
	print('Training model...')
	print('RMSE (on training data):')
	root_mean_squared_errors = []
	for period in range(0, periods):
		# train the model, starting from the prior state
		linear_regressor.train(
			input_fn=training_input_fn,
			steps=steps_per_period
		)
		# take a break and compute predictions
		predictions = linear_regressor.predict(input_fn=prediction_input_fn)
		predictions = np.array([item['predictions'][0] for item in predictions])

		# compute the loss
		root_mean_squared_error = math.sqrt(metrics.mean_squared_error(predictions,targets))
		# occasionally print the current loss
		print('	period %02d : %0.2f' % (period, root_mean_squared_error))
		# add the loss metrics from this period to our list
		root_mean_squared_errors.append(root_mean_squared_error)
		# finally, track the weighrs and biases over time
		# apply some math to ensure that the data and line are plotted neatly
		y_extents = np.array([0, sample[my_label].max()])

		weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)
		bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

		x_extents = (y_extents - bias)/weight
		x_extents = np.maximum(np.minimum(x_extents, 
											sample[my_feature].max()),
											sample[my_feature].min())
		y_extents = weight * x_extents + bias
		plt.plot(x_extents, y_extents, color=colors[period])
		
	print('Model training finished.')

	# output a graph of loss metrics over periods
	plt.subplot(1,2,2)
	plt.ylabel('RMSE')
	plt.xlabel('Periods')
	plt.title("Root Mean Squared Error vs. Periods")
	plt.tight_layout()
	plt.plot(root_mean_squared_errors)

	  # Output a table with calibration data.
	calibration_data = pd.DataFrame()
	calibration_data["predictions"] = pd.Series(predictions)
	calibration_data["targets"] = pd.Series(targets)
	display.display(calibration_data.describe())

	print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)
	# display graph
	plt.show()
	return calibration_data

california_housing_dataframe["rooms_per_person"] = california_housing_dataframe["total_rooms"]/california_housing_dataframe["population"]


# clip outliers
california_housing_dataframe['rooms_per_person'].hist()
california_housing_dataframe['rooms_per_person'] = (california_housing_dataframe['rooms_per_person']).apply(lambda x: min(x,5))
california_housing_dataframe['rooms_per_person'].hist()

calibration_data = train_model(
    learning_rate = 0.02,
    steps = 750,
    batch_size = 400,
    input_feature='rooms_per_person'
)
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.scatter(calibration_data["predictions"], calibration_data["targets"])

calibration_data = train_model(
    learning_rate=0.02,
    steps=750,
    batch_size=400,
    input_feature="rooms_per_person"
)