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

ax = plt.subplot(1,3,1)
ax.set_title('Validation Data')

ax.set_autoscaley_on(False)
ax.set_ylim([32,43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126,-112])

plt.scatter(validation_examples['longitude'],
			validation_examples['latitude'],
			cmap='coolwarm',
			c=validation_targets['median_house_value']/validation_targets['median_house_value'].max())


ax = plt.subplot(1,3,2)
ax.set_title('Training Data')

ax.set_autoscaley_on(False)
ax.set_ylim([32,43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126,-112])

plt.scatter(training_examples['longitude'],
			training_examples['latitude'],
			cmap='coolwarm',
			c=training_targets['median_house_value']/training_targets['median_house_value'].max())

ax = plt.subplot(1,3,3)
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
_ = plt.show()