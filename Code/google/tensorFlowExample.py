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

# load dataframe online and save it offline
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
#california_housing_dataframe.to_csv('california_housing_dataframe.csv', sep=',')

#california_housing_dataframe = pd.read_csv('california_housing_dataframe.csv', sep=',')

# randomising the data to make sure no pathological ordering effects that might harm the performance of the Stochastic Gradient Descent exists
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe['median_house_value'] /= 1000.0

print(california_housing_dataframe.describe())

# define the input deature: total_rooms
my_feature = california_housing_dataframe[['total_rooms']]
# configure a numeric feature column for total_rooms
feature_columns = [tf.feature_column.numeric_column('total_rooms')]

# define the target/label
targets = california_housing_dataframe['median_house_value']

# configure the linear regressor
# use gradient descent as the optimizer for traininf the model
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# configure the linear regression model with our feature columns and optimizer 
# set a learning rate of 0.00000001 for the Gradient Descent
linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)

# define the input function
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


# print(my_feature)
# print(feature_columns)

_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(my_feature, targets),
    steps=100
)

# create an input function for prediction
# Since we're making just one prediction for each example, we don't need to repeat 
# or shuffle the data here
prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# call predict() on the linear_regressor to make predictions
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# format predictions as a NuPy array, so we can calculate error metrics
predictions = np.array([item['predictions'][0] for item in predictions])

# print mean square errors and root mean squared errors
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)

min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print("Min. Median House Value: %0.3f" % min_house_value)
print("Max. Median House Value: %0.3f" % max_house_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)

# Reducing the model errors
calibration_data = pd.DataFrame()
calibration_data['predictions'] = pd.Series(predictions)
calibration_data['targets'] = pd.Series(targets)
print(calibration_data.describe())
print(calibration_data.head())

# visualise our model
sample = california_housing_dataframe.sample(n=300)

# get the min and max total_rooms value
x_0 = sample['total_rooms'].min()
x_1 = sample['total_rooms'].max()
# retreive the final weight and bias generaed during training
weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
# get the predicted median_house_value for the min and max total_rooms values
y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias
# plot our regression line from (x_0, y_0) to (x_1, y_1)
plt.plot([x_0, x_1],[y_0, y_1],c='r')
# label the graph axes
plt.ylabel('median_house_value')
plt.xlabel('total_rooms')
# plot a scatter plot from our data sample
plt.scatter(sample['total_rooms'], sample['median_house_value'])
# display graph
plt.show()