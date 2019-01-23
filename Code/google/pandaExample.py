from __future__ import print_function

import pandas as pd
import numpy as np
print(pd.__version__)

city_names = pd.Series(['San Francisco','Yamoussoukro', 'Abidjan', 'San Jose'])
population = pd.Series([456245,8920,89561,1234567])
cities = pd.DataFrame({'City names': city_names, 'Population':population})
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92, 85])
cities['Population density'] = cities['Population'] / cities['Area square miles']
cities['holy'] = pd.Series(city_names.apply(lambda val: 'San' in val) & cities['Area square miles'].apply(lambda val: val>50))
print(cities)
cities=cities.reindex(np.random.permutation(cities.index))
print(cities)
# california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
# print(california_housing_dataframe.describe())
# print(california_housing_dataframe.head())
# california_housing_dataframe.hist('housing_median_age')

### numpy
# population/1000.
# np.log(population)
# population.apply(lambda val: val>100000)

