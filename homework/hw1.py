import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

names = ['sepal length', 'sepal width', 'petal lengt', 'petal width', 'class']
data = pd.read_csv("iris.data", names=names)

print(data.head(10))
print(data.shape)
print(data.dtypes)
print(data.info())

data_class = data.groupby('class').size()

print(data_class)

describe = data.describe()

print(describe)

data_without_class=data.drop(columns = ['class'],axis = 1)

print(data_without_class.skew())
print(data_without_class.kurtosis())

pearson = data_without_class.corr(method='pearson')
print(pearson)



#histogramm

data.hist(figsize=(8,8))
plt.show()



#nuclear function

data.plot(kind='kde', subplots=True, layout=(3,3), sharex=False,sharey=False,figsize=(8,8))
plt.show()


#box with mustache

data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(8,8))
plt.show()


#correlogram

fig = plt.figure()
ax = fig.add_subplot(111)
warm = ax.matshow(pearson, vmin=-1, vmax=1)
fig.colorbar(warm)
ticks = np.arange(0,4,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
names2=['sepal length', 'sepal width', 'petal lengt', 'petal width']
ax.set_xticklabels(names2)
ax.set_yticklabels(names2)
plt.show()



#scattering

scatter_matrix(data,figsize=(8,8))
plt.show()