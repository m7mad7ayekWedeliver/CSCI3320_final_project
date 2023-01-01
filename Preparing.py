import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv("health_data.csv")
plt.boxplot(df["column"])
plt.show()

from sklearn import preprocessing

names = df.columns
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=names)

from sklearn import preprocessing

df = pd.read_csv('dataset.csv')
min_max_scaler = preprocessing.MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df)
df = pd.DataFrame(df_scaled)


from sklearn.preprocessing import Binarizer

df = pd.read_csv('testset.csv')

#we’re selecting the columns to binarize
age = df.iloc[:, 1].values
gpa = df.iloc[: ,4].values

#now we turn them into values we can work with
x = age
x = x.reshape (1, -1)
y = gpa
y =y.reshape (1, -1)

#we need to set a threshold to define as 1 or 0
binarizer_1 = Binarizer(35)
binarizer_2 = Binarizer(3)
#finally we run the Binarizer function
binarizer_1.fit_transform(x)
binarizer_2.fit_transform(y)



from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris = load_iris()
x, y = iris.data, iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)