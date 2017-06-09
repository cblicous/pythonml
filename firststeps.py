
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

fruits = pd.read_table('fruit_data_with_colors.txt')
fruits.head()

#define a dictionary , map label -> name
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(),fruits.fruit_name.unique()))
lookup_fruit_name

X = fruits[['mass', 'width', 'height']]
y = fruits['fruit_label']

#raondom state means fixed random , if not set random
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

#2d scatter plot
from matplotlib import cm
cmap = cm.get_cmap('gnuplot')
scatter = pd.scatter_matrix(X_train,c=y_train,marker = 'o',s=40,hist_kwds={'bins':15},figsize=(12,12),cmap=cmap)
scatter.show

#3d
from mpl_toolkits.mplot3d import Axes3d
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_train['width'], X_train['height'],X_train['mass'], c = y_train , marker = 'o', s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('mass')
plt.show()

# knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)

# check the score of the algorithm
knn.score(X_test,y_test)
