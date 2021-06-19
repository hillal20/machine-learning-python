
# ==============================   K Nearest neighbour ===============

# holds the training data in memory (salary , purchase , age ) per every point
# let say we need to project all points on a graph , if pusrchase is true so the point 
# will be red and if not the point is green
# let say that we have new point to plot , the KNN algorithm will look for the 6 nearest 
# points on the graph , we count how many red and geen , and we clor the point based on 
# the biggest number of color  
# we have the option to choose the k how much 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# 1- obtaining data 
dataset = pd.read_csv("k-nearest-neighbor.csv")
x4 = dataset.iloc[:,[2,3]].values
y4 = dataset.iloc[:,[-1]]

# 2 - no missing data
# 3 - no categorical data 

# 4 - feature scaling 
from sklearn.preprocessing import StandardScaler
x4_scaler = StandardScaler()
y4_scaler = StandardScaler()


x4 = y4_scaler.fit_transform(x4)
# y4 =  y4_scaler.fit_transform(y4)

# 5 - training data 
from sklearn.model_selection import train_test_split
x4_train, x4_test , y4_train, y4_test = train_test_split(x4, y4, test_size=0.3, random_state=0)

# implementing k nearest neighbor algorithm 
from sklearn.neighbors import KNeighborsClassifier 
classifier = KNeighborsClassifier(n_neighbors=7)
# we always try to estimate k => sqr(number of rows, it is likely to be odd number)

# we train the model to find the relation between x and y 
classifier.fit(x4_train, y4_train)

# we predict 
y4_predict = classifier.predict(x4_test)


# we check the accuracy of the prediction 
from sklearn.metrics import confusion_matrix

confusionResult = confusion_matrix(y4_test, y4_predict)

























