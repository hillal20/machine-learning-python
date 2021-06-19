import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 1 -  getting the data 
dataset = pd.read_csv('logistic-regression-data.csv')
x = dataset.iloc[:,[2,3]]
y = dataset.iloc[:,[-1]]

# 2- no missing data 

# 3 - no categorical data 

# 4 - we need feature scaling because big gap between  age and salaries 
from sklearn.preprocessing import StandardScaler

scaler_x  = StandardScaler()
scaler_y = StandardScaler()

x = scaler_x.fit_transform(x)

# 5 - tain data rations 80-20
from sklearn.model_selection  import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0 )


#  6- classification via logistic regression 
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression()

# we trian the data to get the right model 
classifier.fit(x_train,y_train)

# predict result 
y_predict = classifier.predict(x_test)
y_predict_probability = classifier.predict_proba(x_test)


#  n = 200     predicted true  | predicted false 
# actual true    100           | 40  
# actual false   40            | 20

#      the total prediction  accuracy  is 100 + 20 = 120/200 = 60%

# we could calculate the accuracy using the confusion matrix 
from sklearn.metrics import confusion_matrix as confusionmatrix
cm = confusionmatrix(y_test, y_predict)

# behind the sceen it is working as linear regression but limited to between 0 and 1
print('classifier coefficient ===> ', classifier.coef_)
print('classifier intercept ===> ', classifier.intercept_)























