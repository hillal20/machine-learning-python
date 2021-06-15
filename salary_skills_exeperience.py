import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 



# import the dataset
dataset2 = pd.read_csv("salary_skills_exeperience.csv")
x2 = dataset2.iloc[:,[3]].values # matrix of IV is years of experience 
y2 = dataset2.iloc[:,[2]].values # vector of DV is the salary 

# we skip the step of the missing culomns in the rows 

# we skip the step of the categorical data because we don't have string data 

# we need to apply the the trianig data to create the model
from sklearn.model_selection  import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2, test_size = 0.33 , random_state=0)


# not need  feature scaling because x values are petaining only one culonm and 
# the values are only 2 figures which is under control , the scaling will take place if the 
# x is having more then one colunm and the values of one column is \
    
    
#=============================implimenting the linear regression 

#importing the linear regression class 
from sklearn.linear_model import LinearRegression
linearRegressionObj = LinearRegression()  

# we traing the data  
linearRegressionObj.fit(x2_train, y2_train) 

# we do the prediction by the unused data 
y2_predicted = linearRegressionObj.predict(x2_test)


# regesssore coef_ is  slop of the linearRegression line in the graph 
print("regesssore coef_ is : ", linearRegressionObj.coef_)
# regressor intercept is the initial y of the linearRegression line in the graph 
print("regesssore intercept is : ", linearRegressionObj.intercept_)

y_pridected_10years = linearRegressionObj.predict([[10]])


# visual representation of the train_set 
plt.scatter(x2_train,y2_train, color='red')

# to plot the line we have to provide the x's (x2_train) but to the values of y
# they have to be the predicted ones and not the trained ones. the reason why we 
# are using the predicted because it will give a stait line of the learned model
# if we give the train y's the line will never be straight becuase the real data
# is never that accurate
plt.plot(x2_train,linearRegressionObj.predict(x2_train), color='blue')
plt.xlabel('experience thru years')
plt.ylabel('salary')
plt.title('distribution of salaries over years')


#visual representation of the test_set
# we plot the test data first
plt.scatter(x2_test,y2_test, color='green')
# we keep the line  the same because we want to test the test data to it 
plt.plot(x2_train,linearRegressionObj.predict(x2_train), color='blue')
plt.xlabel('experience thru years test')
plt.ylabel('salary test')
plt.title('distribution of salaries over years test ')
























