#=============================implimenting muliple linear regression 
# this means more than one column culmn  in the x's 
#==========================================================================

# these is new consept is called dummy-variable-trap 
# this always associated with the data preproceesing phase
# this consept is available if this a column with string and more then 2 rows 


# there is principle is called: your data should lack multi-collinearity , this means
#  multi-collinearity  are the columns where the values are dependents upon one to another
# ex if i have the value of one i will automatically know the values of the others 
# if we look at the startups.csv in the states we find 3 states, this priciple appy then
 
import  matplotlib  as plt
import numpy as np
import pandas as pd 
from sklearn.model_selection import  train_test_split 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer 
from sklearn.linear_model import LinearRegression






# 1- reading the file of the data 
dataset3 = pd.read_csv('startups.csv')
x3 = dataset3.iloc[:,:-1].values  # matrix of the IV
y3 = dataset3.iloc[:,[-1]].values # verctor of the DV 


# 2-  no need to handle the missing data 
 

# 3-  clearing categorical data 
labelEncoder_x3 = LabelEncoder()
x3[:,3] = labelEncoder_x3.fit_transform(x3[:,3])

# 4- since number of the states is greater then 2 we need to ceate colunms for them
transformer = ColumnTransformer( 
    transformers = [('state', OneHotEncoder(), [3])],
    remainder='passthrough')

x3 = transformer.fit_transform(x3)


# 5 - avoid the dummy-variable-trap,
# deleting the one of the new 3 colunms ceated by the lelbelEncoded & Onehotencoder
x3 = x3[:,1:] 


# 6 - getting the train data and the test data 
x3_train,  x3_test, y3_trian, y3_test = train_test_split(x3, y3, test_size=0.3, random_state=0)
# the oreder  here matters we needs all x's then y's 



# 7 feature scaling S
standardScaler_x3 = StandardScaler()

# creat scaler for x
x3_train  = standardScaler_x3.fit_transform(x3_train) # this will calculate the mean and sd 
x3_test =  standardScaler_x3.transform(x3_test) 
# # we only tranform because we already get the mean and sd in th ss_x3_train 

# creat scaler for y
standardScaler_y3 = StandardScaler()
y3_trian =  standardScaler_y3.fit_transform(y3_trian)




# =============== implementing the lineare regression Algorithm 
# ==========================================================================

regressor3 = LinearRegression()
regressor3.fit(x3_train, y3_trian) 

# making the prediction for the y3_test 
y3_predict = regressor3.predict(x3_test)

# so to compare the y3_tain and y3_test we need to get both feature scalled 
# or both not feature scale, in this example we will revese the effect of 
# the feature scale from y3_predict , then we compare  
y3_predict = standardScaler_y3.inverse_transform(y3_predict)

print('regressor3  p0 ===> ', regressor3.intercept_)
print('regressor3 coef or th slop ===> ', regressor3.coef_)
# there is 5 values for coef  becuase the matrix IV has 5 columns 

#  let predict a hardcoded value using the featured scaled values only 
print(' predict y hardcoded  ===> ', 
   standardScaler_y3.inverse_transform(
       regressor3.predict([[2.23607,-1,0.139206,-0.702817,0.285133]])
       ))

# the mean square error 
import  math
from sklearn.metrics import mean_squared_error 

squareError3 = mean_squared_error(y3_test, y3_predict)
realSquarError3 = math.sqrt(squareError3)

# ======== backward-elimination =============
# to check the impectful columns from the metrix of IV for a give vector of DV 

import statsmodels.regression.linear_model as statsModelRegression
# the simple  linear regression class generate a b0 (bais coefficient)
# there is another way to get the prediction and has more features and never need to generate b0 
# as we know b0 is not associated with any column of metrix of IV 
# we can associate b0 to one column and that column has only 1's,so to not change the value of b0 
# statsModelRegression is good for showing the influence of column  a on DV 
# we need to force statsModelRegression to generate b0 
# we need to add column filled with 1's to the start of the IV,
  # the coefficient of that one will be considered as b0
x3 =  np.append( 
    arr = np.ones(shape = (23,1)).astype(float),# generating new metrix of IV contains a single column of ones 10 rows and 1 column
    values = x3,# i am  adding the new metrix to the metrix x3 
    axis = 1, # one means fuse the 2 metrixs horizontally on the x axis 
)

x3=x3.astype(float)


# creating new matrix of IV which gradually get optomized , we take all x's 
x3_optomize = x3[:,:]

# we creat new regressor from the statsModelRegrssion 
regressor3_SMR = statsModelRegression.OLS( exog = x3_optomize, endog = y3).fit()

# # we could predict  y3 as well
y3_predict_SMR = regressor3_SMR.predict(x3_optomize)

# so know this new regressor3_SMR will give me extra features like p value 

summary = regressor3_SMR.summary()        # this will print a table contain more statistics 

print('summary ===> ', summary)
# printing the summary wil show us a  table, we need to look at : P > |t|
# since the largest is great than 0.05, we need to regenerate x3_optomize and we keep less then 0.05

# new optomise , we skip the highest p value 0.795 which is for x2 , means index 2 
x3_optomize = x3[:,[0,1,3,4,5]]
regressor3_SMR = statsModelRegression.OLS( exog = x3_optomize, endog = y3).fit()
y3_predict_SMR = regressor3_SMR.predict(x3_optomize)
summary = regressor3_SMR.summary()       
print(' new summary ===> ', summary)


# one more new optomise , we skip the highest p value 0.483 which is for x2 , means index 2 
x3_optomize = x3[:,[0,1,3,4]]
regressor3_SMR = statsModelRegression.OLS( exog = x3_optomize, endog = y3).fit()
y3_predict_SMR = regressor3_SMR.predict(x3_optomize)
summary = regressor3_SMR.summary()       
print(' new summary ===> ', summary)


# one more new optomise , we skip the highest p value 0.280 which is for x1 , means index 2 
x3_optomize = x3[:,[0,2,3]]
regressor3_SMR = statsModelRegression.OLS( exog = x3_optomize, endog = y3).fit()
y3_predict_SMR = regressor3_SMR.predict(x3_optomize)
summary = regressor3_SMR.summary()       
print(' new summary ===> ', summary)


# one more new optomise , we skip the highest p value  0.630 which is for x1 , means index 2 
x3_optomize = x3[:,[0,2]]
regressor3_SMR = statsModelRegression.OLS( exog = x3_optomize, endog = y3).fit()
y3_predict_SMR = regressor3_SMR.predict(x3_optomize)
summary = regressor3_SMR.summary()       
print(' new summary ===> ', summary)



# the conclusion is the backware elimination is a technique used to know the P values
# which represet the most affacting colunms   in x's  in the prediction. we keep
# removing the colunms until we get only the less then 0.05  
# and the we can shoose the ring x colunm and y's to predict the result
# a good p value is less or equal than 0.05
















