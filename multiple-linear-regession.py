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

x3_train  = standardScaler_x3.fit_transform(x3_train) # this will calculate the mean and sd 
x3_test =  standardScaler_x3.transform(x3_test) 
# # we only tranform because we already get the mean and sd in th ss_x3_train 







