
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 


# import the dataset
dataset = pd.read_csv("data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,[3]].values

# delete the row where all the culomns are null 
#dataset1 = dataset.dropna(how = 'all', axis = 0 )
# delete the row where one  the culomns are null 
#dataset1 = dataset.dropna(how = 'any', axis = 0 )



# hadling the missing data 
from sklearn.impute  import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(x[:, 1:3])  # fit do the logic 
x[:,1:3] = imputer.transform(x[:,1:3]) # trasform return the result 

# handling categorical data like text or string to turn them to degits  
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# labelecoded is used to switch strings to numbers to use them in math algos 
labelEncoder_x = LabelEncoder()
x[:,0] = labelEncoder_x.fit_transform(x[:,0])

# because  the new country number will  effect the salary,  we need only to
# mentioned their existance by 1 or 0 in a specific country, to treat the coutries same 
# onehotencoding is require because the number of the countries is greater then 2 values 
transformer = ColumnTransformer(
    transformers=[('country', OneHotEncoder(), [0])],
    remainder='passthrough')

x = transformer.fit_transform(x) 


# we need to convert y vector values into numbers as well
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y[:,0])


# ================== trainingsets and testsets 
# in the machine learning world we always avoid to use 100 % of the data becuase we     
# we need to use build the model based on 80% and we test the validity of the model based 
# 20% 

from sklearn.model_selection  import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2 , random_state=0)


#===============  feaures scaling =========================
# 1-standardisation 
# s = (x- mean(x's)) / sd(x's)

from sklearn.preprocessing import StandardScaler
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train) # covers 80% of data 
# fit device the logic 
# tranform applys the deviced logic and return data 
x_test = ss_x.transform(x_test) # covers 20% of data 
### i did not use fit and i used only the trasform becuase the line 70 applied 
## the standardisation and already calculated the the mean and the sd 
## so we need to keep them the same and therefor we use only trasform and no need to device the logic 


## ==== simple linear regression : yhot = titaZiro + titaOne * x
## titaOne = sum of:  (x - mean(x's)) * (y - mean(y's))/(x - mean(x's))^2
## titaZero = mean(y's) - titaOne * mean(x's)
## unit-error is : y/yHot ( culculted via the linear regression fomula, predicted value )
## mean square error : sum of (y - yHot)^2
## root mean saquar error: root((y - yHot)^2)















