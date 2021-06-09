# setting the virtul evirenment for python we need to install pipenv
# we cd into the folder of the project then we type : pip install pipenv
# then we need to type : pipenv install requests 
# if you see the Pipfile inside the project this means the project is ready 




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
imputer = imputer.fit(x[:, 1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
print('x ===> ', x )