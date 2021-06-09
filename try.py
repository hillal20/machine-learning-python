import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

     
# import the dataset
dataset = pd.read_csv("data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,[3]].values