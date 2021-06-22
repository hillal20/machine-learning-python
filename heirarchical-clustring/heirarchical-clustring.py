# agglomirative heirrachical clusting 
# treat each point as cluster , let say we have 100 points  means 100 cluster
# 2 closest points and fuse them into one cluster 
# 2 closest clusters fuse into one bigger cluster
# after the merge of two clusters. the distance between them is not easy to calculate 
# Linkage is a way to solve them 
# Single lingkage : the distance between 2 closest  points from 2 clusters
# Average Linkage : is the average of all the possible linkages 
# Ward Method : find the mimimum increase of wcss if a merge happen between 2 clusters 
# after getting one large cluster, we contruct a dendogram to find the optimal k (number of clusters )
# a dendogram show how the merging is happening for the bottom to the top, it display the order of the 
# merging between clusters. the hight of the pilar formed due to the merge is the distance between 2 clusters 
# the tallest pilar reprent the merge of the last clusters
# to decide the optimal k of the cluseters,  we draw horistontal lines touching the top of the pliars and
# we condider only those vertical(lines pass thru points) lines which are not intersecting by 
# the horizantal lines, then  we choose the longest line or the segment which is not cut , 
# then draw  an horisontal line inside it and count how many intersection is there.,
# so the number of the clusters is the number of the intersections 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 1 obtaining data 
dataset = pd.read_csv('heirarchical-clustring.csv')
x5 = dataset.iloc[:,[3,4]].values
# there id no y because it is a classification there is no need to DV 

# 2 - there is no missing data 

# 3 - there is not categorical data 

# 4 - there is ne tairing or testing data 

# 5 - there is no feature scaling 

# 6 - cnnstruction of the dendogram to get the optimal number of the clusters 
# from scipy.cluster import hierarchy as hr 

# dendrogram = hr.dendrogram(hr.linkage(x5,method='ward',  metric='euclidean' ))
# # ward method means minmum increase of wcss when 2 cluster get fused 

# plt.title('dendogram for customers grouping')
# plt.xlabel('Customers')
# plt.ylabel('customers')


# =====  implimenting the heierarchical clustring 
from sklearn.cluster import AgglomerativeClustering as aggClustring 
# 2 is assumed based on watching the dendogram, the number of the intersets 
hhclustering = aggClustring(n_clusters=2,   affinity="euclidean", linkage='ward')

y5_predict = hhclustering.fit_predict(x5)




# visual representation 

plt.scatter(
           x5[y5_predict == 0 ,0], x5[y5_predict == 0,1],
            s = 100,
            c='red',
            label='group 0')

plt.scatter(
            x5[y5_predict == 1 ,0], x5[y5_predict == 1,1],
            s = 100,
            c='blue',
            label='group 0')

