# it is an easy way to visualize the data 
# there is 2 algorithms based on claustring 
# k-means claustring algorithm is one of them 
# we assume that the ploting is on 2 dementional graph x and y 
# there is  2 priciple aspects of grouping the data 
# k is the number of the groups or clausters 
# we assign centroids, there are random points 
# for each point in the graph we calculate the distances between each point and every centroid 
# the point will be assigned to the closest centroid 
# shift the centroids and we need to reset the points to their new centroids
# keep shifting the centroids until there is no reassignment happen, it will be the final cut 
# of the graph, because this mean that the  centroids are in the middle of the clausters 
# we decide the number of the shifting times allowed 
# sometimes there is a problem rise "randon initialization trap" , this problem is that
# we could shap a lot vesions of grouping and its is because the random selection of the centroids
# in the begining. this problem is inhirit in this algorithm and no way to avoid it 
# there is a solution for it which is "k-means++" 
# the centroids are placed in the most dense areas 
# k-means intuition is finding the optimal number  of clusters 
# WCSS : within cluster sum of squares, is used to get the right numbers of clusters 
# in other words wcss is the sum of  all the distances squared ,
# between every point and its centroid 
# from this formula of wcss we conclude that as we group more as 
# the distances betweeen points its centroids will be small so the wcss will 
# be small and vise versa. as k increase as the wcss decrease  
# the optimal number for wcss vs  K is when k = 3 becuse in the turning point the graph elbow 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# 1- reading the data 
dataset = pd.read_csv('claustring.csv')

# it is a big mistake to think for IV and DV here in the claustring 
# we creat x based on all the culonms i need. there is no need to y here 


# 2- we pick the right column to group 
x5= dataset.iloc[:,[3,4]].values # annual income and spending score 


# 3- since the 2 column are 2 digits or 3 digits there is no need to scale 

# 4- we try to find the right number of k 
from  sklearn.cluster import KMeans 

# we creat a list to hold all wcss 
wcss = []


for i in range(1,11):
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10) 
    # k-means++ wiil palce the centroids in the dense areas 

    # if system found a problem to creat the clausters, the system will get the  minimal movement 
    # of the centroids as the right solution for claustring 

    kmeans.fit(x5) # in every loop we give the data and we append the result 
    wcss.append(kmeans.inertia_)
    print("inertia ======> ",kmeans.inertia_)

  print("wcss ======> ",wcss)
# we plot the graph of wcss vs k clausters , to explain the elbow 
# plt.plot(range(1,11), wcss)
# plt.xlabel("number of clausters")
# plt.ylabel("wcss-values")
# plt.title("wcss vs k clausters")



# 5- implementing k-means by 5
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10) 

# we predict 

k_means_predict = kmeans.fit_predict(x5)

# the way we read the k_means_predict is we compare it to x5, each row from x5
# is located at the clauster number mentioned in the same   k_means_predict row 


# 6 - ploting the points from x5 

plt.scatter(
    # we plot every point based on the the clauster with the k number corresponded 
    x5[k_means_predict == 0 ,0], x5[k_means_predict == 0,1],
    s = 100,# size of the points with pixcels on the plot 
    c='red',
    label='clauster 0')

plt.scatter(
     x5[k_means_predict == 1 ,0], x5[k_means_predict == 1,1],
    s = 100,
    c='blue',
    label='clauster 1')


plt.scatter(
     x5[k_means_predict == 2 ,0], x5[k_means_predict == 2,1],
    s = 100,
    c='green',
    label='clauster 2')

plt.scatter(
     x5[k_means_predict == 3 ,0], x5[k_means_predict == 3,1],
    s = 100,
    c='orange',
    label='clauster 3')



plt.scatter(
     x5[k_means_predict == 4 ,0], x5[k_means_predict == 4,1],
    s = 100,
    c='purple',
    label='clauster 4')

plt.xlabel('annual income')
plt.ylabel('spending score')


# 7 - we plot the centroids 
plt.scatter(
# average of the x-cordine of each group,  average of the y-cordinate of each group 
    kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
    s = 300, # size in pixele 
    c='black',
    label = 'centroids'
    
    )


plt.legend()







