# importing required packages
import csv
import math
import random
import argparse
import numpy as np
from scipy import interp
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument(
    '--k_twodim',
    type=int,
    help='Value of k for twodim dataset',
    required=True)
parser.add_argument(
    '--k_wine',
    type=int,
    help='Value of k for wine dataset',
    required=True)
args = parser.parse_args()

#reading CSV file - TwoDimHard
with open('TwoDimHard.csv') as csvtrfile:
    reader = csv.DictReader(csvtrfile)
    data = {}
    for row in reader:
        for header, value in row.items():
          try:
            data[header].append(value)
          except KeyError:
            data[header] = [value]
    ID = np.array(data['ID']).astype(np.int)
    x1 = np.array(data['X.1']).astype(np.float)
    x2 = np.array(data['X.2']).astype(np.float)
    clust_orig = np.array(data['cluster']).astype(np.int)


# k for twodim dataset
k = args.k_twodim

def dist_fun(a, b):
    return np.sqrt(np.sum((a-b)**2))

#Putting all the attributes into a 2D array where each row represents a record in the dataset
rec = np.empty(shape=(400, 2))
arr = [x1, x2]
new_arr = zip(*arr)
rec = np.array(list(new_arr))
x_trans = rec.astype(np.float)

#Normalizing continuous data attributes in range zero to one
scaler = MinMaxScaler()
x = scaler.fit_transform(x_trans)

# finding initial centroid
c = np.empty([k, 2])
index = np.random.randint(x.shape[0], size=k)
for i in range (0, k):
    c[i][0] = x1[index[i]]
    c[i][1] = x2[index[i]]

#storing prevoius values of centroids for computing errors
c_prev = np.zeros(c.shape)
clust_pred = np.zeros(400, dtype=int)
err = dist_fun(c, c_prev)
size_pred = np.empty([k])

# Loop will run till the error becomes zero
distances = np.empty(shape=(k))
while err != 0:
    # Assigning each value to its closest cluster
    for i in range(0, 400):
        for j in range (0, k):
            distances[j] = dist_fun(x[i], c[j])
        cluster = np.argmin(distances)
        clust_pred[i] = cluster
    # Storing the old centroid values
    c_prev = deepcopy(c)
    # Finding the new centroids by taking the average value
    for i in range(0, k):
        points = [x[j] for j in range(400) if clust_pred[j] == i]
        size_pred[i] = len(points)
        c[i] = np.mean(points, axis=0)
    err = dist_fun(c, c_prev)

#preparing output for writing into a csv file
heading_list = []
heading_list.append('Record Number')
heading_list.append('Predicted Cluster')
list_twodim = [[0 for j in range (2)] for i in range (400)]
for i in range(0, 400):
    list_twodim[i][0] = ID[i]
    list_twodim[i][1] = clust_pred[i]

#writing to csv file
with open('twodim_results.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(heading_list)
    wr.writerows(list_twodim)

#wine dataset
#reading CSV - wine
with open('wine.csv') as csvtrfile:
    reader = csv.DictReader(csvtrfile)
    data = {}
    for row in reader:
        for header, value in row.items():
          try:
            data[header].append(value)
          except KeyError:
            data[header] = [value]
    ID = np.array(data['ID']).astype(np.int)
    fx = np.array(data['fx_acidity']).astype(np.float)
    vol = np.array(data['vol_acidity']).astype(np.float)
    citric = np.array(data['citric_acid']).astype(np.float)
    resid = np.array(data['resid_sugar']).astype(np.float)
    chl = np.array(data['chlorides']).astype(np.float)
    free = np.array(data['free_sulf_d']).astype(np.float)
    tot = np.array(data['tot_sulf_d']).astype(np.float)
    den = np.array(data['density']).astype(np.float)
    ph = np.array(data['pH']).astype(np.float)
    sulph = np.array(data['sulph']).astype(np.float)
    alc = np.array(data['alcohol']).astype(np.float)
    clust_orig_wine = np.array(data['quality']).astype(np.int)


# value of k for wine dataset
k = args.k_wine

#Putting all the attributes into a 2D array where each row represents a record in the dataset
rec = np.empty(shape=(1599, 11))
arr = [fx, vol, citric, resid, chl, free, tot, den, ph, sulph, alc]
new_arr = zip(*arr)
rec = np.array(list(new_arr))
wine_trans = rec.astype(np.float)


#Normalizing continuous data attributes in range zero to one
scaler_wine = MinMaxScaler()
wine = scaler_wine.fit_transform(wine_trans)

#finding initial centroid
c_wine = np.empty([k, 11])
index = np.random.randint(wine.shape[0], size=k)
for i in range (0, k):
    for j in range(0, 11):
        c_wine[i][j] = wine[index[i]][j]

#storing previous value of centroids for computing errors
c_wine_prev = np.zeros(c_wine.shape)
clust_pred_wine = np.zeros(1599, dtype=int)
err = dist_fun(c_wine, c_wine_prev)
size_pred_wine = np.empty([k])

# Loop will run till the error becomes zero
distances_wine = np.empty(shape=(k))
while err != 0:
    # Assigning each value to its closest cluster
    for i in range(0, 1599):
        for j in range (0, k):
            distances_wine[j] = dist_fun(wine[i], c_wine[j])
        cluster = np.argmin(distances_wine)
        clust_pred_wine[i] = cluster
    # Storing the old centroid values
    c_wine_prev = deepcopy(c_wine)
    # Finding the new centroids by taking the average value
    for i in range(0, k):
        points = [wine[j] for j in range(1599) if clust_pred_wine[j] == i]
        size_pred_wine[i] = len(points)
        c_wine[i] = np.mean(points, axis=0)
    err = dist_fun(c_wine, c_wine_prev)

#preparing output for writing into a csv file
list_wine = [[0 for j in range (2)] for i in range (1599)]
for i in range(0, 1599):
    list_wine[i][0] = ID[i]
    list_wine[i][1] = clust_pred_wine[i]

#writing to csv file
with open('wine_results.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(heading_list)
    wr.writerows(list_wine)





