import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# atplotlib inline

# df = pd.DataFrame({'a': [None, 1, None], 'b': [None, 1, 2]})


bins = 3

df = pd.read_csv("train.csv")
testdf = df
df1 = pd.read_csv("test.csv")
structureDF = pd.read_csv('Structure.txt', sep=' ', header=None, names=['att', 'att_name', 'types'], usecols=[1, 2])
numericNames = []
names = structureDF['att_name'].tolist()
types = structureDF['types'].tolist()
for i in range(len(names)):
    if types[i] == 'NUMERIC':
        numericNames.append(names[i])
df = df.dropna(how="all")
# prdf.dropna(how="all"))
# print(df1.dropna(how="all"))
# print(df)

for i in names:
    if i in numericNames:
        df.update(df[i].fillna(math.floor(df[i].sum() / df[i].count())))
        df1.update(df[i].fillna(math.floor(df[i].sum() / df[i].count())))
    else:
        df.update(df[i].fillna(max(df[i].astype(str))).str.lower())
        df1.update(df[i].fillna(max(df[i].astype(str))).str.lower())
        # change category columns
        if df[i].dtype == 'object':
            df[i] = df[i].astype('category')
            df[i] = df[i].cat.codes
        if df1[i].dtype == 'object':
            df1[i] = df1[i].astype('category')
            df1[i] = df1[i].cat.codes

######################################
# Discretization of numeric columns in df's :
for name in numericNames:
    df.update(pd.cut(df[name], bins, include_lowest=True, labels=[0, 1, 2]))
    df1.update(pd.cut(df1[name], bins, include_lowest=True, labels=[0, 1, 2]))

# print(df.head(10))
# df.info()
# print(df.describe())


X_train = df.iloc[:, :-1].values
y_train = df["class"].values
# print(X_train)
# print(y_train)
X_test = df1.iloc[:, :-1].values
y_test = df1["class"].values
# print(X_train)


scaler = StandardScaler()
scaler.fit(X_train)
normalizer = preprocessing.Normalizer().fit(X_train)
# print(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# print(X_train)
print("//////////////////////////////////////////////")

'''''

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
kmeans=KMeans(n_clusters=3)
kmeans.fit(X_train)
labels=kmeans.labels_
print(labels)
#testdf["cluster"]=labels
#dr=testdf.groupby("cluster").mean()
#print(dr)
df["cluster"]=labels
dr=df.groupby("cluster").mean()
print(dr.head(20))
#print(df.head(200))
y_kmeans = kmeans.predict(X_train)
#plt.scatter(X_train[:, 0], X_train[:, 1], c=y_kmeans, s=50, cmap='viridis')
#plt.show()
#centers = kmeans.cluster_centers_
#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
#kmeans.cluster_centers_
#plt.scatter(X_train[:,0],X_train[:,1],c=y_kmeans,cmap="rainbow")
#plt.show()
#x=kmeans.score(X_test)
#print("K-means",x)
print("//////////////////////////////////////////////")
from sklearn.metrics import confusion_matrix,classification_report
#print(confusion_matrix(y_train,kmeans.labels_))
#print(kmeans.cluster_centers_)
#print(classification_report(y_train,kmeans.labels_))
'''''

from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()
knn.fit(X_train, y_train)
x = knn.score(X_test, y_test)
print(x)
error = []
from sklearn.neighbors import KNeighborsClassifier

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

# Instantiate learning model (k = 3)
# classifier = KNeighborsClassifier(n_neighbors=3)

# Fitting the model
# classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = knn.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)*100
# print("score:",x*100,"%")
# print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
