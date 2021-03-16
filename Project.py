import math
import operator
import random
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn import neighbors
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, Normalizer
import entropy_based_binning as ebb

'''
Shmuel Atias 300987443
Dmitry Korkin 336377429
Shay Peleg 302725643
'''
class Project:
    structureFile, trainFile, testFile, cleanTrainFile, cleanTestFile, names, numericNames = None, None, None, None, \
                                                                                             None, None, []
    X_train, y_train, X_test, y_test = None, None, None, None  # variables to compare and create CM and statistics

    joblib_Model = None  # variable to save and load a model

    def __init__(self, struct, train, test):
        self.structureFile = struct
        self.trainFile = train
        self.testFile = test
        self.names = struct['att_name'].tolist()
        types = struct['types'].tolist()
        for i in range(len(self.names)):
            if types[i] == 'NUMERIC':
                self.numericNames.append(self.names[i])

    def clean(self):

        # check for NaN rows and remove them
        self.cleanTrainFile = self.trainFile.dropna(how="all")  # remove a empty row if exists
        self.cleanTestFile = self.testFile.dropna(how="all")  # -||-
        self.cleanTrainFile = self.trainFile.dropna(subset=["class"])  # remove row if class NaN
        self.cleanTestFile = self.testFile.dropna(subset=["class"])  # -||-
        # outliers
        # check for missing values in columns and replace them
        for i in self.names:
            if i in self.numericNames:
                self.cleanTrainFile.update(self.cleanTrainFile[i].fillna(
                    math.floor(
                        self.cleanTrainFile[i].sum() / self.cleanTrainFile[i].count())))  # if NaN replace with avg
                self.cleanTestFile.update(self.cleanTestFile[i].fillna(
                    math.floor(self.cleanTestFile[i].sum() / self.cleanTestFile[i].count())))  # if NaN replace with avg
            else:
                self.cleanTrainFile.update(
                    self.cleanTrainFile[i].fillna(
                        max(self.cleanTrainFile[i].astype(str))).str.lower())  # if NaN replace with common
                self.cleanTestFile.update(
                    self.cleanTestFile[i].fillna(
                        max(self.cleanTestFile[i].astype(str))).str.lower())  # if NaN replace with common
                # change categories to numbers
                if self.cleanTrainFile[i].dtype == 'object':
                    self.cleanTrainFile[i] = self.cleanTrainFile[i].astype('category')
                    self.cleanTrainFile[i] = self.cleanTrainFile[i].cat.codes
                if self.cleanTestFile[i].dtype == 'object':
                    self.cleanTestFile[i] = self.cleanTestFile[i].astype('category')
                    self.cleanTestFile[i] = self.cleanTestFile[i].cat.codes

        self.X_train = self.cleanTrainFile.iloc[:, :-1].values
        self.y_train = self.cleanTrainFile["class"].values

        self.X_test = self.cleanTestFile.iloc[:, :-1].values
        self.y_test = self.cleanTestFile["class"].values
        # save clean file

        # return clean file

    def standardization(self):

        scaler = StandardScaler()
        scaler.fit(self.X_train)

        col = list(self.cleanTrainFile.iloc[:, :-1].columns)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        self.cleanTestFile = pd.DataFrame(self.X_test, columns=col)
        self.cleanTestFile["class"] = self.y_test

        self.cleanTrainFile = pd.DataFrame(self.X_train, columns=col)
        self.cleanTrainFile["class"] = self.y_train

    def normalization(self):

        normalize = Normalizer()
        normalize.fit(self.X_train)
        col = list(self.cleanTrainFile.iloc[:, :-1].columns)

        self.X_train = normalize.transform(self.X_train)
        self.X_test = normalize.transform(self.X_test)

        self.cleanTestFile = pd.DataFrame(self.X_test, columns=col)
        self.cleanTestFile["class"] = self.y_test

        self.cleanTrainFile = pd.DataFrame(self.X_train, columns=col)
        self.cleanTrainFile["class"] = self.y_train

    def discretization(self, bins, discr):
        label = []
        for i in range(bins):
            label.append(i + 1)

        def eqwidthbinnig(data, bins):
            w = ((max(data) - min(data)) / bins)  # width of eash bin
            ranges = []
            for i in range(0, bins + 1):
                ranges = ranges + [min(data) + w * i]
            res = {}
            for i in range(len(ranges) - 1):
                res[(ranges[i], ranges[i + 1])] = i
            for i in range(len(data)):
                for key in res.keys():
                    if key[0] <= data[i] <= key[1]:
                        data[i] = res[key]
            return data

        def eqfreq(data, num):
            sorted = []
            for i in range(len(data)):
                sorted.append(data[i])
            sorted.sort()
            binSize = int(len(sorted) / bins) + 1
            ranges = []
            for i in range(0, len(sorted), binSize):
                ranges.append(sorted[i])
                if i + binSize < len(data):
                    pass
                else:
                    ranges.append(max(sorted))
            res = {}
            for i in range(len(ranges) - 1):
                res[(ranges[i], ranges[i + 1])] = i
            for i in range(len(data)):
                for key in res.keys():
                    if key[0] <= data[i] <= key[1]:
                        data[i] = res[key]
            return data

        if discr == "equalwidth_pandas":
            for name in self.numericNames:
                self.cleanTrainFile.update(pd.cut(self.cleanTrainFile[name], bins, include_lowest=True, labels=label))
                self.cleanTestFile.update(pd.cut(self.cleanTestFile[name], bins, include_lowest=True, labels=label))
        if discr == "equalfreaqncy_pandas":
            for name in self.numericNames:
                self.cleanTrainFile.update(pd.qcut(self.cleanTrainFile[name], bins, labels=label, retbins=True))
                self.cleanTestFile.update(pd.qcut(self.cleanTestFile[name], bins, labels=label, retbins=True))

        if discr == "equalwidth":
            for n in self.numericNames:
                m = eqwidthbinnig(self.cleanTrainFile[n].values, bins)
                self.cleanTrainFile[n] = m
                m = eqwidthbinnig(self.cleanTestFile[n].values, bins)
                self.cleanTestFile[n] = m
        if discr == "equalfreaqncy":
            for n in self.numericNames:
                m = eqfreq(self.cleanTrainFile[n].values, bins)
                self.cleanTrainFile[n] = m
                m = eqfreq(self.cleanTestFile[n].values, bins)
                self.cleanTestFile[n] = m
        if discr == "entropybinning_external":
            for name in self.numericNames:
                x1, x2 = self.cleanTrainFile[name].astype("int64").to_numpy(), self.cleanTestFile[name].astype(
                    "int64").to_numpy()
                a1, a2 = ebb.bin_array(x1, nbins=bins, axis=0), ebb.bin_array(x2, nbins=bins, axis=0)
                list1, list2 = a1.tolist(), a2.tolist()
                d1, d2 = pd.DataFrame({name: list1}), pd.DataFrame({name: list2})
                self.cleanTrainFile.update(d1)
                self.cleanTestFile.update(d2)

        self.X_train = self.cleanTrainFile.iloc[:, :-1].values
        self.y_train = self.cleanTrainFile["class"].values

        self.X_test = self.cleanTestFile.iloc[:, :-1].values
        self.y_test = self.cleanTestFile["class"].values

    def entropy(self, col):
        value, counts = np.unique(col, return_counts=True)
        # print(value)
        # print(counts)
        # print(sum(counts))
        total = 0
        for i in counts:
            if i != 0:
                p = i / sum(counts)
                print("p:", p, "   log:", math.log(p, 2), "   ***", p * math.log(p, 2))
                total -= p * math.log(p, 2)

        return total

    def saveModel(self, filename):
        filename += ".sav"
        joblib.dump(self.joblib_Model, filename)
        return "model saved"

    def loadModel(self, filename):
        filename += ".sav"
        self.joblib_Model = joblib.load(filename)
        return "model loaded"


class KNNClassifier(Project):
    y_pred = []

    def __init__(self, struct, train, test):
        super().__init__(struct, train, test)

    def buildModel(self):
        if self.joblib_Model is None:
            self.joblib_Model = neighbors.KNeighborsClassifier()
            self.joblib_Model.fit(self.X_train, self.y_train)
        # x = knn.score(self.X_test, self.y_test)
        self.y_pred = self.joblib_Model.predict(self.X_test)

    def result(self):
        a = float(accuracy_score(self.y_test, self.y_pred) * 100)
        return "Sklearn KNN result is:\n" + '{}%'.format(a)

    def matrix(self):
        return str(confusion_matrix(self.y_test, self.y_pred))

    def report(self):
        return str(classification_report(self.y_test, self.y_pred))

    def visualization(self):
        error = []
        for i in range(1, 6):
            knn = neighbors.KNeighborsClassifier(n_neighbors=i)
            knn.fit(self.X_train, self.y_train)
            pred_i = knn.predict(self.X_test)
            error.append(np.mean(pred_i != self.y_test))

        plt.figure(figsize=(12, 6))
        plt.plot(range(1, 6), error, color='red', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)
        plt.title('Error Rate K Value')
        plt.xlabel('K Value')
        plt.ylabel('Mean Error')
        plt.show()


class SklearnNB(Project):
    y_pred = []

    def __init__(self, struct, train, test):
        super().__init__(struct, train, test)

    def buildModel(self):
        if self.joblib_Model is None:
            self.joblib_Model = GaussianNB()
            self.joblib_Model.fit(self.X_train, self.y_train)
        self.y_pred = self.joblib_Model.predict(self.X_test)

    def result(self):
        a = float(accuracy_score(self.y_test, self.y_pred) * 100)
        return "Sklearn NB result is:\n" + '{}%'.format((a))

    def matrix(self):
        return str(confusion_matrix(self.y_test, self.y_pred))

    def report(self):
        return str(classification_report(self.y_test, self.y_pred))


class NaiveBayesClassifier(Project):
    counter = 0

    def __init__(self, struct, train, test):
        super().__init__(struct, train, test)
        self.y_pred = []

    def build(self):
        def p_dict(attrName):
            # a function that returns a dict structured like so: (no/yes,category):probability
            class_attrCount_dict = self.cleanTrainFile["class"].value_counts().to_dict()
            table = self.cleanTrainFile.groupby(['class', str(attrName)], as_index=True).size()
            b = dict(table)
            # laplacian Correction
            if 0 in b.values():
                for key in b.keys():
                    b[key] += 1
            #######################
            for ClassAttr in class_attrCount_dict.keys():
                for key in b.keys():
                    if key[0] == ClassAttr:
                        b[key] /= class_attrCount_dict[ClassAttr]
            return b

        model = {}
        for n in self.names:
            if n != "class":
                model[n] = p_dict(n)
        return model

    def buildModel(self):
        if self.joblib_Model is None:
            self.joblib_Model = self.build()

        test_class_attrCount_dict = self.cleanTestFile["class"].value_counts().to_dict()
        size = self.cleanTestFile.shape[0]
        test_class_attrProbs_dict = {key: val / size for key, val in test_class_attrCount_dict.items()}

        def predict(row, class_probs):

            for key in self.joblib_Model.keys():
                for k in self.joblib_Model[key].keys():
                    if k[1] == row[key]:
                        if k[0] in class_probs.keys():
                            class_probs[k[0]] *= self.joblib_Model[key][k]
            return max(class_probs.items(), key=operator.itemgetter(1))[0]

        for i in range(self.cleanTestFile.shape[0]):
            probs = {key: val for key, val in test_class_attrProbs_dict.items()}
            self.y_pred.append(predict(self.cleanTestFile.iloc[i], probs))

    def result(self):
        a = float(accuracy_score(self.y_test, self.y_pred) * 100)
        return "NB result is:\n" + '{}%'.format((a))

    def matrix(self):
        return str(confusion_matrix(self.y_test, self.y_pred))

    def report(self):
        return str(classification_report(self.y_test, self.y_pred))


class KMeansClustering(Project):
    counter = 0
    tr = {}

    def __init__(self, struct, train, test):
        super().__init__(struct, train, test)

    def buildModel(self):

        kmeans = KMeans()
        kmeans.fit(self.X_train)

        labels = kmeans.predict(self.X_train)
        labels1 = kmeans.predict(self.X_test)

        self.cleanTrainFile["cluster"] = labels
        self.cleanTestFile["cluster"] = labels1

        # print(self.cleanTrainFile)
        # print(np.unique(self.cleanTrainFile["cluster"]))

        def clusterize(data):
            cluster = data.groupby(['cluster', 'class'], as_index=True).size()
            cluster_majoraty_results = {}
            for i in range(cluster.shape[0] // 2):
                temp = cluster[i].to_dict()
                max_class_attr = max(temp.items(), key=operator.itemgetter(1))[0]
                cluster_majoraty_results[i] = max_class_attr
            return cluster_majoraty_results

        self.tr = clusterize(self.cleanTrainFile)
        te = clusterize(self.cleanTestFile)
        for key in self.tr.keys():
            if self.tr[key] == te[key]:
                self.counter += 1

    def result(self):
        return "accurecy of clusterring: \n", (self.counter * 100) / len(self.tr), '%'

    def matrix(self):
        return "No matrix"


class SklearnID3(Project):
    y_pred = []

    def __init__(self, struct, train, test):
        super().__init__(struct, train, test)

    def buildModel(self):
        if self.joblib_Model is None:
            self.joblib_Model = tree.DecisionTreeClassifier()
            self.joblib_Modelclf = self.joblib_Model.fit(self.X_train, self.y_train)
        self.y_pred = self.joblib_Model.predict(self.X_test)

    def result(self):
        a = float(accuracy_score(self.y_test, self.y_pred) * 100)
        return "Sklearn ID3 result is:\n" + '{}%'.format(a)

    def matrix(self):
        return str(confusion_matrix(self.y_test, self.y_pred))

    def report(self):
        return str(classification_report(self.y_test, self.y_pred))


class ID3(Project):
    count = 0
    y_pred = []

    def __init__(self, struct, train, test):
        super().__init__(struct, train, test)

    def infoGain(self, parentDF, attr):
        categories, counts = np.unique(parentDF[attr], return_counts=True)
        d = dict(zip(categories, counts))
        probs = {key: val / sum(counts) for key, val in d.items()}

        entropies = {}
        for i in categories:
            filteredDF = parentDF.loc[parentDF[attr] == i]
            entropies[i] = float(entropy(filteredDF["class"].values))

        muls = []
        for i in categories:
            muls.append(probs[i] * entropies[i])

        parentEntropy = entropy(parentDF["class"].values)
        return parentEntropy - sum(muls)

    def buildModel(self):

        def buildTree(data, originaldata, features, target_attribute_name="class", parent_node_class=None):
            if len(np.unique(data[target_attribute_name])) <= 1:
                return np.unique(data[target_attribute_name])[0]
            elif len(data) == 0:
                return np.unique(originaldata[target_attribute_name])[
                    np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
            elif len(features) == 0:
                return parent_node_class
            else:
                parent_node_class = np.unique(data[target_attribute_name])[
                    np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

                item_values = [self.infoGain(data, feature) for feature in features]
                best_feature_index = np.argmax(item_values)
                best_feature = features[best_feature_index]

                tree = {best_feature: {}}

                features = [i for i in features if i != best_feature]

                for value in np.unique(data[best_feature]):
                    value = value
                    sub_data = data.where(data[best_feature] == value).dropna()

                    subtree = buildTree(sub_data, data, features, target_attribute_name, parent_node_class)

                    tree[best_feature][value] = subtree

                return tree

        def traverseTree(tree, row, class_attrs):
            if isinstance(tree, float):
                return tree
            for key in row.keys():
                if key in tree.keys():

                    item = row[key]
                    row.pop(key)
                    if item in tree[key].keys():
                        return traverseTree(tree[key][item], row, class_attrs)
                    return random.choice(class_attrs)

        def predict(tree, df):
            queries = df.iloc[:, :-1].to_dict(orient="records")

            categories, counts = np.unique(df["class"], return_counts=True)
            d = dict(zip(categories, counts))

            for i in range(len(df)):
                row = queries[i]
                res = traverseTree(tree, row, list(d.keys()))
                if res == df["class"][i]:
                    self.count += 1
                self.y_pred.append(res)

        if self.joblib_Model is None:
            self.joblib_Model = buildTree(self.cleanTrainFile, self.cleanTrainFile, self.cleanTrainFile.columns[:-1])
        predict(self.joblib_Model, self.cleanTestFile)

    def result(self):
        a = float(accuracy_score(self.y_test, self.y_pred) * 100)
        return "ID3 result is:\n" + '{}%'.format(a)

    def matrix(self):
        return str(confusion_matrix(self.y_test, self.y_pred))

    def report(self):
        return str(classification_report(self.y_test, self.y_pred))


# traindf = pd.read_csv("train.csv")
# testdf = pd.read_csv("test.csv")
# # print(traindf.info())
# structureDF = pd.read_csv('Structure.txt', sep=' ', header=None, names=['att', 'att_name', 'types'], usecols=[1, 2])
# a = ID3(structureDF, traindf, testdf)
# a.clean()
#
# a.standardization()
# print("1")
# a.discretization(3, "equalwidth_pandas")
# print("2")
# a.buildModel()
# print(a.result())
