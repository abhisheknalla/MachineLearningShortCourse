from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn import tree

cancer = load_breast_cancer()
        # cancer.target_names = np.sort(cancer.target_names)
        # print(cancer)
########################################################
import random
from scipy.spatial import distance

def euc(a, b):
    return distance.euclidean(a,b)

class myKNN():
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        return self

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc (row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.Y_train[best_index]

X = cancer.data
Y = cancer.target

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#################################################################
my_classifier = myKNN()

my_classifier = my_classifier.fit(X_train, Y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy of myKNN:", accuracy_score(Y_test, predictions))

#################################################################
from sklearn.neighbors import KNeighborsClassifier
my_classifier2 = KNeighborsClassifier()

my_classifier2 =  my_classifier2.fit(X_train, Y_train)
predictions = my_classifier2.predict(X_test)

print("Accuracy of standard KNN:", (accuracy_score(Y_test, predictions)))
