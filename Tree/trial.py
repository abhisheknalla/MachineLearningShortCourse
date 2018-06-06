from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

iris = load_iris()
test_idx = [4,124,78]

train_data = np.delete(iris.data, test_idx, axis = 0)
train_target = np.delete(iris.target, test_idx)

test_data = iris.data[test_idx]
test_target = iris.target[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print("Expected", test_target)
print("Predicted", clf.predict(test_data))
