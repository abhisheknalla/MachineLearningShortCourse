from sklearn.datasets import load_digits
import numpy as np
from sklearn import tree

digits = load_digits()
# digits = np.sort(digits)
# print(digits.target_names)

test_idx = []
for i in range(5,1797,180):
    test_idx.append(i)

# print(test_idx)
train_target = np.delete(digits.target, test_idx)
train_data = np.delete(digits.data, test_idx, axis=0)

test_target = digits.target[test_idx]
test_data = digits.data[test_idx]

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier =  my_classifier.fit(train_data, train_target)
predictions = my_classifier.predict(test_data)

print("Expected:", test_target)
print("Predicted Ouput:", predictions)

from sklearn.metrics import accuracy_score
print("Accuracy:", (accuracy_score(test_target, predictions)))
