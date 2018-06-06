from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
Y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
my_classifier = my_classifier.fit(X_train, Y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))
