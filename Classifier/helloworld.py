from sklearn import tree

# features = [[140, "smooth"],[130, "smooth"],[150, "bumpy"], [170, "bumpy"]]
# labels = ["apple", "apple", "orange", "orange"]
features = [[140, 1],[130, 1],[150, 0], [170, 0]]
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()#classifier created / clf is  object of class DecisionTreeClassifier
clf = clf.fit(features, labels)#think of fit as find patterns in data, it is a method of the class DTC
#fit function finds patterns in data-training algorithm
print(clf.predict([[160,0]]))
