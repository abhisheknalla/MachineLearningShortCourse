from sklearn import tree

#0----> RED
#1----> BLUE
features = [[0, 2],[0, 5], [1, 20],[1, 15]]
labels = ["Liverpool", "ManUtd", "Chelsea", "ManCity"]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(clf.predict([[1,16]]))
