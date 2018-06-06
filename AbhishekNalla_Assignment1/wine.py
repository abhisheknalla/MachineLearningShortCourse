from sklearn.datasets import load_wine
import numpy as np
from sklearn import tree

wine = load_wine()
test_idx = [10,100,150]

# print(wine.feature_names)
# print(wine.target_names)

# for i in range(len(wine.target)):
#     print("Index: %d Features: %s Target: %s" % (i, wine.data[i],wine.target[i]))

train_target = np.delete(wine.target, test_idx)
train_data = np.delete(wine.data, test_idx, axis=0)

test_target = wine.target[test_idx]
test_data = wine.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print("Expected:", test_target)
print("Predicted Ouput:", clf.predict(test_data))

from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                    feature_names = wine.feature_names,
                    class_names = wine.target_names,
                    filled = True, rounded = True,
                    impurity = False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("wine_abhi.pdf")
