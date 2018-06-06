from sklearn.datasets import load_iris

iris = load_iris()#load_iris() is a class

print(iris.feature_names)#classss attributes
print(iris.target_names)

print(iris.data[0])
print(iris.target[0])

for i in range(len(iris.target)):
    print("Example: %d, Label: %s, Features %s" % (i, iris.target[i], iris.data[i]))
