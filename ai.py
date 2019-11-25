import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

#print(iris.keys())
#print(iris["DESCR"])

features = iris.data.T

sepal_length = features[0]
sepal_width = features[1]
petal_length = features[2]
petal_width = features[3]

#print(iris.feature_names) 
sepal_length_label = iris.feature_names[0]
sepal_width_label = iris.feature_names[1]
petal_length_label = iris.feature_names[2]
petal_width_label = iris.feature_names[3]

#print(sepal_length_label)

#wykres sepal length against sepal width
#kolory korespondują do odpowiednich gatunków
plt.scatter(sepal_length, sepal_width, c = iris.target)
plt.xlabel(sepal_length_label)
plt.ylabel(sepal_width_label)
plt.show()

#petal - to samo co wyżej
plt.scatter(petal_length, petal_width, c = iris.target)
plt.xlabel(petal_length_label)
plt.ylabel(petal_width_label)
plt.show()