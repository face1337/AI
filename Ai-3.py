'''training set vs testing set
75% for training, 25% for testing
'''
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import numpy as np

from sklearn.datasets import load_iris
iris = load_iris()

X_train, X_test, Y_train, Y_test = train_test_split(iris['data'], iris['target'], random_state=0) #Splitowanie danych oraz drugi parametr do celu ( to co próbujemy [przewidzieć])
knc = KNeighborsClassifier(n_neighbors=1)

knc.fit(X_train, Y_train) #X- data, Y - target

X_new = np.array([[5.0, 2.9, 1.0, 0.2]]) #Create numpy array | jakieś dane które podajemy, nie wiemy skąd się wzieły
#print(X_new.shape)

#prediction = knc.predict(X_new)
#print(prediction)

print(knc.score(X_test, Y_test))