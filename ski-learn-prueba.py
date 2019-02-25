import numpy as np 
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

"""__________________"""
iris = load_iris()
print(type(iris))
print("keys:			",iris.keys())
print("data:			",iris["data"])
print("target_names:	",iris["target_names"])
print("target:			",iris["target"])
print("feature_names:	",iris["feature_names"])

"""__________________"""
X_train, X_test, Y_train, Y_test = train_test_split(iris["data"], iris["target"])
print("X_train:			", X_train.shape)
print("X_test:				", X_test.shape)
print("Y_train:			", Y_train.shape)
print("Y_test:				", Y_test.shape)

"""__________________"""
knn = KNeighborsClassifier(n_neighbors=6)
print("Entreno:			",knn.fit(X_train, Y_train))
print("Comprobacion:		",knn.score(X_test, Y_test))
print("Prediccion:			",iris.target_names[knn.predict([[5., 3., 5., 1.8]])])
