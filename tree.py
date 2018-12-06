#se importa el clasificador de arbol
from sklearn.tree import DecisionTreeClassifier
#Se importan los datasets que se van a trabajar
from sklearn.datasets import load_breast_cancer, load_iris
#Se importa el divisor de datos para sacr los datos de entrenamiento y de test
#la data en Xs y Ys de entrenamiento y de test
from sklearn.model_selection import train_test_split
#Graficar el arbol de desiciones que generamos
from sklearn.tree import export_graphviz
import graphviz
from graphviz import Digraph
import numpy as np 
print("HOLA")
import matplotlib.pyplot as plt 
from os import system
# os.environ["PATH"] += os.pathsep + 'c:/Program Files (x86)/Graphviz2.38/bin/'﻿



"""__________________"""
iris = load_iris()
print(type(iris))
print("keys:			",iris.keys())
print("data:			",iris["data"])
print("target_names:	",iris["target_names"])
print("target:			",iris["target"])
print("feature_names:	",iris["feature_names"])

"""Divide la data en set de entrenamiento y de test"""
X_train, X_test, Y_train, Y_test = train_test_split(iris["data"], iris["target"])
print("X_train:			", X_train.shape)
print("X_test:				", X_test.shape)
print("Y_train:			", Y_train.shape)
print("Y_test:				", Y_test.shape)

"""__________________"""
arbol = DecisionTreeClassifier()
print("Entreno:			",arbol.fit(X_train, Y_train))
print("Comprobacion1:		",arbol.score(X_test, Y_test))
print("Comprobacion2:		",arbol.score(X_train, Y_train))

"""__________________"""
G = export_graphviz(arbol,out_file='arbol.dot',class_names=iris.target_names,
	feature_names=iris.feature_names,impurity=False,filled=True)

with open('arbol.dot') as f:
	dot_graph = f.read()

# graph=graphviz.Source(dot_graph)
# graphviz.Source(dot_graph).view()
# graph.view()
graphviz.Source(dot_graph).render ('arbol', view=True, format='png')
# graph.render ('arbol', view=True, format='png')

caract = iris.data.shape[1]
print(caract)
plt.bar(range(caract), arbol.feature_importances_)
plt.xticks(np.arange(caract), iris.feature_names)
plt.xlabel("importancia de las caracteristicas")
plt.ylabel("Caracteristica")
plt.show()