#se importa el clasificador de arbol
from sklearn.tree import ExtraTreeClassifier
#Se importan los datasets que se van a trabajar
from sklearn.datasets import load_breast_cancer, load_iris
#Se importa el divisor de datos para sacr los datos de entrenamiento y de test
#la data en Xs y Ys de entrenamiento y de test
from sklearn.model_selection import train_test_split
#Graficar el arbol de desiciones que generamos
from sklearn.tree import export_graphviz
import graphviz
from graphviz import Digraph, Graph
import numpy as np 
print("HOLA")
import matplotlib.pyplot as plt 
from os import system




"""__________________"""
iris = load_iris()
iris = load_breast_cancer()
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
arbol = ExtraTreeClassifier(max_depth=3)
print("Entreno:			",arbol.fit(X_train, Y_train))
print("Comprobacion1:		",arbol.score(X_test, Y_test))
print("Comprobacion2:		",arbol.score(X_train, Y_train))

"""__________________"""
G = export_graphviz(arbol,out_file='arbol.dot',class_names=iris.target_names,
	feature_names=iris.feature_names,impurity=False,filled=True)

with open('arbol.dot') as f:
	dot_graph = f.read()

graphviz.Source(dot_graph).render('arbol', view=False, format='png')
# graphviz.Source(dot_graph).view()
# graph=graphviz.Source(dot_graph)
# graph.render ('arbol', view=True, format='png')
# graph.view()

"""Esto si va en el programa"""
caract = iris.data.shape[1]
print(caract)
plt.bar(range(caract), arbol.feature_importances_)
plt.xticks(np.arange(caract), iris.feature_names)
plt.xlabel("importancia de las caracteristicas")
plt.ylabel("Caracteristica")


n_classes = 3
plot_colors = "bry"
plot_step = 0.02

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
								[1, 2], [1, 3], [2, 3]]):
	print (pair, pairidx)
	X = iris.data[:, pair]
	Y = iris.target
	clf = ExtraTreeClassifier(max_depth = 3).fit(X, Y)

	plt.subplot(2,3, pairidx + 1)

	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
						 np.arange(y_min, y_max, plot_step))
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

	plt.xlabel(iris.feature_names[pair[0]])
	plt.ylabel(iris.feature_names[pair[1]])
	plt.axis("tight")

	for i, color in zip(range(n_classes), plot_colors):
		idx = np.where(Y == i)
		print (i)
		# print (idx)
		# plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
		# 			cmap=plt.cm.Paired)
	plt.axis("tight")
plt.suptitle("Ejemplos de clasificador de arboles")
plt.legend()
plt.show()

