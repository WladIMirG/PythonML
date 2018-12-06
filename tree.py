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
export_graphviz(arbol,out_file='arbol.dot',class_names=iris.target_names,
	feature_names=iris.feature_names,impurity=False,filled=True)

with open('arbol.dot') as f:
	dot_graph = f.read()
# print(dot_graph)

# import pydot 
# dotfile = StringIO() 
# export_graphviz(dtreg, out_file=dotfile) 
# pydot.graph_from_dot_data(dotfile.getvalue()).write_png("arbol.png") 

# print(dot_graph)
graph=graphviz.Source(dot_graph)
graph.render ('arbol', view=True, format='png')
# Digraph(graph)
# graph.format = 'png' 
# graph.render('arbol',view=True)
# print('Hola')
# graph = graphviz.Source(export_graphviz(arbol, out_file=None, feature_names=iris.feature_names)) 
# print('Hola')
# png_bytes = graph.pipe(format='png') 
# print('Hola')
# with open('dtree_pipe.png','w') as f: 
#     f.write(png_bytes) 

# from IPython.display import Image 
# Image(png_bytes) 

# system("dot -Tpng .dot -o /arbol.png")
# graphviz.render()
graphviz.render('dot','png','/home/inggarces/Documentos/Python_Dev/Machine_Learning/arbol.dot')

