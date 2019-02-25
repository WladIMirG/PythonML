import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

"""__________________"""
iris = load_iris()
# print(type(iris))
# print("keys:			",iris.keys())
# print("data:			",iris["data"])
# print("target_names:	",iris["target_names"])
# print("target:			",iris["target"])
# print("feature_names:	",iris["feature_names"])

"""__________________"""
Xe, Xt, Ye, Yt = train_test_split(iris["data"], iris["target"])
# print("X_train:			", X_train.shape)
# print("X_test:				", X_test.shape)
# print("Y_train:			", Y_train.shape)
# print("Y_test:				", Y_test.shape)

"""__________________"""
red = MLPClassifier(max_iter=5000, hidden_layer_sizes=10)
print("Entreno:			", red.fit(Xe, Ye))
# alg_svm.decision_function_shape = "ovr"
print("Insertidumbre en escalar:		",red.score(Xt,Yt))
# NOTA1: El numero mayor es el que ofrece menor incertidumbre.
# print("Insertidumbre en porcentaje:		",alg_svm.predict_proba(Xe)[:10])
# NOTA2: La mayor probabilidad es la que ofrece menor incertidumbre.
# print(":		",alg_svm.predict(Xe)[:10])
# print("Prediccion:			",iris.target_names[knn.predict([[5., 3., 5., 1.8]])])
