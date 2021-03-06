from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston					#Import la data
from sklearn.model_selection import train_test_split		#la data en Xs y Ys de entrenamiento y de test
from sklearn.linear_model import LinearRegression, Ridge

boston = load_boston()


print("keys:			",boston.keys())
print("data:			",boston.data.shape)
# print("target_names:	",boston["target_names"])
print("target:			",boston.target.shape)
print("feature_names:	",boston.feature_names.shape)

"""__________________"""
X_train, X_test, Y_train, Y_test = train_test_split(boston.data, boston.target)
print("X_train:			", X_train.shape)
print("X_test:				", X_test.shape)
print("Y_train:			", Y_train.shape)
print("Y_test:				", Y_test.shape)

knn = KNeighborsRegressor(n_neighbors=5)
# knn.fit(X_train, Y_train)
print("Entreno:			",knn.fit(X_train, Y_train))
print("Comprobacion:		",knn.score(X_test, Y_test))
# print("Prediccion:			",boston.target_names[knn.predict([[5., 3., 5., 1.8]])])

del knn

rl = LinearRegression()
# rl.fit(X_train, Y_train)
print("Entreno:			",rl.fit(X_train, Y_train))
print("Comprobacion:		",rl.score(X_test, Y_test))
# print("Prediccion:			",boston.target_names[knn.predict([[5., 3., 5., 1.8]])])

del rl

ridge = Ridge()
# ridge.fit(X_train, Y_train)
print("Entreno:			",ridge.fit(X_train, Y_train))
print("Comprobacion:		",ridge.score(X_test, Y_test))
# print("Prediccion:			",boston.target_names[knn.predict([[5., 3., 5., 1.8]])])

del ridge
