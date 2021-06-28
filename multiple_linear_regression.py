# Regresion Linear Multiple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargando el conjunto de datos
df = pd.read_csv('50_Startups.csv')
X = df.iloc[:,:-1].values
y = df.iloc[:,4].values

# Transformar las variables categoricas
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Evitar la trampa de las variables ficticias, eliminar 1 de las dummy
X = X[:, 1:]

# Dividir el dataset en conjunto de train y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Ajustar el modelo de regresión lineal multiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regresion = LinearRegression()
regresion.fit(X_train, y_train)

# Prediccion de los resultados en el conjunto de test
y_pred = regresion.predict(X_test)

# Construir el modelo optimo para RLM utilizando la eliminacion hacia atras
import statsmodels.api as sm
# y = b0+b1*x1+b2*x2+...+bn*xn
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
# Significancia
SL = 0.05
# Matriz de caracteristicas optimas
X_opt = X[:, [0,1,2,3,4,5]]
regresion_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regresion_OLS.summary()

# Matriz de caracteristicas optimas
X_opt = X[:, [0,1,3,4,5]]
regresion_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regresion_OLS.summary()

# Matriz de caracteristicas optimas
X_opt = X[:, [0,3,4,5]]
regresion_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regresion_OLS.summary()

# Matriz de caracteristicas optimas
X_opt = X[:, [0,3,5]]
regresion_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regresion_OLS.summary()

# Matriz de caracteristicas optimas
X_opt = X[:, [0,3]]
regresion_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regresion_OLS.summary()



