import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Cargar datos
data = pd.read_csv("source.csv")

# Quitamos columnas de región y canal porque no nos interesan
data = data.drop(["Channel", "Region"], axis=1)

# Visualizamos un poco los datos primero
for key, value in data.iteritems():
    print(key, value.describe())
    print()

# X-> Datos, Y-> Cosa que queremos predecir
X = data[["Fresh", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]]
Y = data[["Milk"]]

# Split en datos de entrenamiento y test
# Es importante separar los datos porque necesitamos algunos datos en los que
# no se haya aprendido para sacar conclusiones correctas
# (Se trata de emular datos "nuevos", pero en los que sabemos el resultado
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Entrenamiento
model = LinearRegression()
model.fit(X_train, y_train)

# Vemos los coeficientes de la recta
print(model.coef_, model.intercept_)

# Predecimos con datos de test para ver qué tal
y_pred = model.predict(X_test)

print(mean_squared_error(y_test, y_pred))


# Predicción con nuevos datos
# Leer archivo
new_data = np.array([[123131,1231231,654,3342,123]])

print(model.predict(new_data))


# Modelo de regresión lineal
# Milk = a * Fresh + b * Delicassen + c * Grocery + d * Frozen + e * DP + f(intercept)


