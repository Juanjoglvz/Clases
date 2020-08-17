from keras.datasets import boston_housing
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

(x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split=0.2)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Definimos el modelo
x = Input(shape=[x_train.shape[1]])
dense1 = Dense(x_train.shape[1] * 3, activation="tanh")(x)
dense2 = Dense(x_train.shape[1] * 5, activation="tanh")(dense1)
dense3 = Dense(x_train.shape[1] * 3, activation="tanh")(dense2)
output = Dense(1)(dense3)

model = Model(inputs=x, outputs=output)

model.compile(optimizer="SGD", loss="mse")

model.fit(x_train, y_train, epochs=100, batch_size=8)

model.evaluate(x_test, y_test)

new_data = np.array([1.23247, 0., 8.14, 0., 0.538, 6.142, 91.7, 3.9769, 4., 307., 21., 396.9, 18.72])

new_data = np.reshape(new_data, (1, 13))

y_pred = model.predict(new_data)

print(y_pred)
