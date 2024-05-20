import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

x_train = pd.read_csv(f'housing_x_train_aab323.csv', sep=',', encoding='utf-8')
y_train = pd.read_csv(f'housing_y_train_aab323.csv', sep=',', encoding='utf-8')
x_test = pd.read_csv(f'housing_x_test_aab323.csv', sep=',', encoding='utf-8')

scaler = MinMaxScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
    x_train_scaled, y_train, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train_split, y_train_split, epochs=250, batch_size=32, validation_data=(x_val_split, y_val_split))

val_loss = model.evaluate(x_val_split, y_val_split)
predictions = model.predict(x_test_scaled)

# Flatten the predictions array to 1D (if necessary) and save to CSV
np.savetxt('housing_y_test.csv', predictions.flatten(), delimiter=",", fmt="%g")
print(f"Validation Loss: {val_loss}")

