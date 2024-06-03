import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten

# Load the dataset
data = pd.read_csv("Parkinsson disease.csv")

# Split features and labels
X = data.drop(columns=['status', 'name'], axis=1)
Y = data['status']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2, stratify=Y)

# Standardize the data
scaler = StandardScaler()
X_train_fit = scaler.fit_transform(X_train)
X_test_fit = scaler.transform(X_test)

# Build the neural network model
model = keras.Sequential([
    Flatten(input_shape=(22,)),
    Dense(18, activation='relu'),
    BatchNormalization(),
    Dense(12, activation='relu'),
    BatchNormalization(),
    Dense(7, activation='relu'),
    BatchNormalization(),
    Dense(2, activation='sigmoid')
])

# Compile the model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_fit, Y_train, batch_size=16, validation_split=0.30, epochs=15)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_fit, Y_test)
print("Accuracy:", accuracy)

# Save the model using TensorFlow's built-in method
model.save("parkinsondiseasedetectionusingneuralnetworks.h5")

# Make predictions
input_data = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
input_array = np.asarray(input_data).reshape(1, -1)
input_std = scaler.transform(input_array)
input_pred = model.predict(input_std)
print("Predicted probability:", input_pred)

# Convert predicted probabilities to labels
input_label = np.argmax(input_pred)
if input_label == 0:
    print("Person does not suffer from Parkinson's Disease")
else:
    print("Person suffers from Parkinson's Disease")
