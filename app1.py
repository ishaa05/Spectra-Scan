from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten



app1 = Flask(__name__)

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


@app1.route("/")
def home():
    return render_template("parkinsondata.html", prediction_text="")

@app1.route("/predict", methods=['POST'])
def predict():
    # Get the input data from the form
    MDVP_Fo_Hz = float(request.form['MDVP_Fo_Hz'])
    MDVP_Fhi_Hz = float(request.form['MDVP_Fhi_Hz'])
    MDVP_Flo_Hz = float(request.form["MDVP_Flo_Hz"])
    MDVP_Jitter = float(request.form["MDVP_Jitter"])
    MDVP_Jitter_Abs = float(request.form["MDVP_Jitter_Abs"])
    MDVP_RAP = float(request.form["MDVP_RAP"])
    MDVP_PPQ = float(request.form["MDVP_PPQ"])
    Jitter_DDP = float(request.form["Jitter_DDP"])
    MDVP_Shimmer = float(request.form["MDVP_Shimmer"])
    MDVP_Shimmer_dB = float(request.form["MDVP_Shimmer_dB"])
    Shimmer_APQ3 = float(request.form["Shimmer_APQ3"])
    Shimmer_APQ5 = float(request.form["Shimmer_APQ5"])
    MDVP_APQ = float(request.form["MDVP_APQ"])
    Shimmer_DDA = float(request.form["Shimmer_DDA"])
    NHR = float(request.form["NHR"])
    HNR = float(request.form["HNR"])
    RPDE = float(request.form["RPDE"])
    DFA = float(request.form["DFA"])
    spread1 = float(request.form["spread1"])
    spread2 = float(request.form["spread2"])
    D2 = float(request.form["D2"])
    PPE = float(request.form['PPE'])
    print(MDVP_Fo_Hz)
    print(PPE)
    # Prepare input data
    input_data = np.array([[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter, MDVP_Jitter_Abs,
                            MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB,
                            Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR,
                            RPDE, DFA, spread1, spread2, D2, PPE]])
    print(input_data)
    input_array = np.asarray(input_data).reshape(1, -1)
    input_std = scaler.transform(input_array)
    input_pred = model.predict(input_std)
    print("Predicted probability:", input_pred)

    # Convert predicted probabilities to labels
    input_label = np.argmax(input_pred)
    print(input_label)
    if input_label == 0:
        result = "Person does not suffer from Parkinson's Disease"
    else:
        result = "Person suffers from Parkinson's Disease"


    # Render the template with prediction result
    return jsonify({'prediction_text': result})

if __name__ == "_main_":
    app1.run(debug=True)
    # Convert prediction to human-readable text
    

if __name__ == "__main__":
    app1.run(debug=True,port=3000)
