# Spectra-Scan
### Disease  detection using deep learning
Spectra Scan is a web application that uses clinical records and mammographic images for the early detection of Breast Cancer and Parkinson’s Disease. The goal is to enable remote monitoring and preliminary analysis, providing users with valuable health insights.
## Literature Survey
The project builds on recent advancements in disease detection, focusing on Breast Cancer and Parkinson’s Disease. Various studies highlight the efficacy of different machine learning algorithms, including CNNs and ensemble methods, in medical diagnostics.
## Design
The system consists of the following components:
* **User Interface:** For user interaction and data input.
* **Authentication:** User verification through Google Login.
* **Image Processing:** Preprocessing and analysis of mammographic images.
* **Disease Analysis:** Machine learning models for disease prediction.
## Algorithms Used
* **Stacking:** Combines predictions from multiple models for improved accuracy.
* **Neural Networks:** Shallow feedforward neural network for binary classification.
* **VGG19:** Convolutional Neural Network for image classification.
## Implementation
* **Landing Page:** User-friendly interface for accessing the application.
* **Login Page:** Secure user authentication.
* **Disease Prediction:** Options for predicting Breast Cancer and Parkinson’s Disease.
* **Multilingual Support:** Interfaces available in multiple languages.
## Results
* **VGG19 Model:** High accuracy in breast cancer detection using mammographic images.
* **Neural Networks:** Effective in analyzing clinical records for Parkinson’s disease.
* **Stacking Model:** Enhanced performance by combining multiple base models.
## Technologies Used
### Frontend
* **HTML/CSS/JavaScript:** For creating the front-end user interface.
* **Bootstrap:** For responsive design.
### Backend
* **Flask:** Web framework for developing the web application.
* **TensorFlow:** Deep learning framework used for model building and training.
* **Keras:** High-level neural networks API, running on top of TensorFlow.
* **Scikit-learn:** Machine learning library for building the stacking model.
* **Google Firebase:** For authentication and real-time database services.
## Conclusions
Spectra Scan demonstrates the potential of deep learning in medical diagnostics, offering a reliable solution for the early detection of Breast Cancer and Parkinson’s Disease. The project contributes to the advancement of healthcare by providing accessible and accurate diagnostic tools.





