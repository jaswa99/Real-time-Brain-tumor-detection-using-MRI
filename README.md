Brain Tumor Detection 
This project implements a hybrid Convolutional Neural Network (CNN) using Keras and TensorFlow to classify brain MRI images. The model is trained to perform binary classification, distinguishing between images that show a brain tumor ("yes") and those that do not ("no").

📋 Project Overview
The goal of this project is to build an accurate and efficient model for detecting brain tumors from MRI scans. It follows a complete machine learning pipeline: data loading, image preprocessing, model building, training, evaluation, and a final prediction script.

⚙️ Workflow
The project is structured as follows:

Data Loading: Images are loaded from a dataset directory which is expected to have two subfolders:

/yes/ (containing images with tumors)

/no/ (containing images without tumors)

Image Preprocessing:

All images are read using OpenCV (cv2).

Regardless of their original size, all images are resized to a uniform (224, 224) pixels.

Images are converted to NumPy arrays, and labels are assigned (1 for "yes", 0 for "no").

Data Splitting:

The complete dataset (images and labels) is split into training and testing sets using sklearn.model_selection.train_test_split.

The split is 80% for training and 20% for testing.

Model Building:

A Sequential CNN model is built using Keras.

The architecture consists of two convolutional layers followed by max-pooling, a flatten layer, and two dense layers.

The final layer uses a sigmoid activation function, making it suitable for binary (yes/no) classification.

Training & Evaluation:

The model is compiled with the adam optimizer and binary_crossentropy loss function.

It is trained for 10 epochs using the training data and validated against the test data.

Training and validation accuracy/loss are plotted to visualize performance and check for overfitting.

Model Saving & Prediction:

The fully trained model is saved as braintumor.h5.

A separate script is included to load this saved model, process a new input image, and predict whether it contains a tumor.

🧠 Model Architecture
The CNN is built with the following layers:

Conv2D(filters=32, kernel_size=(3,3), activation='relu')

MaxPooling2D(pool_size=(2,2))

Conv2D(filters=64, kernel_size=(3,3), activation='relu')

MaxPooling2D(pool_size=(2,2))

Flatten()

Dense(units=64, activation='relu')

Dense(units=1, activation='sigmoid')

📊 Results
The model achieves high accuracy after 10 epochs. Based on the notebook's output, the performance is:

Training Accuracy: ~99.22%

Validation Accuracy: ~96.08%

The training history is visualized in the plots for "Training loss vs Validation loss" and "Training accuracy vs Validation accuracy".

🚀 How to Run
Clone the repository.

Install dependencies:

Bash

pip install tensorflow keras opencv-python numpy pandas matplotlib scikit-learn
Prepare the Dataset:

Create a main data folder (e.g., brain_tumor_dataset/).

Inside this folder, create two subfolders: yes/ and no/.

Place all tumor-positive images in the yes/ folder and all tumor-negative images in the no/ folder.

Update Paths:

In the notebook, change the main_path variable (Cell 3) to point to your main data folder.

In the prediction cell (Cell 16), update the path to your test image.

Run the Notebook:

Execute the cells sequentially to load the data, build, train, and save the model.

The final cells can be used to test the trained model on new images.
