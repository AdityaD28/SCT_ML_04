# Hand Gesture Recognition using Convolutional Neural Networks

**Short Description:** A deep learning model built with TensorFlow and Keras to classify 10 different hand gestures from the Leap Motion dataset, achieving **99.95% accuracy** on the test set.

---

## Project Overview

This project focuses on developing a robust hand gesture recognition model capable of accurately identifying and classifying various hand gestures from image data. The primary objective is to create a system that can serve as a foundation for intuitive human-computer interaction (HCI) and gesture-based control systems, allowing users to interact with devices without traditional input methods.

The model is built using a Convolutional Neural Network (CNN), a class of deep neural networks well-suited for image analysis tasks.

---

## Dataset

The model was trained on the **Leap Motion Hand Gesture Recognition Dataset**. This is a comprehensive dataset containing images of hand gestures from 10 different subjects (5 male, 5 female), with each subject performing 10 distinct gestures.

* **Total Images:** 20,000
* **Image Properties:** Grayscale, 240x640 pixels (resized to 96x96 for training)
* **Number of Classes:** 10
* **Dataset Source:** [Kaggle: Leap Motion Gesture Recognition](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)

### Gesture Classes

| Class ID | Gesture Name |
| :--- | :--- |
| 01_palm | Palm |
| 02_l | L |
| 03_fist | Fist |
| 04_fist_moved | Fist (Moved) |
| 05_thumb | Thumb |
| 06_index | Index |
| 07_ok | OK |
| 08_palm_moved | Palm (Moved) |
| 09_c | C |
| 10_down | Down |

---

## Methodology

The project follows a standard deep learning workflow:

1.  **Data Loading & Exploration:** The dataset is unzipped and loaded. Image paths and corresponding labels are extracted from the directory structure.
2.  **Image Preprocessing:**
    * Images are loaded in grayscale.
    * Each image is resized to a uniform `96x96` pixels.
    * Pixel values are normalized to a range of `[0, 1]` by dividing by 255.
    * Categorical labels (e.g., '01_palm') are one-hot encoded into numerical vectors.
3.  **Data Splitting:** The dataset is split into three sets to ensure robust evaluation:
    * **Training Set:** 80% (16,000 images)
    * **Validation Set:** 10% (2,000 images)
    * **Test Set:** 10% (2,000 images)
4.  **Model Architecture:** A Convolutional Neural Network (CNN) was designed with three convolutional blocks followed by dense layers for classification.
5.  **Training:** The model was trained for 15 epochs using the Adam optimizer and categorical cross-entropy loss function.
6.  **Evaluation:** The trained model's performance was evaluated on the unseen test set using metrics like accuracy, precision, recall, and a confusion matrix.

---

## Model Architecture

The CNN architecture is implemented using TensorFlow/Keras and consists of the following layers:

Model: "sequential_1"
| Layer (type) | Output Shape | Param # |
| :--- | :--- | :--- |
| **conv2d_3** (Conv2D) | (None, 94, 94, 32) | 320 |
| **max_pooling2d_3** (MaxPooling2D) | (None, 47, 47, 32) | 0 |
| **conv2d_4** (Conv2D) | (None, 45, 45, 64) | 18,496 |
| **max_pooling2d_4** (MaxPooling2D) | (None, 22, 22, 64) | 0 |
| **conv2d_5** (Conv2D) | (None, 20, 20, 128) | 73,856 |
| **max_pooling2d_5** (MaxPooling2D) | (None, 10, 10, 128) | 0 |
| **flatten_1** (Flatten) | (None, 12800) | 0 |
| **dense_2** (Dense) | (None, 512) | 6,554,112 |
| **dropout_1** (Dropout) | (None, 512) | 0 |
| **dense_3** (Dense) | (None, 10) | 5,130 |
| **Total params:** | **6,651,914** | **(25.38 MB)** |

---

## Results

The model achieved excellent performance, demonstrating its effectiveness in classifying hand gestures with high precision.

* **Test Accuracy:** **99.95%**
* **Test Loss:** 0.0006

The training process showed fast convergence with validation accuracy reaching nearly 100% within a few epochs.

### Classification Report

The detailed classification report confirms the model's strong performance across all 10 gesture classes.

|                   | precision | recall | f1-score | support |
| :---------------- | :-------: | :----: | :------: | :-----: |
| **01_palm** |   1.00    |  1.00  |   1.00   |   200   |
| **02_l** |   1.00    |  1.00  |   1.00   |   200   |
| **03_fist** |   1.00    |  1.00  |   1.00   |   200   |
| **04_fist_moved** |   1.00    |  1.00  |   1.00   |   200   |
| **05_thumb** |   1.00    |  1.00  |   1.00   |   200   |
| **06_index** |   1.00    |  1.00  |   1.00   |   200   |
| **07_ok** |   1.00    |  1.00  |   1.00   |   200   |
| **08_palm_moved** |   1.00    |  1.00  |   1.00   |   200   |
| **09_c** |   1.00    |  0.99  |   1.00   |   200   |
| **10_down** |   1.00    |  1.00  |   1.00   |   200   |
|                   |           |        |          |         |
| **accuracy** |           |        | **1.00** | **2000** |
| **macro avg** |   1.00    |  1.00  |   1.00   |   2000  |
| **weighted avg** |   1.00    |  1.00  |   1.00   |   2000  |


---

### 2. Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/hand-gesture-recognition.git](https://github.com/your-username/hand-gesture-recognition.git)
    cd hand-gesture-recognition
    ```

2.  **Download the Dataset:**
    * Download the `leapgestrecog` dataset from [Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog).
    * Unzip the file and place the `leapgestrecog` folder inside the project's root directory. The final path should look like `.../hand-gesture-recognition/leapgestrecog/`.

### 3. Running the Code

1.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  Open the `Hand_Gesture_Recognition.ipynb` file.
3.  Execute the cells in order to load the data, build the model, train it, and evaluate its performance. The final cells allow for testing with your own uploaded images.

---

   ## Developed By

* **Aditya Dasappanavar**
* **GitHub:** [AdityaD28](https://github.com/AdityaD28)
* **LinkedIn:** [adityadasappanavar](https://www.linkedin.com/in/adityadasappanavar/)
