# 🐟 Multiclass Fish Image Classification

This project implements deep learning techniques to classify fish species from images. It involves training a CNN model from scratch as well as fine-tuning multiple pre-trained models on a curated fish image dataset. The goal is to achieve high accuracy in recognizing different fish species common in seafood markets.

---

## Dataset and Preprocessing

- Dataset structured in folders by fish species, filtered to include only classes starting with `fish sea_food`.
- Images resized to 224x224 pixels and normalized by rescaling pixel values to [0,1].
- Dataset split into training, validation, and test sets, all loaded with TensorFlow's `ImageDataGenerator`.

---

## Models

### CNN Model from Scratch

- Architecture: 3 convolutional layers with ReLU and max pooling, followed by a dense layer and softmax output for 9 classes.
- Achieved test accuracy: **96.91%**
- Saved as `CNN_Model.keras`

### Transfer Learning with Pre-Trained Models

- Evaluated **InceptionV3**, **EfficientNetB0**, and **MobileNetV2** with frozen convolutional base layers.
- Added global average pooling, dense layers, and softmax output.
- Trained for 5 epochs with Adam optimizer and categorical crossentropy loss.
- Results summary on test set:

| Model          | Test Accuracy |
| -------------- | ------------- |
| MobileNetV2    | 99.66%        |
| InceptionV3    | 98.94%        |
| EfficientNetB0 | 11.00%        |

- MobileNetV2 selected for further analysis and saved as `MobileNet_Model.keras`.

---

## Evaluation

- Used test data to compute classification reports with precision, recall, and F1-score for each class.
- Generated confusion matrices (not shown here) to analyze misclassifications.
- Predictions on individual images demonstrate model confidence scores for class labels.

---

## Usage Highlights

- Image input resized and normalized for prediction.
- Model outputs predicted fish species along with confidence probability.
- Comprehensive label mapping ensures human-readable class names.

---

## Code Structure Summary

- Data loading and preprocessing with `ImageDataGenerator`
- CNN model definition, compilation, training, evaluation
- Transfer learning setup with pre-trained architectures
- Classification reports generation using scikit-learn
- Image-level prediction function for inference
- Model saving for deployment or further use

---

## Author

**Vaisakh Nirupam**
🔗 [LinkedIn](https://www.linkedin.com/in/vaisakh-nirupam)

---
