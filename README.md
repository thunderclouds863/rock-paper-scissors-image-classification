# Rock Paper Scissors Image Classification

A convolutional neural network that classifies photos of hand gestures as rock, paper or scissors. Built with TensorFlow/Keras in Google Colab as the final project of Dicoding's *Belajar Machine Learning untuk Pemula* (Machine Learning for Beginners) course.

**Result:** 94.7% accuracy on the validation set (loss 0.18).

## Dataset

[Rock Paper Scissors dataset](https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip) from Dicoding: 2,188 images in three folders (`rock`, `paper`, `scissors`). The notebook downloads it, so it is not stored in this repository.

| Split | Images |
|---|---|
| Training (60%) | 1,314 |
| Validation (40%) | 874 |

## Approach

1. **Preprocessing and augmentation** with `ImageDataGenerator`: rescale to 0–1, rotation up to 20°, horizontal flip, shear 0.2. Images resized to 150×150.
2. **Model:** three Conv2D + MaxPooling blocks (32, 64 and 128 filters), Flatten, Dropout 0.5, Dense 512 (ReLU) and a 3-class softmax output.
3. **Training:** SGD optimizer with Nesterov momentum, categorical cross-entropy loss, 20 epochs of 60 steps, with an `EarlyStopping` callback that restores the best weights.
4. **Evaluation** on the validation set, then a prediction on an uploaded photo.

## How to run

Open `ML1.ipynb` in Google Colab and run all cells. The last cell asks you to upload an image to classify; it uses `google.colab.files`, so it only works in Colab.

## Notes

- The validation generator uses the same augmentation as the training generator.
- The prediction cell checks `if classes[0, 0]`, which is true for any non-zero probability. Using `np.argmax(classes)` would pick the class with the highest probability.

## Tools

Python, TensorFlow/Keras, NumPy, Matplotlib, Google Colab.
