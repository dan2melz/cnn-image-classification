This project demonstrates how to build and train a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset into 10 categories, including airplanes, cars, cats, and more.

Features

Implements a CNN architecture in PyTorch.

Trains the model on the CIFAR-10 dataset.

Achieves ~71% accuracy on unseen test data.

Includes a script to visualize test predictions vs. ground truth labels.

Dataset

The project uses the CIFAR-10 dataset, a collection of 60,000 32x32 color images across 10 classes:

Airplane

Automobile

Bird

Cat

Deer

Dog

Frog

Horse

Ship

Truck

Getting Started

Prerequisites

Python 3.7+

PyTorch

torchvision

matplotlib

numpy

Install the dependencies using pip:

pip install torch torchvision matplotlib numpy

Running the Code

Clone the repository:

git clone https://github.com/<your-username>/cnn-image-classification.git
cd cnn-image-classification

Run the Python script:

python cnn_image_classification.py

Output

The script will:

Train the model for 10 epochs.

Print the training loss and accuracy.

Display a grid of test images with the predicted and actual labels.

How It Works

Data Loading:

Downloads and normalizes the CIFAR-10 dataset.

Model Architecture:

Uses a CNN with convolution, pooling, and fully connected layers.

Training:

Trains the model using cross-entropy loss and Adam optimizer.

Evaluation:

Measures accuracy on the test set.

Visualization:

Displays some test images and compares predictions with ground truth labels.

Results

Accuracy: 71.17%

Sample Output:

GroundTruth: cat, ship, ship, plane, frog, frog, car, frog

Predicted: cat, ship, ship, plane, cat, frog, car, frog

Next Steps

Experiment with deeper CNN architectures.

Use data augmentation to improve accuracy.

Train on larger datasets for more complex tasks.

Contributing

Feel free to fork this repository and make your own improvements. Pull requests are welcome!


Contact

For questions or suggestions, connect with me on LinkedIn www.linkedin.com/in/daniel-melendez1

