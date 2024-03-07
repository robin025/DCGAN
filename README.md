# DCGAN

Loss Curve for DCGAN on Fashion MNIST
![Loss Curve for DCGAN on Fashion MNIST](https://github.com/robin025/DCGAN/blob/main/results/output_FashionMNIST_curve.png)

Generated Images for DCGAN on Fashion MNIST
![Generated Images for DCGAN on Fashion MNIST](https://github.com/robin025/DCGAN/blob/main/results/output_fashionMNIST.png)

Loss Curve for DCGAN on MNIST
![Loss Curve for DCGAN on MNIST](https://github.com/robin025/DCGAN/blob/main/results/output_MNIST_curve.png)

Generated Images for DCGAN on MNIST
![Generated Images for DCGAN on MNIST](https://github.com/robin025/DCGAN/blob/main/results/output_MNIST.png)

## Introduction
This is an implementation of DCGAN on Fashion MNIST and MNIST datasets using PyTorch.

## Dataset
The Fashion MNIST dataset is used for training and testing the model. It consists of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images. The MNIST dataset is also used for training and testing the model. It consists of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.

## Model
The model consists of a generator and a discriminator. The generator takes a random noise vector as input and generates a 28x28 grayscale image. The discriminator takes an image as input and outputs a scalar value representing the probability that the image is real (as opposed to generated). The generator and discriminator are trained simultaneously using the adversarial loss.


