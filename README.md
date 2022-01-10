### Table of Contents

1. [Project Definition](#project-definition)
2. [Analysis](#analysis)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Algorithm](#algorithm)
6. [Instructions](#instructions)
7. [Files](#files)
8. [Conclusion](#conclusion)
9. [Acknowledgment](#acknowledgment)

## Project Definition <a name="project-definition"></a>

### Project Overview

![results_1](https://user-images.githubusercontent.com/54601061/148759825-26e5f795-b6e3-45d0-bdc9-40191034f3f4.png)
![results_2](https://user-images.githubusercontent.com/54601061/148759831-9b7d7a76-4131-4498-a769-b3463b991a57.png)

This project consists of **classifying images** - people or dogs - for the recognition of *dog breeds* (in the case of a person image, the output will be which breed it most resembles). When it comes to neural networks and images, the first thing that comes to mind are **Convolutional Neural Networks (CNNs)**, and it is through them that classification takes place.

Tests were made for networks created from scratch and transfer learning, using the [VGG16 algorithm](https://keras.io/api/applications/vgg/), from Keras. The databases were made available by Udacity and consist of a folder for dog breeds and another for images of people. The final algorithm runs with [VGG19 model](https://keras.io/api/applications/vgg/).

### Problem Statement

Here, we have a computer vision problem that uses a Convolutional Neural Network (CNN) for an image classification task: given an image of a human or dog, the CNN (from scratch/VGG16/VGG19) will designate the labels (breeds of dogs) that the most resemble the content of the image.

### Metrics

To evaluate the model, we will use a comparison using precision (percentage of correct answers for the model in unseen data).

## Analysis <a name="analysis"></a>

### Data Exploration

In the dataset there are **133** different categories of dogs, divided among the **8351 images** (with different shapes) present in the dataset. Data were separated into *6680 images for training*, *835 for validation* and *836 for testing*. For the human dataset, there are **13233 images**. See below for examples of the two datasets:

![image](https://user-images.githubusercontent.com/54601061/148783146-331f9eed-252d-43bc-b069-e68fd5aa99fd.png)

## Methodology <a name="methodology"></a>

### Data Preprocessing

When using TensorFlow as backend, Keras requise a 4D array as input, with shape `(samples, rows, columns, channels)`. For this, we use the function `path_to_tensor` that loads the image and resizes it to aa square image that is  pixels. Next, the image is converted to an array, which is then resized to a 4D tensor.

We need to rescale too:
```python
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32') / 255
valid_tensors = paths_to_tensor(valid_files).astype('float32') / 255
test_tensors = paths_to_tensor(test_files).astype('float32') / 255
```
With that the data is ready to be used in the model.

### Implementation

For the implementation, two CNNs were built: one from scratch and two others using transfer learning (VGG16/VGG19). Let's take a look at the architecture of the first model.

| Layer                  | Output Shape         |
|------------------------|----------------------|
| InputLayer             | (None, 224, 224, 3)  |
| Conv2D                 | (None, 224, 224, 16) |
| MaxPooling2D           | (None, 112, 112, 16) |
| Conv2D                 | (None, 112, 112, 32) |
| MaxPooling2D           | (None, 3, 3, 32)     |
| Conv2D                 | (None, 3, 3, 64)     |
| MaxPooling2D           | (None, 1, 1, 64)     |
| GlobalAveragePooling2D | (None, 64)           |
| Dense                  | (None, 512)          |
| Dense                  | (None, 133)          |

All models were compiled with the RMSprop optimizer, with `categorical_crossentropy` as the loss funtion, as we are dealing with a multi-class problem.

### Refinement

We implemented the VGG16 architecture by adding more layers. New layers:

| Layer                  | Output Shape         |
|------------------------|----------------------|
| GlobalAveragePooling2D | (None, 512)          |
| Dense                  | (None, 133)          |

The same way we did for VGG19:

| Layer                  | Output Shape         |
|------------------------|----------------------|
| GlobalAveragePooling2D | (None, 512)          |
| Dense                  | (None, 265)          |
| Dense                  | (None, 512)          |
| Dropout                | (None, 512)          |
| Dense                  | (None, 133)          |

## Results <a name="results"></a>

| Model                             | Accuracy    |
|-----------------------------------|-------------|
| Refined Transfer Learning (VGG19) | **71.29%**  |
| Refined Transfer Learning (VGG16) | 47.97%      |
| From Scratch                      | 13.28%      |

Using the accuracy metric as a comparison, it can be seen that the transfer learning models performed better tests, as these models were studied and designed for this type of classification (more elaborate layers compared to the first model).

## Algorithm <a name="algorithm"></a>

### Face Detector

I use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades)

```python
face_cascade = cv2.CascadeClassifier('haarcascades.xml')
faces = face_cascade.detectMultiScale(gray_scale_image)
```

### Dog Detector

I use a [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model, pre-trained on [ImageNet](http://www.image-net.org/), to detect dogs in images. ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).

### Dog Breed Classifier

For this step, first check if there is a dog or a person in the image. If there is, the image will be processed and sent as input to the refined VGG19 model, which will return the label to which that image belongs.

## Instructions <a name="instructions"></a>

1. Create and activate a virtual enviroment
    ```
    pip -m venv capstone
    capstone\Scripts\activate.bat
    ```

2. Clone the project
    ```
    git clone https://github.com/gustaph/dog-breed-classification.git
    ```

3. Install the dependencies
   ```
    pip install -r projectcapstone/requirements.txt
   ```

4. Run the server
    ```
    cd projectcapstone
    streamlit run app.py
    ```

    If the window doesn't open, copy the address that will appear in the console and paste it into your browser

## Files <a name="files"></a>



## Conclusion <a name="conclusion"></a>

During the project, the steps were: understanding the data, exploring them, to know which metrics to use. With this in mind, the data first went through pre-processing, then being trained and tested in 3 different models, where the most challenging was to create a model from scratch and understand the flow of each layer. After that, the models were evaluated and dog and human detectors were made using pre-trained models. In the end, everything was put together in an image classification script.

### Improvement

According to the overall performance of this approach, there are some aspects that need to be improved/added.

1. *Data Augmentation*: important to increase the variety and variability of images. With the right process, the model can improve its performance.
2. *Mora data*: There are far more than 133 breeds of dogs. Feeding the model with more images from different classes (or more images from existing classes) can also improve the quality of the prediction.

## Acknowledgment <a name="acknowledgment"></a>

Must give credit to [**Udacity**](https://www.udacity.com) for providing the project guidance, templates and datasets.
