### Table of Contents

1. [Project Definition](#project-definition)
2. [Analysis](#analysis)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Justification](#justification)
6. [Algorithm](#algorithm)
7. [Instructions](#instructions)
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

The dog dataset does not have highly unbalanced classes, so to evaluate the model, I will use the accuracy (correct predictions/total images) as metric.

## Analysis <a name="analysis"></a>

### Data Exploration

In the dataset there are **133** different categories of dogs, divided among the **8351 images** (with different shapes) present in the dataset. Data were separated into *6680 images for training*, *835 for validation* and *836 for testing*. For the human dataset, there are **13233 images**.

#### Dogs Dataset

This dataset contains **133** different categories of dogs, divided among the **8351 images** (with different shapes). These images were separated into training, testing and validation with percentages of ~80%, ~10% and ~10% respectively. Below some information about the data of each split.

|       | Train | Test  | Validation |
|-------|-------|-------|------------|
| count | 133.0 | 133.0 | 133.0      |
| mean  | 50.22 | 6.28  | 6.27       |
| std   | 11.86 | 1.71  | 1.35       |
| min   | 26.0  | 3.0   | 4.0        |
| 25%   | 42.0  | 5.0   | 6.0        |
| 50%   | 50.0  | 6.0   | 6.0        |
| 75%   | 61.0  | 8.0   | 7.0        |
| max   | 77.0  | 10.0  | 9.0        |

We can also analyze the distribution of this data by the images below. In black, the ideal normal distribution. In blue the current distribution. They are very close curves.

<div align="center">
  <img src="https://user-images.githubusercontent.com/54601061/148851869-41d3d8c1-2872-48d2-9709-0fdc08dcd17b.png"/>
</div>

#### Humans Dataset

The analysis of the human dataset follows the same principle as the previous one, except that, in this case, there is no splitting into training, testing and validation.

|       | Value  |
|-------|--------|
| count | 5749.0 |
| mean  | 2.3    |
| std   | 9.0    |
| min   | 1.0    |
| 25%   | 1.0    |
| 50%   | 1.0    |
| 75%   | 2.0    |
| max   | 530.0  |

If we look at the table above, we can already see that the dataset is clearly unbalanced when comparing the mean, minimum and percentiles ​​with the maximum value.

To clarify this, we see in the graph "Top 20 Scores" that there are more than 500 images of a class, while the second most has approximately half that value. This distortion can cause some bias in the results.

<div align="center">
  <img src="https://user-images.githubusercontent.com/54601061/148852201-10533a96-2ad8-4575-80d5-289628e2ba4f.png"/>
</div>

See below for examples of the two datasets:

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

```python
model = Sequential(
    [
        InputLayer(input_shape=(224, 224, 3)),
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(32),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(2),
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dense(133, activation='softmax'),
    ]
)
```

The model flow starts with and input layer with the image dimensions, followed by 3 pairs of Convolutions and MaxPooling, ending with a GlobalAveragePooling layer to produce an output with the spatial average of the feature maps from the last pair Convolution-MaxPooling layer as the confidence of categories, and then it is placed on the fully connected layer. All activation functions are ReLU, except for the last one (softmax), as it is a multi-class problem (for this reason 133 nodes: one for each breed).

The summarized model should look like this...

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

...and then trained in 10 epochs.

All models here were fitted to training data with `batch_size = 20` and compiled with the `RMSprop` optimizer, with `categorical_crossentropy` as the loss funtion, as we are dealing with a multi-class problem.

### Refinement

We implemented the VGG16 architecture by adding more layers. New layers:

```python
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))
```

The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model. I only add a global average pooling layer and a fully connected layer with a softmax.

| Layer                  | Output Shape         |
|------------------------|----------------------|
| GlobalAveragePooling2D | (None, 512)          |
| Dense                  | (None, 133)          |

The same way we did for VGG19:

```python
vgg19_model = Sequential(
    [
        layers.GlobalAveragePooling2D(input_shape=train_VGG19.shape[1:]),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(133, activation='softmax'),
    ]
)
```

The only different layer is Dropout (`layers.Dropout(0.3)`), which will randomly drop 30% of connections, which helps to avoid overfitting.

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

## Justification <a name="justification"></a>

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

## Conclusion <a name="conclusion"></a>

During the project, the steps were: understanding the data, exploring them, to know which metrics to use. With this in mind, the data first went through pre-processing, then being trained and tested in 3 different models, where the most challenging was to create a model from scratch and understand the flow of each layer. After that, the models were evaluated and dog and human detectors were made using pre-trained models. In the end, everything was put together in an image classification script.

### Improvement

According to the overall performance of this approach, there are some aspects that need to be improved/added.

1. *Data Augmentation*: important to increase the variety and variability of images. With the right process, the model can improve its performance.
2. *Mora data*: There are far more than 133 breeds of dogs. Feeding the model with more images from different classes (or more images from existing classes) can also improve the quality of the prediction.

## Acknowledgment <a name="acknowledgment"></a>

Must give credit to [**Udacity**](https://www.udacity.com) for providing the project guidance, templates and datasets.
