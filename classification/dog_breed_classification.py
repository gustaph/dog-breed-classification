import numpy as np
import cv2
import pickle

from keras.applications.resnet import ResNet50
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing import image
from keras.models import load_model


def path_to_tensor(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)

    return np.expand_dims(x, axis=0)

def img_to_tensor(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)

    return np.expand_dims(x, axis=0)


class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('classification/haarcascade_frontalface_alt.xml')

    def detect(self, img):
        if isinstance(img, str):
            gray_img = cv2.imread(img, 0)

        else:
            img = np.array(img)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray_img)

        return len(faces) > 0


class DogDetector:
    def __init__(self):
        self.resnet = ResNet50(weights='imagenet')

    def _resnet_predict_labels(self, img):
        if isinstance(img, str):
            prep_img = preprocess_input(path_to_tensor(img, (224, 224)))

        else:
            prep_img = img_to_tensor(img)
        
        return np.argmax(self.resnet.predict(prep_img))

    def detect(self, img):
        prediction = self._resnet_predict_labels(img)

        return (prediction <= 268) & (prediction >= 151)


class DogBreedClassification:
    def __init__(self):
        self.model = load_model("classification/VGG19_model")
        self.vgg19 = VGG19(weights='imagenet', include_top=False)
        with open('classification/dog_names.pickle', 'rb') as file_:
            self.dog_names = np.array(pickle.load(file_))

        self.face_detector = FaceDetector()
        self.dog_detector = DogDetector()

    def _extract_VGG19(self, tensor):
        return self.vgg19.predict(preprocess_input(tensor))
        
    def _vgg19_predict_breed(self, img, qnt=4):
        if isinstance(img, str):
            bottleneck_feature = self._extract_VGG19(path_to_tensor(img, (224, 224)))
        
        else:
            bottleneck_feature = self._extract_VGG19(img_to_tensor(img))

        predict_vector = self.model.predict(bottleneck_feature)

        predict_vector = predict_vector.reshape((predict_vector.shape[1],))
        sort_indexes = np.argsort(predict_vector)[::-1][:qnt]
        
        probabilities = predict_vector[sort_indexes]
        dog_names = self.dog_names[sort_indexes]

        clean_names = []
        for name in dog_names:
            clean_names.append(name[name.find('.') + 1:].replace('_', ' ').title())

        predictions = list(zip(probabilities, clean_names))
        return predictions

    def predict(self, img):
        if isinstance(img, str):
            try:
                cv2.imread(img)
            except FileNotFoundError as e:
                raise e
            
        if self.face_detector.detect(img) or self.dog_detector.detect(img):
            return self._vgg19_predict_breed(img)
        
        return None
