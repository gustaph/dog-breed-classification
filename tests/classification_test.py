from projectCapstone.classification.dog_breed_classification import DogBreedClassification

clf = DogBreedClassification()

labrador_retriever = 'tests/images/labrador_retriever.jpg'
french_bulldog = 'tests/images/french_bulldog.png'
human = 'tests/images/human.png'
human2 = 'tests/images/human_2.png'
building = 'tests/images/building.jpg'

print(clf.predict(labrador_retriever))
print(clf.predict(french_bulldog))
print(clf.predict(human))
print(clf.predict(human2))
print(clf.predict(building))