import streamlit as st
from PIL import Image
from projectCapstone.classification.dog_breed_classification import DogBreedClassification

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():

    with st.spinner("Loading the model..."):
        model = DogBreedClassification()
    return model

st.set_page_config(
    page_title="Dog Breed Classifier",
    page_icon=":dog:",
    layout="wide",
)

st.sidebar.title("Project Overview")
st.sidebar.info("This project consists of classifying images - people or dogs - for\
                the recognition of dog breeds (in the case of a person image, the output\
                will be which breed it most resembles).")

st.sidebar.write("---")

st.sidebar.info("**[Project link (GitHub)](https://github.com/gustaph/dog-breed-classification)**")

st.title("Dog Breed Classifier")
st.write("---")

model = load_model()
file = st.file_uploader('Please, upload and human or dog image.', type=["png", "jpg"])

if file is not None:
    image = Image.open(file).convert('RGB')
    height, width = image.size

    with st.spinner("Please wait a moment..."):
        result = model.predict(image)

    if not result:
        st.error("The image must be of a human or a dog")

    else:
        st.success(f"Hey, it look's like a {result[0][1]}!")
        center_image = st.columns(3)
        center_image[1].image(image)

        cols = st.columns(len(result) + 2)

        for index, (prob, class_) in enumerate(result, start=1):
            prob = round(prob * 100, 2)
            cols[index].metric(label=class_, value=f"{prob}%")

        st.info(f"Only the three highest probabilities are shown.")