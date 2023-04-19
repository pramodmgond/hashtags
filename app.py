import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Define a function to predict the top K imagenet labels for an input image
def predict_labels(image_path, K=10):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Use the ResNet50 model to predict the image's class probabilities
    preds = model.predict(x)

    # Decode the predictions into a list of imagenet labels and their probabilities
    imagenet_labels = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt", header=None)[0]
    top_preds = np.argsort(-preds, axis=1)[0][:K]
    labels = [imagenet_labels[idx] for idx in top_preds]
    probs = [preds[0][idx] for idx in top_preds]
    return labels, probs

# Define a function to generate hashtags for an input image
def generate_hashtags(image_path, K=10):
    # Use the predict_labels function to get the top imagenet labels for the image
    labels, _ = predict_labels(image_path, K)

    # Convert the imagenet labels to hashtags and return the top K hashtags
    hashtags = [f"#{label.replace(' ', '')}" for label in labels]
    return hashtags[:K]

# Define the Streamlit app
def app():
    st.title("Image Hashtag Generator")

    #target_size = (160, 160)
    # Upload the image file
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # If an image is uploaded, generate hashtags for it
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        #image = image.resize(target_size)
        st.image(image, caption="Uploaded image", use_column_width=True)

        # Generate hashtags for the image
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        hashtags = generate_hashtags(image_path, K=10)

        # Display the hashtags
        st.subheader("Generated Hashtags")
        for hashtag in hashtags:
            st.write(hashtag)

# Run the Streamlit app
if __name__ == "__main__":
    app()
