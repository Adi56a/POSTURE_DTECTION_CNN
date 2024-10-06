import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('movement_classifier.h5')

# Function to preprocess the uploaded image
def load_and_preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img_array

# Streamlit app
def main():
    st.set_page_config(page_title="Movement Classification", layout="centered")
    
    # Add a header
    st.title("🏃 Movement Classification App")
    st.markdown("Upload an image of a person to classify their posture as **Standing** or **Sitting**.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and display the image with reduced size
        img = image.load_img(uploaded_file, target_size=(224, 224))
        
        # Create a container to hold the image and its prediction
        with st.container():
            st.subheader("Uploaded Image")
            st.image(img, caption='Uploaded Image', width=150)  # Show image at a fixed width
            
            # Prepare the image for prediction
            img_array = load_and_preprocess_image(img)

            # Predict the class
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)

            # Display the prediction with some styling
            class_labels = ['Standing', 'Sitting']
            st.markdown(f"<h3 style='color: blue;'>The predicted class is: {class_labels[predicted_class]}</h3>", unsafe_allow_html=True)
            
        
            st.balloons()  

           

if __name__ == "__main__":
    main()
