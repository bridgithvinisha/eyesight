import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
from pyngrok import ngrok
import time

st.set_page_config(page_title="Eye Disease Detection", layout="centered")
st.title("👁️ Eye Disease Detection with EfficientNet")
st.write("Upload a retinal image to detect eye diseases like **Cataract**, **Glaucoma**, or **Diabetic Retinopathy**.")

@st.cache_resource
def load_trained_model():
    return load_model("model.h5")

model = load_trained_model()
class_names = ['cataract', 'diabetic retinopathy', 'glaucoma', 'normal']

uploaded_file = st.file_uploader("Upload an eye image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Eye Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_name = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100

    if class_name == 'normal':
        st.success(f"✅ The eye is **NOT infected**. (Confidence: {confidence:.2f}%)")
    else:
        st.error(f"⚠ The eye is **INFECTED with {class_name.upper()}**. (Confidence: {confidence:.2f}%)")

# Start ngrok tunnel
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")  # Replace with your own token
streamlit_cmd = "streamlit run app.py &>/content/logs.txt &"
os.system(streamlit_cmd)
time.sleep(5)
public_url = ngrok.connect(8501, "http")
st.write(f"🚀 App is live at: {public_url}")
