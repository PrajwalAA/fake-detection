import streamlit as st
from gradio_client import Client
from PIL import Image
import io

# Page settings
st.set_page_config(page_title="🎨 AI Image Generator", layout="centered")

st.title("🧠 Data Sciences Poster Generator")

# Prompt input
prompt = st.text_area(
    "📝 Enter your image prompt:",
    "A sleek, futuristic poster with 'DATA SCIENCES' text in the center, glowing blue neon, digital background, professional and modern style."
)

# Model choice (example: Stability AI or any Hugging Face model)
model_name = "stabilityai/stable-diffusion-2"  # You can change this if needed
client = Client(model_name)

# Button to generate image
if st.button("🚀 Generate Image"):
    with st.spinner("⏳ Generating image... please wait..."):
        try:
            # ✅ Correct function call — only ONE argument (the prompt)
            result = client.predict(prompt)

            # Convert result to image (depending on API, it can be bytes or URL)
            if isinstance(result, str) and result.startswith("http"):
                st.image(result, caption="Generated Image", use_column_width=True)
            elif isinstance(result, bytes):
                img = Image.open(io.BytesIO(result))
                st.image(img, caption="Generated Image", use_column_width=True)
            else:
                st.warning("⚠️ Unexpected result format received.")
        except Exception as e:
            st.error(f"⚠️ Error while generating image: {e}")
