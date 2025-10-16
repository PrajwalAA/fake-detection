import streamlit as st
from gradio_client import Client
from PIL import Image
import io
import time

# ------------------ Configuration ------------------
SPACE_URL = "https://stabilityai-stable-diffusion-2.hf.space/"  # ✅ use a stable working space
client = Client(SPACE_URL)

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="AI Image Generator", layout="centered")
st.title("🎨 AI Poster Generator")
st.caption("Generate images using Stable Diffusion via Gradio Client")

prompt = st.text_area(
    "📝 Prompt",
    "A sleek futuristic poster with the text 'DATA SCIENCES' glowing in neon blue, modern typography, digital grid background, professional style."
)

# Seed + style controls
col1, col2 = st.columns(2)
with col1:
    num_images = st.slider("Number of Images", 1, 4, 1)
with col2:
    seed = st.text_input("Seed (optional)", "")

generate_btn = st.button("🚀 Generate Image", type="primary", use_container_width=True)

# ------------------ Image Generation ------------------
if generate_btn:
    if not prompt.strip():
        st.error("Please enter a prompt before generating.")
    else:
        with st.spinner("⏳ Generating image... please wait..."):
            try:
                # ✅ Call with 1 argument (prompt)
                result = client.predict(
                    prompt,
                    fn_index=0  # most Gradio image generation apps have fn_index=0
                )

                # Handle results (can be bytes or file path)
                images = result if isinstance(result, list) else [result]

                st.success("✅ Image generation complete!")

                # Display results
                cols = st.columns(min(len(images), 4))
                for i, img_data in enumerate(images):
                    with cols[i % len(cols)]:
                        if isinstance(img_data, str) and img_data.startswith("http"):
                            st.image(img_data, caption=f"Image {i+1}")
                        else:
                            image = Image.open(io.BytesIO(img_data)) if isinstance(img_data, bytes) else Image.open(img_data)
                            st.image(image, caption=f"Image {i+1}")
            except Exception as e:
                st.error(f"⚠️ Error while generating image: {e}")
