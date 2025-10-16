import streamlit as st
import time
from gradio_client import Client

# --- Streamlit Config ---
st.set_page_config(
    page_title="Image Generation Studio",
    page_icon="🎨",
    layout="wide"
)

# --- Cached Gradio Client ---
@st.cache_resource
def get_client():
    try:
        return Client("https://c452d26a281d4a0cde.gradio.live/")
    except Exception as e:
        st.error(f"Failed to connect to Gradio client: {e}")
        return None

# --- Main Function ---
def main():
    st.title("🎨 Image Generation Studio")
    st.markdown("Generate high-quality AI images instantly")

    client = get_client()
    if not client:
        st.stop()

    # --- Inputs ---
    prompt = st.text_area("Prompt", "A beautiful futuristic cityscape at sunset", height=100)
    negative_prompt = st.text_area("Negative Prompt", "low quality, blurry", height=50)

    style = st.selectbox("Style", ["realistic", "cinematic", "digital art", "anime"])
    aspect_ratio = st.selectbox("Aspect Ratio", ["1:1", "16:9", "9:16", "4:3", "3:4"])
    performance = st.radio("Performance", ["Speed", "Quality", "Extreme Speed"])

    col1, col2 = st.columns(2)
    with col1:
        image_number = st.slider("Number of Images", 1, 4, 1)
    with col2:
        use_random_seed = st.checkbox("Random Seed", True)

    if not use_random_seed:
        seed = st.text_input("Seed", "12345")
    else:
        seed = str(time.time()).replace('.', '')[:10]

    generate_button = st.button("🚀 Generate Image", type="primary")

    if generate_button:
        if not prompt.strip():
            st.error("Please enter a valid prompt.")
            st.stop()

        st.info("⏳ Generating image, please wait...")

        try:
            # This is a generic predict() call since the API structure isn't public.
            # fn_index=0 is usually the main image generation endpoint.
            result = client.predict(
                prompt,      # Main input
                negative_prompt,
                style,
                performance,
                aspect_ratio,
                seed,
                image_number,
                fn_index=0   # You may adjust if actual endpoint differs
            )

            st.success("✅ Image generated successfully!")

            # Handle and display result
            if isinstance(result, list):
                for i, img in enumerate(result):
                    st.image(img, caption=f"Generated Image {i+1}")
            else:
                st.image(result, caption="Generated Image")

        except Exception as e:
            st.error(f"⚠️ Error while generating image: {e}")

# --- Run App ---
if __name__ == "__main__":
    main()
