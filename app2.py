import streamlit as st
from gradio_client import Client
import time

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="🎨 Image Generation Studio",
    page_icon="🖼️",
    layout="wide"
)

# --- Cached Gradio Client ---
@st.cache_resource
def get_client():
    try:
        return Client("https://faf7c5ec7c92e2ba15.gradio.live")
    except Exception as e:
        st.error(f"❌ Failed to connect to Gradio client: {e}")
        return None


# --- Auto detect API structure ---
def get_api_structure(client):
    try:
        api_info = client.view_api()
        st.sidebar.success("✅ Connected to Gradio API")
        st.sidebar.write("**Detected API Structure:**")
        st.sidebar.code(api_info)
        return api_info
    except Exception as e:
        st.sidebar.error(f"⚠️ Failed to fetch API structure: {e}")
        return None


# --- Main Function ---
def main():
    st.title("🎨 Image Generation Studio")
    st.caption("Generate high-quality AI images directly via connected Gradio model endpoint.")

    client = get_client()
    if not client:
        st.stop()

    api_info = get_api_structure(client)

    # --- Basic Prompt ---
    prompt = st.text_area("📝 Prompt", "A beautiful futuristic cityscape at sunset", height=100)
    generate_button = st.button("🚀 Generate Image", type="primary")

    if generate_button:
        if not prompt.strip():
            st.error("Please enter a prompt before generating.")
            st.stop()

        st.info("⏳ Generating image... please wait...")

        try:
            # --- Try single-input predict call first ---
            result = None
            try:
                result = client.predict(prompt, fn_index=0)
            except TypeError as e:
                # If API expects multiple arguments, detect how many
                msg = str(e)
                if "got" in msg and "arguments" in msg:
                    import re
                    m = re.search(r"Expected (\d+) arguments", msg)
                    if m:
                        expected_args = int(m.group(1))
                        st.warning(f"API expects {expected_args} arguments. Using placeholders.")
                        args = [prompt] + [""] * (expected_args - 1)
                        result = client.predict(*args, fn_index=0)
                    else:
                        raise

            if result is None:
                st.error("⚠️ No result returned from the model.")
                return

            st.success("✅ Image generated successfully!")

            # --- Display Image(s) ---
            if isinstance(result, list):
                for i, img in enumerate(result):
                    st.image(img, caption=f"Generated Image {i + 1}")
            else:
                st.image(result, caption="Generated Image")

        except Exception as e:
            st.error(f"⚠️ Error while generating image: {e}")


# --- Run App ---
if __name__ == "__main__":
    main()
