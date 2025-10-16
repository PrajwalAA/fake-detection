import streamlit as st
import time
from gradio_client import Client
import os
import io
from PIL import Image
# requests and PIL.Image/io are not strictly needed here because gradio_client handles file paths

# --- Configuration ---
# NOTE: Ensure this URL is correct and active!
# Based on the original request, we use the FAF URL. If you encounter the 
# 'Could not fetch config' error again, you MUST update this URL.
GRADIO_SPACE_URL = "https://faf7c5ec7c92e2ba15.gradio.live/" 
GENERATE_FN_INDEX = 67  # Function index for image generation
STYLES_FN_INDEX = 35    # Function index for getting available styles


# Initialize the client (cached for efficiency)
@st.cache_resource
def get_client(url: str = GRADIO_SPACE_URL):
    """Initializes and caches the Gradio Client."""
    try:
        return Client(url)
    except Exception as e:
        st.error(f"Failed to initialize Gradio client. The Gradio Space is likely down or the URL is incorrect. Error: {e}")
        return None

# Function to generate image
def generate_image(client: Client, prompt: str, negative_prompt: str, style: str, aspect_ratio: str,
                     performance: str, image_number: int, seed: str, use_random_seed: bool):
    """Calls the Gradio API to generate images."""
    if client is None:
        return None

    # Determine the seed value used for the API call
    final_seed = seed if not use_random_seed else str(time.time()).replace('.', '')[:10]

    try:
        # Prepare parameters for the main generation function (fn_index=67)
        # Ensure all 153 required arguments are present (as per the error)
        params = {
            "Generate Image Grid for Each Batch": True,
            "parameter_12": prompt, 
            "Negative Prompt": negative_prompt,
            "Selected Styles": [style] if style else [],
            "Performance": performance,
            "Aspect Ratios": aspect_ratio,
            "Image Number": image_number,
            "Output Format": "png",
            # Ensure the seed passed here is the string version
            "Seed": final_seed,
            "Read wildcards in order": True,
            "Image Sharpness": 0.5,
            "Guidance Scale": 4.0,
            "Base Model (SDXL only)": "juggernautXL_v8Rundiffusion.safetensors",
            "Refiner (SDXL or SD 1.5)": "None",
            "Refiner Switch At": 0.8,
            "Enable_1": True, "LoRA 1": "None", "Weight_1": 0.0,
            "Enable_2": True, "LoRA 2": "None", "Weight_2": 0.0,
            "Enable_3": True, "LoRA 3": "None", "Weight_3": 0.0,
            "Enable_4": True, "LoRA 4": "None", "Weight_4": 0.0,
            "Enable_5": True, "LoRA 5": "None", "Weight_5": 0.0,
            "Input Image": False, "parameter_212": "",
            "Upscale or Variation:": "Disabled", "Image_1": None,
            "Outpaint Direction": [], "Image_2": None,
            "Inpaint Additional Prompt": "", "Image_3": None,
            "Mask Upload": None, "Disable Preview": False,
            "Disable Intermediate Results": False,
            "Disable seed increment": True, "Black Out NSFW": True,
            "Positive ADM Guidance Scaler": 1.0,
            "Negative ADM Guidance Scaler": 1.0,
            "ADM Guidance End At Step": 0.5,
            "CFG Mimicking from TSNR": 1.0, "CLIP Skip": 1,
            "Sampler": "euler", "Scheduler": "normal",
            "VAE": "Default (model)",
            "Forced Overwrite of Sampling Step": -1,
            "Forced Overwrite of Refiner Switch Step": -1,
            "Forced Overwrite of Generating Width": -1,
            "Forced Overwrite of Generating Height": -1,
            "Forced Overwrite of Denoising Strength of \"Vary\"": -1,
            "Forced Overwrite of Denoising Strength of \"Upscale\"": -1,
            "Mixing Image Prompt and Vary/Upscale": True,
            "Mixing Image Prompt and Inpaint": True,
            "Debug Preprocessors": True, "Skip Preprocessors": True,
            "Canny Low Threshold": 1, "Canny High Threshold": 1,
            "Refiner swap method": "joint", "Softness of ControlNet": 0.0,
            "Enabled_ControlNet": True, "B1": 0, "B2": 0, "S1": 0, "S2": 0,
            "Debug Inpaint Preprocessing": True,
            "Disable initial latent in inpaint": True, "Inpaint Engine": "None",
            "Inpaint Denoising Strength": 0.0, "Inpaint Respective Field": 0.0,
            "Enable Advanced Masking Features": True,
            "Invert Mask When Generating": True, "Mask Erode or Dilate": -64,
            "Save only final enhanced image": True,
            "Save Metadata to Images": True, "Metadata Scheme": "fooocus",
            "Image_4": None, "Stop At_1": 0.0, "Weight_1_1": 0.0, "Type_1": "ImagePrompt",
            "Image_5": None, "Stop At_2": 0.0, "Weight_2_1": 0.0, "Type_2": "ImagePrompt",
            "Image_6": None, "Stop At_3": 0.0, "Weight_3_1": 0.0, "Type_3": "ImagePrompt",
            "Image_7": None, "Stop At_4": 0.0, "Weight_4_1": 0.0, "Type_4": "ImagePrompt",
            "Debug GroundingDINO": True,
            "GroundingDINO Box Erode or Dilate": -64, "Debug Enhance Masks": True,
            "Use with Enhance, skips image generation": None, "Enhance": False,
            "Upscale or Variation:_1": "Disabled", "Order of Processing": "Before First Enhancement",
            "Prompt": "Original Prompts", "Enable_Enhance_1": True, "Detection prompt_1": "",
            "Enhancement positive prompt": "", "Enhancement negative prompt": "",
            "Mask generation model_1": "u2net", "Cloth category_1": "full",
            "SAM model_1": "vit_b", "Text Threshold_1": 0.0, "Box Threshold_1": 0.0,
            "Maximum number of detections_1": 0,
            "Disable initial latent in inpaint_1": True, "Inpaint Engine_1": "None",
            "Inpaint Denoising Strength_1": 0.0, "Inpaint Respective Field_1": 0.0,
            "Mask Erode or Dilate_1": -64, "Invert Mask_1": True,
            "Enable_Enhance_2": True, "Detection prompt_2": "",
            "Enhancement positive prompt_1": "", "Enhancement negative prompt_1": "",
            "Mask generation model_2": "u2net", "Cloth category_2": "full",
            "SAM model_2": "vit_b", "Text Threshold_2": 0.0, "Box Threshold_2": 0.0,
            "Maximum number of detections_2": 0,
            "Disable initial latent in inpaint_2": True, "Inpaint Engine_2": "None",
            "Inpaint Denoising Strength_2": 0.0, "Inpaint Respective Field_2": 0.0,
            "Mask Erode or Dilate_2": -64, "Invert Mask_2": True,
            "Enable_Enhance_3": True, "Detection prompt_3": "",
            "Enhancement positive prompt_2": "", "Enhancement negative prompt_2": "",
            "Mask generation model_3": "u2net", "Cloth category_3": "full",
            "SAM model_3": "vit_b", "Text Threshold_3": 0.0, "Box Threshold_3": 0.0,
            "Maximum number of detections_3": 0,
            "Disable initial latent in inpaint_3": True, "Inpaint Engine_3": "None",
            "Inpaint Denoising Strength_3": 0.0, "Inpaint Respective Field_3": 0.0,
            "Mask Erode or Dilate_3": -64, "Invert Mask_3": True,
        }
        
        # Calling predict
        result = client.predict(
            fn_index=GENERATE_FN_INDEX,
            **params
        )
        
        return result
    except Exception as e:
        # The 'Expected 153 arguments, got 1' error is likely happening here
        # or when processing the result. We re-raise to see the full error.
        st.error(f"Error generating image: {str(e)}")
        return None

# Function to get available styles
def get_available_styles(client: Client):
    """Fetches the list of available styles from the Gradio app."""
    if client is None:
        return []
    try:
        styles = client.predict(fn_index=STYLES_FN_INDEX)
        return styles if isinstance(styles, list) else []
    except Exception as e:
        st.error(f"Error fetching styles: {str(e)}")
        return []

# --- Main Streamlit App ---

def main():
    client = get_client()
    if client is None:
        return

    st.title("🎨 Image Generation Studio")
    st.markdown("Generate images using advanced AI models via a **Gradio Client**.")
    
    # --- Sidebar/Settings ---
    
    prompt = st.text_area("Prompt", "A beautiful landscape with mountains and a serene lake, volumetric lighting, highly detailed, cinematic", height=100)
    negative_prompt = st.text_area("Negative Prompt", "blurry, low quality, deformed, bad anatomy, error, logo, text", height=50)
    
    available_styles = []
    with st.spinner("Loading available styles..."):
        available_styles = get_available_styles(client)
    
    style_options = ["None"] + available_styles if available_styles else ["None", "Fooocus V2", "cinematic", "realistic"]
    selected_style = st.selectbox("Style", style_options)
    
    # Aspect ratio display mapping
    aspect_ratios_display = {
        "704×1408 | 1:2": "704×1408",
        "832×1216 | 13:19": "832×1216",
        "896×1152 | 7:9": "896×1152",
        "1024×1024 | 1:1": "1024×1024",
        "1152×896 | 9:7": "1152×896",
        "1216×832 | 19:13": "1216×832",
        "1408×704 | 2:1": "1408×704",
    }
    
    selected_aspect_ratio_key = st.selectbox(
        "Aspect Ratio", 
        list(aspect_ratios_display.keys()), 
        format_func=lambda x: x.split(' | ')[0] + f' | {x.split(" | ")[1]}', 
        index=3
    )
    aspect_ratio_param = aspect_ratios_display[selected_aspect_ratio_key] 
    
    performance = st.radio("Performance", ["Speed", "Quality", "Extreme Speed"], horizontal=True)
    
    # Image settings
    col1, col2 = st.columns(2)
    with col1:
        image_number = st.slider("Number of Images", 1, 8, 1)
    
    with col2:
        # Initializing use_random_seed with a session state key for better control
        if 'use_random_seed' not in st.session_state:
            st.session_state['use_random_seed'] = True
        
        use_random_seed = st.checkbox("Random Seed", st.session_state['use_random_seed'], key="random_seed_toggle")
        st.session_state['use_random_seed'] = use_random_seed

    # Seed input logic
    seed_value = ""
    if use_random_seed:
        # Generate a seed string from current time (first 10 digits of timestamp)
        seed_value = str(int(time.time() * 10000))[-10:]
        st.caption(f"**Seed used:** `{seed_value}` (generated randomly)")
    else:
        seed_value = st.text_input("Seed", "12345")
        if not seed_value.isdigit():
            st.error("Seed must be a number.")
            seed_value = "12345"
            
    # --- Generate Button and Output ---
    
    final_style = selected_style if selected_style != "None" else ""
    
    generate_button = st.button("Generate Image", type="primary", use_container_width=True)
    
    if generate_button:
        if not prompt:
            st.error("Please enter a prompt to generate an image.")
            return

        with st.spinner("⏳ Generating image... This may take a moment."):
            result = generate_image(
                client=client,
                prompt=prompt,
                negative_prompt=negative_prompt,
                style=final_style,
                aspect_ratio=aspect_ratio_param,
                performance=performance,
                image_number=image_number,
                seed=seed_value,
                use_random_seed=use_random_seed
            )
        
        if result is not None:
            st.success("✅ Image generation complete!")
            
            image_paths = []
            
            # --- Robust Gradio Output Parsing ---
            
            # Gradio's predict() often returns a list of outputs, where the 
            # actual file list/path is one element (often the first or last).
            
            if isinstance(result, list) and len(result) > 0:
                # 1. Check if the FIRST element is the file list (common structure for image generation)
                first_output = result[0]
                if isinstance(first_output, list) and all(isinstance(item, str) for item in first_output):
                     # Case 1: result is [ [path1, path2, ...], None, None, ...]
                     image_paths = first_output
                
                # 2. Check if the LAST element is the file list (less common, but possible)
                elif isinstance(result[-1], list) and all(isinstance(item, str) for item in result[-1]):
                    # Case 2: result is [ None, None, ..., [path1, path2, ...] ]
                    image_paths = result[-1]
                
                # 3. Handle single image output as a string path
                elif isinstance(first_output, str):
                    # Case 3: result is [path1, None, None, ...] (for a single image)
                    image_paths = [first_output]

            # --- Display Logic ---
            if image_paths:
                st.subheader("Generated Images")
                cols = st.columns(min(len(image_paths), 4))
                for i, img_path in enumerate(image_paths):
                    try:
                        with cols[i % len(cols)]:
                            st.image(img_path, caption=f"Image {i+1}", use_column_width="auto")
                            # Add download button
                            st.download_button(
                                label="Download",
                                data=open(img_path, "rb").read(),
                                file_name=f"generated_image_{i+1}.png",
                                mime="image/png",
                                key=f"download_{i}_{time.time()}"
                            )
                    except Exception as e:
                        st.warning(f"Could not display image {i+1} from path '{img_path}'. Error: {e}")
                        
            else:
                st.warning("⚠️ Could not locate image files in the Gradio response. The Gradio API signature may have changed.")
                st.write("**Raw Gradio Result (for debugging):**")
                st.json(result)

if __name__ == "__main__":
    main()
