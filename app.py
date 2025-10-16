import streamlit as st
import time
from gradio_client import Client

# Set page configuration
st.set_page_config(
    page_title="Image Generation Studio",
    page_icon="🎨",
    layout="wide"
)

# --- Configuration ---
GRADIO_SPACE_URL = "https://faf7c5ec7c92e2ba15.gradio.live/"
GENERATE_FN_INDEX = 67  # Function index for image generation
STYLES_FN_INDEX = 35    # Function index for getting available styles

# --- Client and Utility Functions ---

# Initialize the client (cached for efficiency)
@st.cache_resource
def get_client(url: str = GRADIO_SPACE_URL):
    """Initializes and caches the Gradio Client."""
    try:
        return Client(url)
    except Exception as e:
        st.error(f"Failed to initialize Gradio client: {e}")
        return None

# Function to generate image
def generate_image(client: Client, prompt: str, negative_prompt: str, style: str, aspect_ratio: str,
                     performance: str, image_number: int, seed: str):
    """Calls the Gradio API to generate images."""
    if client is None:
        return None

    try:
        # Prepare parameters for the main generation function (fn_index=67)
        params = {
            "Generate Image Grid for Each Batch": True,
            "parameter_12": prompt, # This is the primary prompt input
            "Negative Prompt": negative_prompt,
            "Selected Styles": [style] if style else [],
            "Performance": performance,
            "Aspect Ratios": aspect_ratio,
            "Image Number": image_number,
            "Output Format": "png",
            "Seed": seed,
            # Other parameters are kept as defined in the original code,
            # assuming they are the required defaults for fn_index=67
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
        st.error(f"Error generating image: {str(e)}")
        return None

# Function to get available styles
def get_available_styles(client: Client):
    """Fetches the list of available styles from the Gradio app."""
    if client is None:
        return []
    try:
        # This function (fn_index=35) returns available styles as a list
        styles = client.predict(fn_index=STYLES_FN_INDEX)
        # Gradio client often returns the result wrapped in a list, like ['style1', 'style2']
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
    
    # Text inputs
    prompt = st.text_area("Prompt", "A beautiful landscape with mountains and a serene lake, volumetric lighting, highly detailed, cinematic", height=100)
    negative_prompt = st.text_area("Negative Prompt", "blurry, low quality, deformed, malformed, bad anatomy, error, logo, text", height=50)
    
    # Get available styles
    available_styles = []
    with st.spinner("Loading available styles..."):
        available_styles = get_available_styles(client)
    
    # Style selection
    style_options = ["None"] + available_styles if available_styles else ["None", "Fooocus V2", "cinematic", "realistic"]
    selected_style = st.selectbox("Style", style_options)
    
    # Aspect ratio (using markdown for visual cue)
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
        format_func=lambda x: x.split(' | ')[0] + f' <span style="color: grey;">| {x.split(" | ")[1]}</span>', 
        index=3 # Default to 1024x1024
    )
    aspect_ratio_param = aspect_ratios_display[selected_aspect_ratio_key] # Get the raw string for the API
    
    # Performance mode
    performance = st.radio("Performance", ["Speed", "Quality", "Extreme Speed"], horizontal=True)
    
    # Image settings
    col1, col2 = st.columns(2)
    with col1:
        image_number = st.slider("Number of Images", 1, 8, 1)
    
    with col2:
        use_random_seed = st.checkbox("Random Seed", True, key="random_seed_toggle")
    
    # Seed input logic
    if use_random_seed:
        # Generate a seed string from current time (first 10 digits of timestamp)
        seed_value = str(time.time()).replace('.', '')[:10]
        st.caption(f"**Current Random Seed:** `{seed_value}` (will change on each run)")
    else:
        seed_input = st.text_input("Seed", "12345")
        try:
            # Ensure the seed is a valid integer string for the API call
            int(seed_input) 
            seed_value = seed_input
        except ValueError:
            st.error("Seed must be an integer.")
            seed_value = "12345" # Fallback
    
    # --- Generate Button and Output ---
    
    # Map 'None' style to empty string for the API call
    final_style = selected_style if selected_style != "None" else ""
    
    generate_button = st.button("Generate Image", type="primary", use_container_width=True)
    
    if generate_button:
        if not prompt:
            st.error("Please enter a prompt to generate an image.")
        else:
            with st.spinner("⏳ Generating image... This may take a moment based on the server load."):
                result = generate_image(
                    client=client,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    style=final_style,
                    aspect_ratio=aspect_ratio_param,
                    performance=performance,
                    image_number=image_number,
                    seed=seed_value
                )
            
            if result is not None:
                st.success("✅ Image generated successfully!")
                
                # --- Crucial Fix: Gradio Client Output Handling ---
                
                # Gradio API for an image output typically returns a dictionary 
                # or a list of dictionaries with a file path under a 'name' key.
                
                # Attempt to extract image paths/data from the result
                image_paths = []
                
                # Check if the result is a list and contains a file object (the generated image(s))
                if isinstance(result, list) and len(result) > 0:
                    # The main output is often the first element, which might be a list of paths or a dict of a path
                    first_output = result[0]
                    if isinstance(first_output, list):
                         # If it's a list of paths (common for multiple images)
                         image_paths = first_output
                    elif isinstance(first_output, dict) and 'name' in first_output:
                         # If it's a single dict with a 'name' key (single image)
                         image_paths = [first_output['name']]

                if image_paths:
                    st.subheader("Generated Images")
                    cols = st.columns(min(image_number, 4)) # Display up to 4 images per row
                    for i, img_path in enumerate(image_paths):
                        try:
                            # Use the columns to display images in a grid
                            with cols[i % len(cols)]:
                                st.image(img_path, caption=f"Image {i+1}", use_column_width="auto")
                                # Provide a download button
                                st.download_button(
                                    label="Download",
                                    data=open(img_path, "rb").read(),
                                    file_name=f"generated_image_{i+1}_{prompt[:20].replace(' ', '_')}.png",
                                    mime="image/png",
                                    key=f"download_{i}"
                                )
                        except Exception as e:
                            st.warning(f"Could not display image {i+1} from path '{img_path}'. Error: {e}")
                            
                else:
                    st.warning("⚠️ Could not locate image files in the Gradio response. Raw result is below.")
                    st.json(result) # Display raw result for debugging

if __name__ == "__main__":
    main()
