import streamlit as st
import requests
import io
from PIL import Image
from gradio_client import Client
import time
import os

# Set page configuration
st.set_page_config(
    page_title="Image Generation Studio",
    page_icon="🎨",
    layout="wide"
)

# Initialize the client
@st.cache_resource
def get_client():
    return Client("https://c452d26a281d4a0cde.gradio.live/")

# Function to generate image
def generate_image(client, prompt, negative_prompt, style, aspect_ratio, 
                  performance, image_number, seed, use_random_seed):
    try:
        # Prepare parameters for the main generation function (fn_index=67)
        params = {
            "Generate Image Grid for Each Batch": True,
            "parameter_12": prompt,
            "Negative Prompt": negative_prompt,
            "Selected Styles": [style] if style else [],
            "Performance": performance,
            "Aspect Ratios": aspect_ratio,
            "Image Number": image_number,
            "Output Format": "png",
            "Seed": seed if not use_random_seed else str(time.time()).replace('.', '')[:10],
            "Read wildcards in order": True,
            "Image Sharpness": 0.5,
            "Guidance Scale": 4.0,
            "Base Model (SDXL only)": "juggernautXL_v8Rundiffusion.safetensors",
            "Refiner (SDXL or SD 1.5)": "None",
            "Refiner Switch At": 0.8,
            "Enable_1": True,
            "LoRA 1": "None",
            "Weight_1": 0.0,
            "Enable_2": True,
            "LoRA 2": "None",
            "Weight_2": 0.0,
            "Enable_3": True,
            "LoRA 3": "None",
            "Weight_3": 0.0,
            "Enable_4": True,
            "LoRA 4": "None",
            "Weight_4": 0.0,
            "Enable_5": True,
            "LoRA 5": "None",
            "Weight_5": 0.0,
            "Input Image": False,
            "parameter_212": "",
            "Upscale or Variation:": "Disabled",
            "Image_1": None,
            "Outpaint Direction": [],
            "Image_2": None,
            "Inpaint Additional Prompt": "",
            "Image_3": None,
            "Mask Upload": None,
            "Disable Preview": False,
            "Disable Intermediate Results": False,
            "Disable seed increment": True,
            "Black Out NSFW": True,
            "Positive ADM Guidance Scaler": 1.0,
            "Negative ADM Guidance Scaler": 1.0,
            "ADM Guidance End At Step": 0.5,
            "CFG Mimicking from TSNR": 1.0,
            "CLIP Skip": 1,
            "Sampler": "euler",
            "Scheduler": "normal",
            "VAE": "Default (model)",
            "Forced Overwrite of Sampling Step": -1,
            "Forced Overwrite of Refiner Switch Step": -1,
            "Forced Overwrite of Generating Width": -1,
            "Forced Overwrite of Generating Height": -1,
            "Forced Overwrite of Denoising Strength of \"Vary\"": -1,
            "Forced Overwrite of Denoising Strength of \"Upscale\"": -1,
            "Mixing Image Prompt and Vary/Upscale": True,
            "Mixing Image Prompt and Inpaint": True,
            "Debug Preprocessors": True,
            "Skip Preprocessors": True,
            "Canny Low Threshold": 1,
            "Canny High Threshold": 1,
            "Refiner swap method": "joint",
            "Softness of ControlNet": 0.0,
            "Enabled_ControlNet": True,
            "B1": 0,
            "B2": 0,
            "S1": 0,
            "S2": 0,
            "Debug Inpaint Preprocessing": True,
            "Disable initial latent in inpaint": True,
            "Inpaint Engine": "None",
            "Inpaint Denoising Strength": 0.0,
            "Inpaint Respective Field": 0.0,
            "Enable Advanced Masking Features": True,
            "Invert Mask When Generating": True,
            "Mask Erode or Dilate": -64,
            "Save only final enhanced image": True,
            "Save Metadata to Images": True,
            "Metadata Scheme": "fooocus",
            "Image_4": None,
            "Stop At_1": 0.0,
            "Weight_1_1": 0.0,
            "Type_1": "ImagePrompt",
            "Image_5": None,
            "Stop At_2": 0.0,
            "Weight_2_1": 0.0,
            "Type_2": "ImagePrompt",
            "Image_6": None,
            "Stop At_3": 0.0,
            "Weight_3_1": 0.0,
            "Type_3": "ImagePrompt",
            "Image_7": None,
            "Stop At_4": 0.0,
            "Weight_4_1": 0.0,
            "Type_4": "ImagePrompt",
            "Debug GroundingDINO": True,
            "GroundingDINO Box Erode or Dilate": -64,
            "Debug Enhance Masks": True,
            "Use with Enhance, skips image generation": None,
            "Enhance": False,
            "Upscale or Variation:_1": "Disabled",
            "Order of Processing": "Before First Enhancement",
            "Prompt": "Original Prompts",
            "Enable_Enhance_1": True,
            "Detection prompt_1": "",
            "Enhancement positive prompt": "",
            "Enhancement negative prompt": "",
            "Mask generation model_1": "u2net",
            "Cloth category_1": "full",
            "SAM model_1": "vit_b",
            "Text Threshold_1": 0.0,
            "Box Threshold_1": 0.0,
            "Maximum number of detections_1": 0,
            "Disable initial latent in inpaint_1": True,
            "Inpaint Engine_1": "None",
            "Inpaint Denoising Strength_1": 0.0,
            "Inpaint Respective Field_1": 0.0,
            "Mask Erode or Dilate_1": -64,
            "Invert Mask_1": True,
            "Enable_Enhance_2": True,
            "Detection prompt_2": "",
            "Enhancement positive prompt_1": "",
            "Enhancement negative prompt_1": "",
            "Mask generation model_2": "u2net",
            "Cloth category_2": "full",
            "SAM model_2": "vit_b",
            "Text Threshold_2": 0.0,
            "Box Threshold_2": 0.0,
            "Maximum number of detections_2": 0,
            "Disable initial latent in inpaint_2": True,
            "Inpaint Engine_2": "None",
            "Inpaint Denoising Strength_2": 0.0,
            "Inpaint Respective Field_2": 0.0,
            "Mask Erode or Dilate_2": -64,
            "Invert Mask_2": True,
            "Enable_Enhance_3": True,
            "Detection prompt_3": "",
            "Enhancement positive prompt_2": "",
            "Enhancement negative prompt_2": "",
            "Mask generation model_3": "u2net",
            "Cloth category_3": "full",
            "SAM model_3": "vit_b",
            "Text Threshold_3": 0.0,
            "Box Threshold_3": 0.0,
            "Maximum number of detections_3": 0,
            "Disable initial latent in inpaint_3": True,
            "Inpaint Engine_3": "None",
            "Inpaint Denoising Strength_3": 0.0,
            "Inpaint Respective Field_3": 0.0,
            "Mask Erode or Dilate_3": -64,
            "Invert Mask_3": True,
        }
        
        result = client.predict(
            fn_index=67,
            **params
        )
        
        return result
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

# Function to get available styles
def get_available_styles(client):
    try:
        # This function (fn_index=35) returns available styles
        styles = client.predict(fn_index=35)
        return styles
    except Exception as e:
        st.error(f"Error fetching styles: {str(e)}")
        return []

# Main app
def main():
    client = get_client()
    
    st.title("🎨 Image Generation Studio")
    st.markdown("Generate images using advanced AI models")
    
    # Sidebar for settings
    st.sidebar.title("Settings")
    
    # Text inputs
    prompt = st.text_area("Prompt", "A beautiful landscape with mountains and a lake", height=100)
    negative_prompt = st.text_area("Negative Prompt", "blurry, low quality", height=50)
    
    # Get available styles
    with st.spinner("Loading available styles..."):
        available_styles = get_available_styles(client)
    
    # Style selection
    selected_style = st.selectbox("Style", ["None"] + available_styles if available_styles else ["Fooocus V2", "cinematic", "realistic"])
    
    # Aspect ratio
    aspect_ratios = [
        "704×1408 <span style=\"color: grey;\"> ∣ 1:2</span>",
        "832×1216 <span style=\"color: grey;\"> ∣ 13:19</span>",
        "896×1152 <span style=\"color: grey;\"> ∣ 7:9</span>",
        "1024×1024 <span style=\"color: grey;\"> ∣ 1:1</span>",
        "1152×896 <span style=\"color: grey;\"> ∣ 9:7</span>",
        "1216×832 <span style=\"color: grey;\"> ∣ 19:13</span>",
        "1408×704 <span style=\"color: grey;\"> ∣ 2:1</span>"
    ]
    aspect_ratio = st.selectbox("Aspect Ratio", aspect_ratios)
    
    # Performance mode
    performance = st.radio("Performance", ["Speed", "Quality", "Extreme Speed"])
    
    # Image settings
    col1, col2 = st.columns(2)
    with col1:
        image_number = st.slider("Number of Images", 1, 8, 1)
    with col2:
        use_random_seed = st.checkbox("Random Seed", True)
    
    if not use_random_seed:
        seed = st.text_input("Seed", "12345")
    else:
        seed = str(time.time()).replace('.', '')[:10]
    
    # Generate button
    generate_button = st.button("Generate Image", type="primary")
    
    # Display generated images
    if generate_button:
        if not prompt:
            st.error("Please enter a prompt")
        else:
            with st.spinner("Generating image..."):
                result = generate_image(
                    client=client,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    style=selected_style if selected_style != "None" else "",
                    aspect_ratio=aspect_ratio,
                    performance=performance,
                    image_number=image_number,
                    seed=seed,
                    use_random_seed=use_random_seed
                )
                
                if result:
                    # The result should contain image paths or data
                    # This will depend on the actual API response format
                    st.success("Image generated successfully!")
                    
                    # Try to display the result
                    try:
                        # This is a placeholder - you'll need to adjust based on actual response format
                        if isinstance(result, list) and len(result) > 0:
                            for i, img_path in enumerate(result):
                                st.image(img_path, caption=f"Generated Image {i+1}")
                        elif isinstance(result, str):
                            st.image(result, caption="Generated Image")
                        else:
                            st.json(result)
                    except Exception as e:
                        st.error(f"Error displaying image: {str(e)}")
                        st.write("Raw result:")
                        st.write(result)

if __name__ == "__main__":
    main()
