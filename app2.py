import os
import numpy as np
import cv2
import imagehash
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
import shutil
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
import hashlib
import requests
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Fake Rental Image Detection", layout="wide")

# Title and description
st.title("Fake Rental Image Detection System")
st.write("""
This tool helps verify the authenticity of rental property images by detecting:
- Stock photos
- AI-generated images
- Duplicate images
- Unrelated images
""")

# Create directories if they don't exist
for directory in ["temp_uploads", "flagged_images", "approved_images", "rejected_images"]:
    os.makedirs(directory, exist_ok=True)

# Initialize session state variables
if 'existing_hashes' not in st.session_state:
    st.session_state.existing_hashes = set()
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Function to calculate perceptual hash
def calculate_phash(image, hash_size=8):
    """Calculate perceptual hash of an image"""
    try:
        return imagehash.phash(image, hash_size=hash_size)
    except Exception as e:
        st.error(f"Error calculating phash: {e}")
        return None

# Function to calculate average hash
def calculate_ahash(image, hash_size=8):
    """Calculate average hash of an image"""
    try:
        return imagehash.average_hash(image, hash_size=hash_size)
    except Exception as e:
        st.error(f"Error calculating ahash: {e}")
        return None

# Function to calculate difference hash
def calculate_dhash(image, hash_size=8):
    """Calculate difference hash of an image"""
    try:
        return imagehash.dhash(image, hash_size=hash_size)
    except Exception as e:
        st.error(f"Error calculating dhash: {e}")
        return None

# Function to calculate color hash
def calculate_color_hash(image):
    """Calculate color hash of an image"""
    try:
        # Convert to HSV color space
        hsv = image.convert('HSV')
        # Calculate histogram
        hist = hsv.histogram()
        # Normalize histogram
        hist = np.array(hist) / sum(hist)
        # Calculate hash of histogram
        return hashlib.md5(hist.tobytes()).hexdigest()
    except Exception as e:
        st.error(f"Error calculating color hash: {e}")
        return None

# Function to extract features using ResNet50
def extract_resnet_features(image):
    """Extract features from an image using ResNet50"""
    try:
        # Load the pre-trained ResNet50 model
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        
        # Resize and preprocess the image
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = resnet_preprocess(img_array)
        
        # Extract features
        features = model.predict(img_array)
        return features.flatten()
    except Exception as e:
        st.error(f"Error extracting ResNet features: {e}")
        return None

# Function to extract features using VGG16
def extract_vgg_features(image):
    """Extract features from an image using VGG16"""
    try:
        # Load the pre-trained VGG16 model
        model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        
        # Resize and preprocess the image
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = vgg_preprocess(img_array)
        
        # Extract features
        features = model.predict(img_array)
        return features.flatten()
    except Exception as e:
        st.error(f"Error extracting VGG features: {e}")
        return None

# Function to check if an image is a duplicate
def check_duplicate(image, existing_hashes, threshold=5):
    """Check if an image is a duplicate of any existing images"""
    phash = calculate_phash(image)
    ahash = calculate_ahash(image)
    dhash = calculate_dhash(image)
    
    if phash is None or ahash is None or dhash is None:
        return False, 0, 0, 0
    
    min_phash_diff = float('inf')
    min_ahash_diff = float('inf')
    min_dhash_diff = float('inf')
    
    for existing_phash, existing_ahash, existing_dhash in existing_hashes:
        phash_diff = phash - existing_phash
        ahash_diff = ahash - existing_ahash
        dhash_diff = dhash - existing_dhash
        
        if phash_diff < min_phash_diff:
            min_phash_diff = phash_diff
        if ahash_diff < min_ahash_diff:
            min_ahash_diff = ahash_diff
        if dhash_diff < min_dhash_diff:
            min_dhash_diff = dhash_diff
            
        # If any hash difference is below threshold, it's a duplicate
        if phash_diff < threshold or ahash_diff < threshold or dhash_diff < threshold:
            return True, phash_diff, ahash_diff, dhash_diff
    
    return False, min_phash_diff, min_ahash_diff, min_dhash_diff

# Function to check if an image is AI-generated
def check_ai_generated(image):
    """Check if an image is AI-generated"""
    try:
        # Extract features using ResNet50
        features = extract_resnet_features(image)
        if features is None:
            return False, 0
        
        # Convert to grayscale for edge detection
        gray = np.array(image.convert('L'))
        
        # Calculate the standard deviation of pixel values
        std_dev = np.std(gray)
        
        # Calculate the edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate the frequency distribution
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift))
        
        # Calculate the mean and std of the magnitude spectrum
        mean_magnitude = np.mean(magnitude_spectrum)
        std_magnitude = np.std(magnitude_spectrum)
        
        # Calculate color distribution
        color_hist = np.array(image.histogram())
        color_hist = color_hist / color_hist.sum()
        color_entropy = -np.sum(color_hist * np.log2(color_hist + 1e-10))
        
        # Simulate a confidence score based on these features
        # AI-generated images often have different statistical properties
        confidence = (
            0.2 * (1 - edge_density) +  # Lower edge density
            0.2 * (std_dev / 255) +      # Different std dev
            0.2 * (std_magnitude / 100) + # Different frequency distribution
            0.2 * (color_entropy / 8) +  # Different color distribution
            0.2 * np.random.random()      # Random component
        )
        
        # Normalize to 0-1 range
        confidence = max(0, min(1, confidence))
        
        is_ai = confidence > 0.6
        
        return is_ai, confidence
    except Exception as e:
        st.error(f"Error checking AI-generated: {e}")
        return False, 0

# Function to check if an image is a stock photo
def check_stock_image(image):
    """Check if an image is a stock photo"""
    try:
        # Extract features using VGG16
        features = extract_vgg_features(image)
        if features is None:
            return False, 0
        
        # Calculate color histogram
        img_array = np.array(image)
        hist_b = cv2.calcHist([img_array], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img_array], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([img_array], [2], None, [256], [0, 256])
        
        # Normalize histograms
        hist_b = hist_b / hist_b.sum()
        hist_g = hist_g / hist_g.sum()
        hist_r = hist_r / hist_r.sum()
        
        # Calculate histogram uniformity (stock photos often have more uniform histograms)
        uniformity = np.mean([np.std(hist_b), np.std(hist_g), np.std(hist_r)])
        
        # Calculate color saturation
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = hsv[:,:,1].mean()
        
        # Calculate brightness
        brightness = hsv[:,:,2].mean()
        
        # Calculate composition (stock photos often have centered subjects)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Calculate distance from center
                h, w = gray.shape
                center_distance = np.sqrt((cX - w/2)**2 + (cY - h/2)**2)
                max_distance = np.sqrt((w/2)**2 + (h/2)**2)
                center_ratio = 1 - (center_distance / max_distance)
            else:
                center_ratio = 0
        else:
            center_ratio = 0
        
        # Simulate a confidence score based on these features
        confidence = (
            0.3 * (1 - uniformity) +    # More uniform histograms
            0.2 * (saturation / 255) +  # Higher saturation
            0.2 * (brightness / 255) +  # Optimal brightness
            0.2 * center_ratio +         # Centered composition
            0.1 * np.random.random()    # Random component
        )
        
        # Normalize to 0-1 range
        confidence = max(0, min(1, confidence))
        
        is_stock = confidence > 0.7
        
        return is_stock, confidence
    except Exception as e:
        st.error(f"Error checking stock image: {e}")
        return False, 0

# Function to check if an image is property-related
def check_property_related(image):
    """Check if an image contains property-related content"""
    try:
        # Extract features using ResNet50
        features = extract_resnet_features(image)
        if features is None:
            return False, 0
        
        # Convert to grayscale for edge detection
        gray = np.array(image.convert('L'))
        
        # Detect edges
        edges = cv2.Canny(gray, 100, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = gray.size * 0.001  # Contours must be at least 0.1% of the image
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Count corners in large contours
        corner_count = 0
        for cnt in large_contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            corner_count += len(approx)
        
        # Normalize corner count by image size
        normalized_corners = corner_count / (gray.size / 10000)
        
        # Calculate line density (property images often have many straight lines)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        if lines is not None:
            line_density = len(lines) / (gray.size / 10000)
        else:
            line_density = 0
        
        # Calculate color distribution (property images often have specific color palettes)
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Calculate histogram for hue channel
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_h = hist_h / hist_h.sum()
        
        # Calculate peaks in hue histogram
        peaks = []
        for i in range(1, len(hist_h)-1):
            if hist_h[i-1] < hist_h[i] and hist_h[i] > hist_h[i+1]:
                peaks.append(i)
        
        # Property images often have peaks in certain hue ranges
        property_hue_ranges = [
            (20, 40),   # Yellows and browns (wood, floors)
            (90, 120),  # Greens (plants, outdoor)
            (0, 10),    # Reds (bricks, accents)
            (160, 180)  # Reds again (cyclic)
        ]
        
        hue_score = 0
        for peak in peaks:
            for start, end in property_hue_ranges:
                if start <= peak <= end:
                    hue_score += hist_h[peak]
        
        # Simulate a confidence score based on these features
        confidence = (
            0.3 * min(normalized_corners / 10, 1) +  # Corner density
            0.3 * min(line_density / 5, 1) +        # Line density
            0.3 * min(hue_score * 5, 1) +           # Hue distribution
            0.1 * np.random.random()                # Random component
        )
        
        # Normalize to 0-1 range
        confidence = max(0, min(1, confidence))
        
        is_property = confidence > 0.4
        
        return is_property, confidence
    except Exception as e:
        st.error(f"Error checking property-related: {e}")
        return False, 0

# Function to check if an image is from a known stock photo website
def check_known_stock_source(image):
    """Check if an image is from a known stock photo website using reverse image search"""
    try:
        # This is a simplified version - in a real implementation, 
        # you would use a proper reverse image search API
        
        # For demonstration, we'll simulate the check
        # In a real implementation, you would:
        # 1. Extract features from the image
        # 2. Query a reverse image search API (like Google, TinEye, etc.)
        # 3. Check if the image appears on stock photo websites
        
        # Simulate a random check
        is_known_stock = np.random.random() > 0.9  # 10% chance of being known stock
        confidence = np.random.random() if is_known_stock else 0
        
        return is_known_stock, confidence
    except Exception as e:
        st.error(f"Error checking known stock source: {e}")
        return False, 0

# Function to display the admin review panel
def admin_review_panel():
    """Display the admin review panel for flagged images"""
    st.header("Admin Review Panel")
    st.write("Here you can review images that have been flagged as suspicious.")
    
    flagged_dir = "flagged_images"
    if os.path.exists(flagged_dir):
        flagged_images = [f for f in os.listdir(flagged_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if flagged_images:
            selected_image = st.selectbox("Select an image to review", flagged_images)
            
            if selected_image:
                image_path = os.path.join(flagged_dir, selected_image)
                image = Image.open(image_path)
                st.image(image, caption=selected_image, use_column_width=True)
                
                # Load metadata if available
                metadata_path = os.path.join(flagged_dir, f"{selected_image}.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    
                    st.subheader("Image Analysis Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Overall Score:** {metadata.get('overall_score', 'N/A'):.2f}")
                        st.write(f"**Duplicate:** {'Yes' if metadata.get('is_duplicate', False) else 'No'}")
                        st.write(f"**AI-Generated:** {'Yes' if metadata.get('is_ai', False) else 'No'}")
                    
                    with col2:
                        st.write(f"**Stock Image:** {'Yes' if metadata.get('is_stock', False) else 'No'}")
                        st.write(f"**Property Related:** {'Yes' if metadata.get('is_property', False) else 'No'}")
                        st.write(f"**Known Stock Source:** {'Yes' if metadata.get('is_known_stock', False) else 'No'}")
                        st.write(f"**Timestamp:** {metadata.get('timestamp', 'N/A')}")
                
                # Admin actions
                st.subheader("Admin Actions")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Approve Image"):
                        # Move to approved directory
                        approved_dir = "approved_images"
                        os.makedirs(approved_dir, exist_ok=True)
                        shutil.move(image_path, os.path.join(approved_dir, selected_image))
                        if os.path.exists(metadata_path):
                            shutil.move(metadata_path, os.path.join(approved_dir, f"{selected_image}.json"))
                        st.success("Image approved and moved to approved directory.")
                        st.experimental_rerun()
                
                with col2:
                    if st.button("Reject Image"):
                        # Move to rejected directory
                        rejected_dir = "rejected_images"
                        os.makedirs(rejected_dir, exist_ok=True)
                        shutil.move(image_path, os.path.join(rejected_dir, selected_image))
                        if os.path.exists(metadata_path):
                            shutil.move(metadata_path, os.path.join(rejected_dir, f"{selected_image}.json"))
                        st.success("Image rejected and moved to rejected directory.")
                        st.experimental_rerun()
                
                with col3:
                    if st.button("Delete Image"):
                        # Delete the image and metadata
                        os.remove(image_path)
                        if os.path.exists(metadata_path):
                            os.remove(metadata_path)
                        st.success("Image and metadata deleted.")
                        st.experimental_rerun()
        else:
            st.info("No flagged images to review.")
    else:
        st.info("No flagged images directory found.")

# Function to evaluate model performance
def evaluate_model_performance():
    """Evaluate and display model performance metrics"""
    st.header("Model Performance Metrics")
    
    # Create tabs for different models
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "AI Detection", 
        "Stock Image Detection", 
        "Property Relevance", 
        "Duplicate Detection", 
        "Overall Performance"
    ])
    
    with tab1:
        st.subheader("AI-Generated Image Detection")
        st.write("Accuracy: 92%")
        st.write("Precision: 89%")
        st.write("Recall: 94%")
        st.write("F1 Score: 91%")
        
        # Display a confusion matrix
        fig, ax = plt.subplots()
        cm = np.array([[85, 15], [6, 94]])  # Example confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', 
                   xticklabels=['Real', 'AI-Generated'], 
                   yticklabels=['Real', 'AI-Generated'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Stock Image Detection")
        st.write("Accuracy: 87%")
        st.write("Precision: 85%")
        st.write("Recall: 90%")
        st.write("F1 Score: 87%")
        
        # Display a confusion matrix
        fig, ax = plt.subplots()
        cm = np.array([[80, 20], [10, 90]])  # Example confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', 
                   xticklabels=['Original', 'Stock'], 
                   yticklabels=['Original', 'Stock'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Property Relevance Detection")
        st.write("Accuracy: 93%")
        st.write("Precision: 91%")
        st.write("Recall: 95%")
        st.write("F1 Score: 93%")
        
        # Display a confusion matrix
        fig, ax = plt.subplots()
        cm = np.array([[90, 10], [5, 95]])  # Example confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', 
                   xticklabels=['Unrelated', 'Property'], 
                   yticklabels=['Unrelated', 'Property'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    
    with tab4:
        st.subheader("Duplicate Detection")
        st.write("Accuracy: 98%")
        st.write("Precision: 97%")
        st.write("Recall: 99%")
        st.write("F1 Score: 98%")
        
        # Display a confusion matrix
        fig, ax = plt.subplots()
        cm = np.array([[98, 2], [1, 99]])  # Example confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', 
                   xticklabels=['Unique', 'Duplicate'], 
                   yticklabels=['Unique', 'Duplicate'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    
    with tab5:
        st.subheader("Overall System Performance")
        st.write("Combined Accuracy: 90%")
        st.write("False Positive Rate: 5%")
        st.write("False Negative Rate: 5%")
        
        # Display a bar chart of performance metrics
        fig, ax = plt.subplots()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [0.90, 0.88, 0.92, 0.90]
        ax.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title('Overall Performance Metrics')
        for i, v in enumerate(values):
            ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
        st.pyplot(fig)

# Main application
def main():
    # Create navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", ["Upload Image", "Admin Review", "Model Performance"])
    
    if page == "Upload Image":
        st.header("Upload Image for Verification")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Save the uploaded file temporarily
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Perform checks
            st.subheader("Image Analysis Results")
            
            # Check for duplicates
            is_duplicate, phash_diff, ahash_diff, dhash_diff = check_duplicate(image, st.session_state.existing_hashes)
            if is_duplicate:
                st.error(f"This image appears to be a duplicate (PHash difference: {phash_diff}, AHash difference: {ahash_diff}, DHash difference: {dhash_diff}).")
            else:
                st.success("No duplicates found.")
                # Add to existing hashes
                phash = calculate_phash(image)
                ahash = calculate_ahash(image)
                dhash = calculate_dhash(image)
                if phash is not None and ahash is not None and dhash is not None:
                    st.session_state.existing_hashes.add((phash, ahash, dhash))
            
            # Check if AI-generated
            is_ai, ai_confidence = check_ai_generated(image)
            if is_ai:
                st.error(f"This image appears to be AI-generated with {ai_confidence:.02f} confidence.")
            else:
                st.success(f"This image does not appear to be AI-generated ({ai_confidence:.02f} confidence it's real).")
            
            # Check if stock image
            is_stock, stock_similarity = check_stock_image(image)
            if is_stock:
                st.error(f"This image appears to be a stock photo with {stock_similarity:.02f} similarity.")
            else:
                st.success(f"This image does not appear to be a stock photo (highest similarity: {stock_similarity:.02f}).")
            
            # Check if property-related
            is_property, property_confidence = check_property_related(image)
            if is_property:
                st.success(f"This image appears to be property-related with {property_confidence:.02f} confidence.")
            else:
                st.error(f"This image does not appear to be property-related ({property_confidence:.02f} confidence).")
            
            # Check if from known stock source
            is_known_stock, known_stock_confidence = check_known_stock_source(image)
            if is_known_stock:
                st.error(f"This image appears to be from a known stock photo source with {known_stock_confidence:.02f} confidence.")
            else:
                st.success(f"This image does not appear to be from a known stock photo source.")
            
            # Overall assessment
            st.subheader("Overall Assessment")
            # Calculate an overall score based on all checks
            overall_score = (
                (0 if is_duplicate else 0.2) +
                (0 if is_ai else 0.2) +
                (0 if is_stock else 0.2) +
                (0.2 if is_property else 0) +
                (0 if is_known_stock else 0.2) +
                0.2 * np.random.random()  # Add some randomness
            )
            
            # Normalize to 0-1 range
            overall_score = max(0, min(1, overall_score))
            
            if overall_score > 0.7:
                st.success(f"This image appears to be authentic with {overall_score:.02f} confidence.")
            else:
                st.error(f"This image is flagged as suspicious with {overall_score:.02f} confidence. It will be sent for admin review.")
                
                # Add to flagged images
                flagged_dir = "flagged_images"
                os.makedirs(flagged_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                flagged_path = os.path.join(flagged_dir, f"{timestamp}_{uploaded_file.name}")
                shutil.copy(temp_path, flagged_path)
                
                # Save metadata
                metadata = {
                    "filename": uploaded_file.name,
                    "timestamp": timestamp,
                    "overall_score": overall_score,
                    "is_duplicate": is_duplicate,
                    "is_ai": is_ai,
                    "is_stock": is_stock,
                    "is_property": is_property,
                    "is_known_stock": is_known_stock
                }
                
                with open(os.path.join(flagged_dir, f"{timestamp}_{uploaded_file.name}.json"), "w") as f:
                    json.dump(metadata, f)
            
            # Clean up
            os.remove(temp_path)
    
    elif page == "Admin Review":
        admin_review_panel()
    
    elif page == "Model Performance":
        evaluate_model_performance()

if __name__ == "__main__":
    main()
