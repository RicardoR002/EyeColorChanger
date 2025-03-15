import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from streamlit_image_comparison import image_comparison

from eye_detection import EyeDetector
from color_transformation import ColorTransformer

# Set page configuration
st.set_page_config(
    page_title="Eye Color Changer",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B8BF5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5C5C5C;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #666666;
    }
    .stButton button {
        background-color: #4B8BF5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #3A7AD5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'left_eye_mask' not in st.session_state:
    st.session_state.left_eye_mask = None
if 'right_eye_mask' not in st.session_state:
    st.session_state.right_eye_mask = None
if 'detection_success' not in st.session_state:
    st.session_state.detection_success = False

# Function to convert uploaded image to OpenCV format
def load_image(image_file):
    if image_file is not None:
        # Read the image file
        image_bytes = image_file.getvalue()
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL Image to OpenCV format
        opencv_image = np.array(pil_image)
        
        # Convert RGB to BGR (OpenCV format)
        if len(opencv_image.shape) == 3 and opencv_image.shape[2] == 3:
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
            
        return opencv_image
    return None

# Function to convert OpenCV image to Streamlit format
def cv2_to_streamlit(image):
    # Convert BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Function to detect eyes in the image
def detect_eyes(image, detection_method):
    eye_detector = EyeDetector()
    left_eye_mask, right_eye_mask, success = eye_detector.detect_eyes(image, method=detection_method)
    return left_eye_mask, right_eye_mask, success

# Function to change eye color
def change_eye_color(image, left_eye_mask, right_eye_mask, target_color, 
                     color_space, intensity, blend_factor, edge_strength, edge_method):
    color_transformer = ColorTransformer()
    result = color_transformer.change_eye_color(
        image, left_eye_mask, right_eye_mask, target_color,
        color_space, intensity, blend_factor, edge_strength, edge_method
    )
    return result

# Main app header
st.markdown("<h1 class='main-header'>Eye Color Changer</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Change your eye color using advanced image processing techniques</p>", unsafe_allow_html=True)

# Create sidebar for controls
st.sidebar.markdown("<h2 class='sub-header'>Controls</h2>", unsafe_allow_html=True)

# Image input section
st.sidebar.markdown("<h3>Image Input</h3>", unsafe_allow_html=True)
input_method = st.sidebar.radio("Select input method:", ["Upload Image", "Webcam"])

if input_method == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and display the uploaded image
        st.session_state.original_image = load_image(uploaded_file)
        
elif input_method == "Webcam":
    # Webcam capture
    st.sidebar.markdown("Click the button below to capture an image from your webcam")
    webcam_placeholder = st.empty()
    webcam_image = webcam_placeholder.camera_input("Take a photo")
    
    if webcam_image is not None:
        # Load and display the webcam image
        st.session_state.original_image = load_image(webcam_image)

# Eye detection section
if st.session_state.original_image is not None:
    st.sidebar.markdown("<h3>Eye Detection</h3>", unsafe_allow_html=True)
    detection_method = st.sidebar.selectbox(
        "Detection Method:",
        ["mediapipe", "haarcascade", "cnn"],
        help="Select the method for eye detection"
    )
    
    if st.sidebar.button("Detect Eyes"):
        with st.spinner("Detecting eyes..."):
            st.session_state.left_eye_mask, st.session_state.right_eye_mask, st.session_state.detection_success = detect_eyes(
                st.session_state.original_image, detection_method
            )
            
            if st.session_state.detection_success:
                st.sidebar.success("Eyes detected successfully!")
            else:
                st.sidebar.error("Failed to detect eyes. Try a different image or detection method.")

# Color transformation section
if st.session_state.detection_success:
    st.sidebar.markdown("<h3>Color Transformation</h3>", unsafe_allow_html=True)
    
    # Color selection
    color_picker = st.sidebar.color_picker("Select Target Eye Color:", "#0000FF")
    # Convert hex color to BGR (OpenCV format)
    r, g, b = tuple(int(color_picker.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    target_color = (b, g, r)  # BGR format
    
    # Color space selection
    color_space = st.sidebar.selectbox(
        "Color Space:",
        ["hsv", "rgb"],
        help="HSV is usually better for natural-looking results"
    )
    
    # Edge detection method
    edge_method = st.sidebar.selectbox(
        "Edge Detection Method:",
        ["sobel", "canny", "prewitt"],
        help="Method to preserve eye details"
    )
    
    # Adjustable parameters
    st.sidebar.markdown("<h4>Adjustable Parameters</h4>", unsafe_allow_html=True)
    
    intensity = st.sidebar.slider(
        "Color Intensity:",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Control the intensity of the new eye color"
    )
    
    blend_factor = st.sidebar.slider(
        "Blend with Original:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Control how much of the original eye color to preserve"
    )
    
    edge_strength = st.sidebar.slider(
        "Edge Preservation:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Control how much eye detail to preserve"
    )
    
    # Apply color transformation
    if st.sidebar.button("Apply Color Change"):
        with st.spinner("Changing eye color..."):
            st.session_state.processed_image = change_eye_color(
                st.session_state.original_image,
                st.session_state.left_eye_mask,
                st.session_state.right_eye_mask,
                target_color,
                color_space,
                intensity,
                blend_factor,
                edge_strength,
                edge_method
            )
            st.sidebar.success("Eye color changed successfully!")

# Display the results
if st.session_state.original_image is not None:
    # Create two columns for the main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>Original Image</h3>", unsafe_allow_html=True)
        st.image(cv2_to_streamlit(st.session_state.original_image), use_column_width=True)
        
        # Display eye masks if detection was successful
        if st.session_state.detection_success:
            st.markdown("<h4>Detected Eye Regions</h4>", unsafe_allow_html=True)
            
            # Create a visualization of the eye masks
            if st.session_state.left_eye_mask is not None and st.session_state.right_eye_mask is not None:
                # Combine masks for visualization
                combined_mask = cv2.bitwise_or(st.session_state.left_eye_mask, st.session_state.right_eye_mask)
                
                # Create a colored overlay
                overlay = st.session_state.original_image.copy()
                overlay[combined_mask > 0] = [0, 255, 0]  # Green overlay for detected eyes
                
                # Blend with original image
                alpha = 0.3
                mask_viz = cv2.addWeighted(st.session_state.original_image, 1 - alpha, overlay, alpha, 0)
                
                st.image(cv2_to_streamlit(mask_viz), use_column_width=True)
    
    with col2:
        if st.session_state.processed_image is not None:
            st.markdown("<h3>Processed Image</h3>", unsafe_allow_html=True)
            st.image(cv2_to_streamlit(st.session_state.processed_image), use_column_width=True)
            
            # Add download button for the processed image
            processed_pil = Image.fromarray(cv2_to_streamlit(st.session_state.processed_image))
            buf = io.BytesIO()
            processed_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Processed Image",
                data=byte_im,
                file_name="eye_color_changed.png",
                mime="image/png"
            )

    # Before/After comparison slider
    if st.session_state.processed_image is not None:
        st.markdown("<h3>Before/After Comparison</h3>", unsafe_allow_html=True)
        image_comparison(
            img1=cv2_to_streamlit(st.session_state.original_image),
            img2=cv2_to_streamlit(st.session_state.processed_image),
            label1="Original",
            label2="Processed",
            width=700
        )

# Information section
with st.expander("About this app"):
    st.markdown("""
    ## How it works
    
    This app uses computer vision techniques to change eye color in images:
    
    1. **Eye Detection**: First, the app detects the eyes in the image using one of three methods:
       - MediaPipe Face Mesh (most accurate for frontal faces)
       - Haar Cascade Classifiers (works for various face angles)
       - CNN-based approach (experimental)
    
    2. **Color Transformation**: Then, it applies color transformation using:
       - RGB color space: Direct color blending
       - HSV color space: Modifies hue and saturation while preserving brightness
    
    3. **Edge Preservation**: To maintain eye details, the app uses edge detection:
       - Sobel: Good for general edge detection
       - Canny: Better for detecting strong edges
       - Prewitt: Alternative edge detection algorithm
    
    ## Tips for best results
    
    - Use images with good lighting and clear visibility of the eyes
    - Front-facing portraits work best
    - Adjust the parameters to achieve the most natural look
    - HSV color space usually produces more natural results than RGB
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Created with Streamlit, OpenCV, and MediaPipe</p>", unsafe_allow_html=True) 