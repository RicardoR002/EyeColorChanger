# Eye Color Changer

A Streamlit application that allows users to change eye colors in images using convolutional image filtering and edge detection techniques.

## Features

- **Image Input Options**:
  - Upload an image
  - Capture from webcam

- **Color Selection**:
  - Choose any color for the eyes

- **Adjustable Parameters**:
  - Color intensity
  - Blending with original eye color
  - Edge preservation strength

- **Multiple Processing Methods**:
  - CNN-based approach for accurate eye detection
  - HSV vs. RGB color space transformations
  - Different edge detection methods (Sobel, Canny)

- **Before/After Comparison**:
  - Side-by-side comparison of original and processed images

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
2. Upload an image or take a photo using your webcam
3. Adjust the parameters to achieve the desired eye color effect
4. Download the processed image

## Technical Details

This application uses:
- MediaPipe for face and eye detection
- OpenCV for image processing
- TensorFlow for CNN-based eye detection
- Streamlit for the user interface
- Various edge detection algorithms for preserving eye details

## Requirements

See `requirements.txt` for a complete list of dependencies. 