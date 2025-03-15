# Eye Color Changer

A Streamlit application that allows users to change eye colors in images using image processing techniques.

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
  - Different eye detection methods (MediaPipe, Haar Cascade)
  - HSV vs. RGB color space transformations
  - Different edge detection methods (Sobel, Canny, Prewitt)

- **Before/After Comparison**:
  - Side-by-side comparison of original and processed images

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app locally:
   ```
   streamlit run app.py
   ```
2. Upload an image or take a photo using your webcam
3. Adjust the parameters to achieve the desired eye color effect
4. Download the processed image

## Deployment to Streamlit Cloud

This application can be deployed to Streamlit Cloud:

1. Create a GitHub repository and push all the files:
   - app.py
   - eye_detection.py
   - color_transformation.py
   - requirements.txt
   - packages.txt

2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)

3. Create a new app in Streamlit Cloud and connect it to your GitHub repository

4. Set the main file path to `app.py`

5. Deploy the app

### Troubleshooting Deployment Issues

If you encounter dependency installation errors:

- Try using the minimal set of dependencies in requirements.txt
- Make sure packages.txt includes necessary system dependencies
- Simplify the app by removing complex features
- Check the logs in Streamlit Cloud for specific error messages

## Technical Details

This application uses:
- MediaPipe for face and eye detection
- OpenCV for image processing and edge detection
- Streamlit for the user interface

## Requirements

- Python packages: See `requirements.txt` for a complete list of dependencies
- System packages: See `packages.txt` for required system libraries 