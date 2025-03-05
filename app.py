import streamlit as st
from PIL import Image  # For image processing
import numpy as np    # For numerical operations
import io            # For handling binary I/O
from colorization_model import colorize_image, enhance_image, detect_faces, colorize_with_advanced_model, detect_faces_dnn
import time
import os
from typing import Dict, Any, Tuple

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Image Processing App",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
        }
        .stProgress .st-bo {
            background-color: #00ff00;
        }
        .success-message {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            color: #155724;
            margin: 1rem 0;
        }
        .error-message {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f8d7da;
            color: #721c24;
            margin: 1rem 0;
        }
        .info-message {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #cce5ff;
            color: #004085;
            margin: 1rem 0;
        }
        .stImage {
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_validate_image(uploaded_file) -> np.ndarray:
    """
    Load and validate the uploaded image with caching for better performance.
    
    Args:
        uploaded_file: The uploaded file object
        
    Returns:
        np.ndarray: The validated image array
        
    Raises:
        ValueError: If the image is invalid or too large
    """
    try:
        image = Image.open(uploaded_file)
        
        # Validate image size
        if image.size[0] * image.size[1] > 5000 * 5000:
            raise ValueError("Image is too large. Maximum size allowed is 5000x5000 pixels.")
        
        # Validate image mode
        if image.mode not in ['RGB', 'L']:
            image = image.convert('RGB')
        
        # Convert to numpy array for consistent hashing
        return np.array(image)
    except Exception as e:
        raise ValueError(f"Error loading image: {str(e)}")

@st.cache_data
def process_image(_image_array: np.ndarray, process_option: str) -> Dict[str, Any]:
    """
    Process the image based on selected mode with caching for better performance.
    
    Args:
        _image_array: The input image array
        process_option: The selected processing option
        
    Returns:
        Dict[str, Any]: Dictionary containing processed image and metadata
        
    Raises:
        RuntimeError: If processing fails
    """
    try:
        # Convert numpy array back to PIL Image
        image = Image.fromarray(_image_array)
        
        if process_option == "Convert to Grayscale":
            return {
                'image': image.convert('L'),
                'caption': "Grayscale Image",
                'filename': "grayscale_image.png"
            }
        
        elif process_option == "Add Artistic Color (Simple)":
            gray_image = image.convert('L')
            colored_array = colorize_image(gray_image)
            return {
                'image': Image.fromarray(colored_array),
                'caption': "Artistically Colored Image (Simple)",
                'filename': "colored_image_simple.png"
            }
            
        elif process_option == "Add Artistic Color (AI-Powered)":
            gray_image = image.convert('L')
            colored_array = colorize_with_advanced_model(gray_image)
            return {
                'image': Image.fromarray(colored_array),
                'caption': "Artistically Colored Image (AI-Powered)",
                'filename': "colored_image_ai_powered.png"
            }
        
        elif process_option == "Enhance Image Quality":
            enhanced_array = enhance_image(image)
            return {
                'image': Image.fromarray(enhanced_array),
                'caption': "Enhanced Image",
                'filename': "enhanced_image.png"
            }
        
        elif process_option == "Detect Faces (Haar)":
            result_array, num_faces = detect_faces(image)
            return {
                'image': Image.fromarray(result_array.astype('uint8')),
                'caption': f"Detected {num_faces} face{'s' if num_faces != 1 else ''} (Haar)",
                'filename': "detected_faces_haar.png",
                'metadata': {'num_faces': num_faces}
            }
            
        elif process_option == "Detect Faces (DNN)":
            result_array, num_faces = detect_faces_dnn(image)
            return {
                'image': Image.fromarray(result_array.astype('uint8')),
                'caption': f"Detected {num_faces} face{'s' if num_faces != 1 else ''} (DNN)",
                'filename': "detected_faces_dnn.png",
                'metadata': {'num_faces': num_faces}
            }
        
    except Exception as e:
        raise RuntimeError(f"Error processing image: {str(e)}")

def display_error_message(message: str) -> None:
    """Display an error message with custom styling."""
    st.markdown(f'<div class="error-message">⚠️ {message}</div>', unsafe_allow_html=True)

def display_success_message(message: str) -> None:
    """Display a success message with custom styling."""
    st.markdown(f'<div class="success-message">✅ {message}</div>', unsafe_allow_html=True)

def display_info_message(message: str) -> None:
    """Display an info message with custom styling."""
    st.markdown(f'<div class="info-message">ℹ️ {message}</div>', unsafe_allow_html=True)

def main():
    # Set up the Streamlit interface
    st.title("🎨 Advanced Image Processing App")
    
    # Add app description with markdown formatting
    st.markdown("""
    ### Welcome to the Advanced Image Processing App!
    
    This application was developed using [Cursor](https://cursor.sh), the AI-first code editor. 
    For image colorization, we provide two options:
    - A fast, simple colorization using OpenCV
    - An advanced AI-powered colorization using a deep ResNet model
    
    #### Features:
    - 🖼️ Convert color images to grayscale
    - 🎨 Add artistic color (Simple or AI-powered)
    - ✨ Enhance image quality
    - 👤 Detect faces using Haar features or DNN
    - 🔍 Zoom images while maintaining quality
    
    👇 Get started by uploading an image below!
    """)

    # Create a dropdown in the sidebar for selecting the conversion mode
    with st.sidebar:
        st.header("⚙️ Settings")
        process_option = st.selectbox(
            "Select Processing Mode",
            [
                "Enhance Image Quality",
                "Convert to Grayscale",
                "Add Artistic Color (Simple)",
                "Add Artistic Color (AI-Powered)",
                "Detect Faces (Haar)",
                "Detect Faces (DNN)",
                "Zoom Image"
            ],
            help="Choose how to process your image"
        )
        
        # Add processing options help
        st.markdown("""
        #### Processing Modes:
        - **Enhance Image Quality**: Improves contrast, sharpness, and color balance
        - **Convert to Grayscale**: Creates a black and white version
        - **Add Artistic Color (Simple)**: Quick colorization using OpenCV
        - **Add Artistic Color (AI-Powered)**: High-quality colorization using deep learning
        - **Detect Faces (Haar)**: Fast face detection using Haar features
        - **Detect Faces (DNN)**: More accurate face detection using deep learning
        - **Zoom Image**: Enlarges the image while maintaining quality
        
        > Note: AI-powered colorization may take longer but provides more realistic results.
        """)

    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a file uploader widget with drag and drop
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload an image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )

    if uploaded_file is not None:
        try:
            # Load and validate the image
            with st.spinner("Loading image..."):
                image_array = load_and_validate_image(uploaded_file)
                # Convert numpy array to PIL Image for display
                display_image = Image.fromarray(image_array)
            
            # Display original image
            with col1:
                st.subheader("Original Image")
                st.image(display_image, caption="Original Image", use_container_width=True)

            # Process image with progress bar
            with col2:
                st.subheader("Processed Image")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate progress for better UX
                for i in range(4):
                    progress_bar.progress((i + 1) * 25)
                    status_text.text(f"Processing image... {(i + 1) * 25}%")
                    time.sleep(0.1)
                
                result = process_image(image_array, process_option)
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                
                # Display processed image
                st.image(result['image'], caption=result['caption'], use_container_width=True)
                
                # Add download button with success message
                buf = io.BytesIO()
                result['image'].save(buf, format="PNG")
                if st.download_button(
                    label=f"📥 Download {result['caption']}",
                    data=buf.getvalue(),
                    file_name=result['filename'],
                    mime="image/png"
                ):
                    display_success_message("Image downloaded successfully!")
                
                # Display additional information for face detection
                if process_option in ["Detect Faces (Haar)", "Detect Faces (DNN)"] and 'metadata' in result:
                    display_info_message(f"🔍 Number of faces detected: {result['metadata']['num_faces']}")

        except ValueError as ve:
            display_error_message(f"Validation Error: {str(ve)}")
        except RuntimeError as re:
            display_error_message(f"Processing Error: {str(re)}")
        except Exception as e:
            display_error_message(f"An unexpected error occurred: {str(e)}")
            display_error_message("Please try again with a different image or processing mode.")
    else:
        # Show example images and instructions when no file is uploaded
        with col2:
            display_info_message("Waiting for image upload...")
            st.markdown("""
            #### Tips:
            1. Select a processing mode from the sidebar
            2. Upload an image using the file uploader
            3. Wait for processing to complete
            4. Download the processed image
            """)

if __name__ == "__main__":
    main() 