import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
import os
import requests
import ssl
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Create unverified SSL context for downloading weights
ssl._create_default_https_context = ssl._create_unverified_context

class AdvancedColorizer(nn.Module):
    def __init__(self):
        super(AdvancedColorizer, self).__init__()
        # Use ResNet18 as the base model with weights enum
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Modify the first layer to accept grayscale input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = torch.sum(resnet.conv1.weight.data, dim=1, keepdim=True)
        
        # Extract features from ResNet
        self.features = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        return x

def get_advanced_colorizer():
    """
    Initialize and return the advanced colorization model.
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        model = AdvancedColorizer().to(device)
        model.eval()
        return model, device
    except Exception as e:
        print(f"Error initializing advanced colorization model: {str(e)}")
        return None, None

def colorize_with_advanced_model(image):
    """
    Colorize an image using the advanced ResNet-based model.
    """
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        original_size = img_array.shape[:2]  # Store original size
        
        # Ensure the image is grayscale
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Resize to a size that works with the model's architecture
        # The size should be divisible by 32 due to the 5 downsampling operations
        target_height = ((original_size[0] + 31) // 32) * 32
        target_width = ((original_size[1] + 31) // 32) * 32
        img_resized = cv2.resize(img_array, (target_width, target_height))
        
        # Normalize and prepare for model
        img_tensor = torch.from_numpy(img_resized).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Get model and device
        model, device = get_advanced_colorizer()
        if model is None:
            raise RuntimeError("Failed to initialize advanced colorization model")
        
        print("Model initialized successfully, processing image...")
        print(f"Input tensor shape: {img_tensor.shape}")
        
        # Process image
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            output = model(img_tensor)
            
        print(f"Output tensor shape: {output.shape}")
        
        # Convert output to RGB
        output = output.squeeze().cpu().numpy()
        output = output.transpose(1, 2, 0)
        output = (output * 128).astype(np.int8)
        
        # Create result array with resized dimensions
        result = np.zeros((target_height, target_width, 3))
        result[:, :, 0] = img_resized  # L channel
        result[:, :, 1:] = output  # ab channels
        
        # Convert to RGB
        result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        # Resize back to original dimensions
        result = cv2.resize(result, (original_size[1], original_size[0]))
        
        return result
        
    except Exception as e:
        print(f"Detailed error in advanced colorization: {str(e)}")
        raise RuntimeError(f"Error in advanced colorization: {str(e)}")

def colorize_image(gray_image):
    """
    Apply artistic colorization to a grayscale image using OpenCV's COLORMAP.
    This is a simpler alternative to neural network-based colorization.
    """
    try:
        # Convert PIL image to numpy array
        img_array = np.array(gray_image)
        
        # Ensure the image is grayscale
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply pseudo-colorization using OpenCV's COLORMAP
        colored = cv2.applyColorMap(img_array, cv2.COLORMAP_MAGMA)
        
        # Convert back to RGB for PIL
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        return colored_rgb
    except Exception as e:
        print(f"Error in simple colorization: {str(e)}")
        return np.array(gray_image)

def enhance_image(image):
    """
    Enhance image quality using various OpenCV techniques:
    - Contrast enhancement using CLAHE
    - Sharpening using kernel convolution
    - Noise reduction using Non-local Means Denoising
    - Color balance improvement using LAB color space
    """
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Ensure the image is in the correct format
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        
        # Convert to LAB color space for better color enhancement
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge the LAB channels back
        lab = cv2.merge((l, a, b))
        
        # Convert back to RGB color space
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Apply gentle sharpening
        kernel = np.array([[-0.5,-0.5,-0.5],
                         [-0.5, 5,-0.5],
                         [-0.5,-0.5,-0.5]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Ensure the output is in the correct range
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
        
    except Exception as e:
        # If any error occurs, try a simpler enhancement approach
        try:
            # Convert to array
            img_array = np.array(image)
            
            # Simple contrast enhancement
            enhanced = cv2.convertScaleAbs(img_array, alpha=1.2, beta=5)
            
            # Gentle sharpening
            kernel = np.array([[-0.5,-0.5,-0.5],
                             [-0.5, 5,-0.5],
                             [-0.5,-0.5,-0.5]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # Ensure the output is in the correct range
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            return enhanced
            
        except Exception as e2:
            # If all else fails, return the original image as a numpy array
            return np.array(image)

def detect_faces(image):
    """
    Detect faces in an image using Haar Cascade Classifier.
    Returns the image with detected faces marked and the number of faces found.
    """
    try:
        # Convert PIL image to numpy array if needed
        if not isinstance(image, np.ndarray):
            img_array = np.array(image)
        else:
            img_array = image.copy()
            
        # Convert to RGB if needed
        if len(img_array.shape) == 2:  # If grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
        # Convert to BGR for OpenCV processing
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Load the pre-trained face detection classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise Exception("Failed to load Haar cascade classifier")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            # Draw rectangle with rounded corners
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add text with background for better visibility
            text = 'Face Detected'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(img_bgr, 
                         (x, y - text_height - 10),
                         (x + text_width, y),
                         (0, 255, 0),
                         -1)
            
            # Draw text
            cv2.putText(img_bgr, text, (x, y - 5),
                       font, font_scale, (0, 0, 0), thickness)
        
        # Convert back to RGB
        result_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return result_img, len(faces)
        
    except Exception as e:
        print(f"Error in Haar face detection: {str(e)}")
        # Return the original image and 0 faces if something goes wrong
        if isinstance(image, np.ndarray):
            return image, 0
        return np.array(image), 0

def detect_faces_dnn(image):
    """
    Detect faces in an image using OpenCV's DNN face detector.
    This method is more accurate than Haar features but requires downloading the model.
    Returns the image with detected faces marked and the number of faces found.
    """
    try:
        # Convert PIL image to numpy array if needed
        if not isinstance(image, np.ndarray):
            img_array = np.array(image)
        else:
            img_array = image.copy()
            
        # Convert to RGB if needed
        if len(img_array.shape) == 2:  # If grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
        # Convert to BGR for OpenCV processing
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Get image dimensions
        height, width = img_bgr.shape[:2]
        
        # Load the pre-trained face detection model
        model_path = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
        config_path = "face_detection_model/deploy.prototxt"
        
        # Create model directory if it doesn't exist
        os.makedirs("face_detection_model", exist_ok=True)
        
        # Download model files if they don't exist
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            print("Downloading face detection model files...")
            
            # Download the model file
            model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            config_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            
            try:
                # Download model
                response = requests.get(model_url, timeout=30)
                if response.status_code == 200:
                    with open(model_path, 'wb') as f:
                        f.write(response.content)
                    print("Successfully downloaded face detection model")
                else:
                    raise Exception("Failed to download model file")
                
                # Download config
                response = requests.get(config_url, timeout=30)
                if response.status_code == 200:
                    with open(config_path, 'wb') as f:
                        f.write(response.content)
                    print("Successfully downloaded model configuration")
                else:
                    raise Exception("Failed to download config file")
                    
            except Exception as e:
                print(f"Error downloading model files: {str(e)}")
                raise Exception("Failed to download face detection model files")
        
        # Load the DNN model
        net = cv2.dnn.readNet(model_path, config_path)
        if net.empty():
            raise Exception("Failed to load DNN model")
        
        # Prepare image for the model
        blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (300, 300), [104, 117, 123], False, False)
        
        # Set input for the model
        net.setInput(blob)
        
        # Forward pass
        detections = net.forward()
        
        # Process detections
        faces = []
        confidences = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter detections by confidence
            if confidence > 0.5:  # Confidence threshold
                # Get bounding box coordinates
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                faces.append((x1, y1, x2-x1, y2-y1))
                confidences.append(confidence)
        
        # Draw rectangles around detected faces
        for (x, y, w, h), conf in zip(faces, confidences):
            # Draw rectangle with rounded corners
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add text with confidence score
            text = f'Face ({conf:.2f})'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(img_bgr, 
                         (x, y - text_height - 10),
                         (x + text_width, y),
                         (0, 255, 0),
                         -1)
            
            # Draw text
            cv2.putText(img_bgr, text, (x, y - 5),
                       font, font_scale, (0, 0, 0), thickness)
        
        # Convert back to RGB
        result_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return result_img, len(faces)
        
    except Exception as e:
        print(f"Error in DNN face detection: {str(e)}")
        # Return the original image and 0 faces if something goes wrong
        if isinstance(image, np.ndarray):
            return image, 0
        return np.array(image), 0

def download_age_model():
    """
    Download pre-trained age estimation model files if they don't exist locally.
    """
    model_dir = "age_model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Define the content of the prototxt file
    prototxt_content = '''
name: "AgeNet"
input: "data"
input_dim: 1
input_dim: 3
input_dim: 227
input_dim: 227

layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: 96
    kernel_size: 7
    stride: 4
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "norm1"
  type: "LRN"
  bottom: "pool1"
  top: "norm1"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "fc8"
  type: "InnerProduct"
  bottom: "norm1"
  top: "fc8"
  inner_product_param {
    num_output: 8
  }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "fc8"
  top: "prob"
}
'''
    
    # Write the prototxt file
    prototxt_path = os.path.join(model_dir, "age_deploy.prototxt")
    if not os.path.exists(prototxt_path):
        try:
            with open(prototxt_path, 'w') as f:
                f.write(prototxt_content.strip())
            print("Successfully created age model architecture file")
        except Exception as e:
            print(f"Error creating prototxt file: {str(e)}")
            return False
    
    # Download the caffemodel file
    caffemodel_path = os.path.join(model_dir, "age_net.caffemodel")
    if not os.path.exists(caffemodel_path):
        print("Downloading age model weights...")
        urls = [
            "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel",
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/age_net.caffemodel",
            "https://drive.google.com/uc?export=download&id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW"
        ]
        
        success = False
        for url in urls:
            try:
                print(f"Trying to download from: {url}")
                response = requests.get(url, timeout=30)
                if response.status_code == 200 and len(response.content) > 1000000:  # Ensure file is at least 1MB
                    with open(caffemodel_path, 'wb') as f:
                        f.write(response.content)
                    print("Successfully downloaded age model weights")
                    success = True
                    break
                else:
                    print(f"Invalid response from {url}")
            except Exception as e:
                print(f"Failed to download from {url}: {str(e)}")
                continue
        
        if not success:
            print("Failed to download age model weights from all sources")
            return False
    
    # Verify both files exist and are valid
    if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
        print("Missing required model files")
        return False
    
    if os.path.getsize(caffemodel_path) < 1000000:  # Check if caffemodel is at least 1MB
        print("Downloaded caffemodel file appears to be invalid")
        os.remove(caffemodel_path)  # Remove invalid file
        return False
        
    print("Age model files are ready")
    return True

def detect_faces_and_age(image):
    """
    Detect faces in an image, estimate their ages, and draw rectangles with age labels.
    Returns the image with detected faces marked and a list of detected ages.
    """
    try:
        # Convert PIL image to numpy array if needed
        if not isinstance(image, np.ndarray):
            img_array = np.array(image)
        else:
            img_array = image.copy()
            
        # Convert to RGB if needed
        if len(img_array.shape) == 2:  # If grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
        # Convert to BGR for OpenCV processing
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Load the face detection classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise Exception("Failed to load face detection model")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            print("No faces detected in the image")
            return img_array, 0, []
        
        # Load age detection model
        model_dir = "age_model"
        prototxt = os.path.join(model_dir, "age_deploy.prototxt")
        weights = os.path.join(model_dir, "age_net.caffemodel")
        
        # Check if model files exist, if not download them
        if not os.path.exists(prototxt) or not os.path.exists(weights):
            print("Age model files missing, attempting to download...")
            if not download_age_model():
                raise Exception("Failed to download age detection model files")
        
        # Load the age detection model
        print("Loading age detection model...")
        age_model = cv2.dnn.readNet(prototxt, weights)
        if age_model.empty():
            raise Exception("Failed to load age detection model")
            
        age_ranges = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']
        detected_ages = []
        
        print(f"Processing {len(faces)} detected faces...")
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            try:
                # Extract and preprocess face region for age detection
                face_img = img_bgr[y:y+h, x:x+w]
                # Ensure face region is not empty
                if face_img.size == 0:
                    print(f"Empty face region detected at coordinates ({x}, {y}, {w}, {h})")
                    continue
                
                # Resize face image to required input size
                face_blob = cv2.dnn.blobFromImage(
                    face_img, 1.0, (227, 227),
                    (78.4263377603, 87.7689143744, 114.895847746),
                    swapRB=False
                )
                
                # Predict age
                age_model.setInput(face_blob)
                age_preds = age_model.forward()
                age_index = age_preds[0].argmax()
                age_range = age_ranges[age_index]
                detected_ages.append(age_range)
                
                print(f"Detected age range for face at ({x}, {y}): {age_range}")
                
                # Add age label next to the face box
                label = f"Age: {age_range}"
                
                # Calculate text size and position
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Position text next to the right edge of the bounding box
                text_x = x + w + 10  # 10 pixels padding from the box
                text_y = y + h // 2  # Vertically centered with the box
                
                # Draw white background for better visibility
                cv2.rectangle(img_bgr, 
                            (text_x - 2, text_y - text_height - 2),
                            (text_x + text_width + 2, text_y + 2),
                            (255, 255, 255),
                            -1)
                
                # Draw the text
                cv2.putText(img_bgr, label, (text_x, text_y),
                          font, font_scale, (0, 128, 0), thickness)
                
            except Exception as e:
                print(f"Error processing face for age detection: {str(e)}")
                # Add "Face Detected" label if age estimation fails
                cv2.putText(img_bgr, "Face Detected", (x + w + 10, y + h // 2),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Convert back to RGB
        result_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return result_img, len(faces), detected_ages
        
    except Exception as e:
        print(f"Error in face and age detection: {str(e)}")
        # Return the original image and no detections if something goes wrong
        if isinstance(image, np.ndarray):
            return image, 0, []
        return np.array(image), 0, [] 