# Computer Vision Demo

A comprehensive demonstration of computer vision capabilities using Python, featuring:

- 🎨 AI-powered image colorization using ResNet architecture
- 🖼️ Simple colorization using OpenCV
- ✨ Image enhancement and processing
- 👤 Face detection with age estimation
- 🔄 Grayscale conversion

## Features

- Modern, user-friendly interface built with Streamlit
- Multiple processing options for different computer vision tasks
- Real-time image processing and analysis
- Easy-to-use file upload and download functionality
- Advanced AI-powered colorization using deep learning

## Installation

1. Clone the repository:
```bash
git clone https://github.com/heyad/CvDemo.git
cd CvDemo
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Requirements

- Python 3.8+
- See `requirements.txt` for full list of dependencies

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── colorization_model.py   # Computer vision and deep learning models
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Dependencies

- Python 3.8+
- streamlit==1.42.2
- pillow==11.1.0
- numpy==1.26.4
- opencv-python-headless==4.8.1.78
- torch==2.2.0
- torchvision==0.17.0

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Acknowledgments

- OpenCV for computer vision algorithms
- PyTorch for deep learning capabilities
- Streamlit for the web interface
- The open-source community for various CV tools and libraries

## License

MIT License 