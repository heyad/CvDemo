# Advanced Image Processing App

A Streamlit-based web application for advanced image processing, featuring:

- 🎨 AI-powered image colorization using ResNet architecture
- 🖼️ Simple colorization using OpenCV
- ✨ Image enhancement
- 👤 Face detection
- 🔄 Grayscale conversion

## Features

- Modern, user-friendly interface built with Streamlit
- Multiple processing options for different use cases
- Real-time image processing
- Easy-to-use file upload and download functionality

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
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

## License

MIT License

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── colorization_model.py   # Image processing and colorization logic
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Dependencies

- Python 3.6+
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

- OpenCV for providing the colorization algorithms
- Streamlit for the excellent web framework
- The open-source community for various image processing tools and libraries

## Future Improvements

- Add more colorization algorithms
- Implement neural network-based colorization
- Add batch processing capabilities
- Improve error handling and user feedback
- Add image preprocessing options 