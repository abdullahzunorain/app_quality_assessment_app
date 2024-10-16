# Apple Quality Assessment App

## Overview

The Apple Quality Assessment App is a web application built using Streamlit that allows users to upload images of apple leaves and assess their health condition. The application leverages a pre-trained image classification model from Hugging Face's model hub to identify various leaf diseases and determine whether the apple leaf is healthy or infected. This project aims to facilitate farmers and agricultural enthusiasts in quickly diagnosing leaf conditions, thus promoting better agricultural practices.

## Features

- **User-Friendly Interface**: The app provides a simple and intuitive interface for users to upload images of apple leaves.
- **Real-Time Image Classification**: Users can upload images, and the app will analyze them in real-time to determine the condition of the leaf.
- **Disease Detection**: The app can identify various conditions affecting apple leaves, including:
  - Alternaria leaf spot
  - Brown spot
  - Frogeye leaf spot
  - Grey spot
  - Healthy
  - Mosaic
  - Powdery mildew
  - Rust
  - Scab
- **Visual Feedback**: After analysis, the app displays the uploaded image alongside the predicted condition, providing clear visual feedback.

## Technologies Used

- **Streamlit**: A powerful and easy-to-use framework for building data applications.
- **Hugging Face Transformers**: Utilized to load and use pre-trained deep learning models for image classification.
- **PIL (Python Imaging Library)**: For image processing and manipulation.

## Installation

To run this application locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abdullahzunorain/apple_quality_assessment_app.git
   cd apple_quality_assessment_app
   ```

2. **Install the required packages**:
   Create a virtual environment (recommended) and install the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   Start the Streamlit application using the following command:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Navigate to the application in your web browser.
2. Click on the "Upload an apple leaf image" button to upload your image file (supports JPG, JPEG, PNG formats).
3. The app will display the uploaded image and analyze it to determine its condition.
4. The predicted condition will be shown below the image after the analysis is complete.

## Contributing

Contributions to enhance the functionality of this app are welcome! Please feel free to fork the repository and submit pull requests with improvements, new features, or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Thanks to [Hugging Face](https://huggingface.co/) for providing a rich ecosystem of pre-trained models.
- Special thanks to the open-source community for their contributions to the development of tools like Streamlit.

---
