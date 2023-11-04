# Sign Language Recognition Project

## Project Overview

The Sign Language Recognition project is designed to recognize and interpret sign language gestures using hand landmarks captured via computer vision. It combines machine learning techniques with real-time hand tracking to convert hand movements into text, making sign language more accessible and inclusive for communication.

## Key Features

- **Real-time Hand Tracking**: Utilizes the MediaPipe library to track and analyze hand movements captured through a webcam.

- **Data Collection**: Allows users to contribute to data collection by using the "Sign Language CSV Dataset Collector" application. This tool records landmark points and their components for building a robust sign language dataset.

- **SVM Model**: Employs a Support Vector Machine (SVM) model for sign language recognition. The model is trained on a dataset of hand landmarks and labels, enabling it to classify hand signs.

- **User Interface**: Incorporates a user-friendly interface using Streamlit, allowing users to interact with the model and view recognition results in real-time.

## Getting Started

### Prerequisites

- [Python](https://www.python.org/) (Python 3.6 or higher)
- [Pip](https://pip.pypa.io/en/stable/installation/)

### Installation

1. Clone the repository to your local machine.

   ```bash
   git clone https://github.com/yourusername/sign-language-recognition.git
   ```

2. Navigate to the project directory.

   ```bash
   cd sign-language-recognition
   ```

3. Install the required Python packages.

   ```bash
   pip install -r requirements.txt
   ```

4. Run the project.

   ```bash
   streamlit run main.py
   ```

## Usage

1. Start the project by running the Streamlit application.
2. Use the "Sign Language CSV Dataset Collector" page to collect data and create a dataset.
3. Interact with the SVM model for real-time sign language recognition.
4. Explore the features and functionalities provided by the project.

## Roadmap

Future enhancements for the Sign Language Recognition project include:

- Extending Gesture Vocabulary: Expanding the dataset and model to recognize a wider range of sign language gestures and expressions.
- Speech Output: Implementing a speech synthesis feature to audibly interpret sign language gestures.
- Mobile Application: Developing a mobile application for on-the-go sign language recognition.
- Multilingual Support: Adding support for various sign languages and languages for improved accessibility on a global scale.
