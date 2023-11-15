# Real-time Face Detection, Age, and Gender Estimation

This project is a real-time face detection and analysis application using Flask, OpenCV, and pre-trained deep learning models to estimate the age and gender of detected faces in a live video stream from the webcam.

## Description

The project utilizes Flask, a Python web framework, to create a web application capable of streaming video frames captured by the user's webcam. The application performs the following tasks:

- **Face Detection:** Utilizes a pre-trained deep neural network model (`opencv_face_detector`) to detect faces in the video frames.
- **Age Estimation:** Applies another pre-trained model (`age_net`) to estimate the age range of detected faces.
- **Gender Classification:** Uses a pre-trained model (`gender_net`) to classify the gender of detected faces.

The project overlays the estimated age and gender information on the detected faces in real-time and streams the processed video frames to a web browser.

## Setup Instructions

To run the project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/chowdhuryriham/Age-_-Gender_Detection.git`
2. Install the necessary libraries and dependencies using `pip install -r requirements.txt`.
3. Download the pre-trained models (`opencv_face_detector`, `age_net`, `gender_net`) and place them in the project directory.
4. Run the Flask application: `main.py`.

Access the web application by navigating to `http://localhost:5000/` in your web browser.

## Requirements

The project requires the following libraries and tools:

- Python 3.x
- Flask
- OpenCV
- NumPy

## Credits

The face detection, age, and gender estimation models used in this project are based on the work and models available in the OpenCV library.

## Acknowledgments

Special thanks to the contributors and developers of the OpenCV library for providing the pre-trained models and resources necessary for this project.
