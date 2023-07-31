# EVCDataAnalysisProject-kullanaAmen

This repository contains the code for "كلنا آمن", a system for classifying reports into one of four categories: car accidents, crime, fire, and robbery. The system is designed to accept reports either as text input or as voice input in Arabic.

## Table of Contents

1. [Project Description](#project-description)
2. [Technologies](#technologies)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Team](#team)

## Project Description

A system that accepts user reports in Arabic and classifies them into one of four categories: car accidents, crime, fire, and robbery. The system uses a trained machine learning model to perform this classification. The system can accept input either as text or as voice recordings.

## Technologies

The project is implemented in Python and uses the following libraries and tools:

- **Streamlit:** for the web interface.
- **SpeechRecognition:** for transcribing audio input.
- **Googletrans:** for translating Arabic text to English.
- **NLTK:** for text preprocessing.
- **Scikit-learn and Joblib:** for machine learning model training and persistence.
- **Sounddevice and Wavio:** for handling audio input.

## Setup

To set up the project locally, follow these steps:

1. Clone the repository.
2. Install the required Python libraries, which are listed in the `requirements.txt` file.
3. Ensure that you have the trained model and vectorizer files (`kullanaAmn1.pkl` and `vectorizer.pkl`) in the root directory.

## Usage

You can run the system locally by navigating to the root directory of the project in a terminal and running `streamlit run app.py`.

The system presents two options for providing input: via text or voice recording. After providing input, press the "إرسال" button to classify the report. The system will display the classified category.

## Team

This project was developed by a team of dedicated developers:

- [Member 1](https://github.com/Member1)
- [Member 2](https://github.com/Member2)
- [Member 3](https://github.com/Member3)
- [Member 4](https://github.com/Member4)
