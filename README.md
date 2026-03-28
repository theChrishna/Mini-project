# AI Sign Language & Gesture Translator

An advanced, real-time hand gesture recognition and sign language translation web application. This project uses your webcam to track hand movements via **MediaPipe**, processes spatial coordinates natively, and translates gestures instantly using a robust **Deep Learning Neural Network (MLP)** pipeline built with scikit-learn.

## Project Structure

- **`hand_tracking.py`**: A computer-vision handler wrapping the MediaPipe Tasks API. It extracts 21 precise 3D hand landmarks, normalizes them, and actively visualizes bounding boxes and tracking data on the live frame.
- **`collect_all_signs.py`**: An automated mass data-collection script designed to sequentially prompt and record hundreds of high-quality dataset frames for every single letter from A to Z, ensuring phenomenal model accuracy.
- **`data_collection.py`**: A manual script for targeted gesture recording and data generation.
- **`train_model.py` / `check_model.py`**: The Artificial Intelligence engine. Reads your custom dataset, auto-scales the features using `StandardScaler`, and trains a highly performant **Deep Neural Network (`MLPClassifier`)** for complex gesture recognition. Saves the compiled network to `model.p`.
- **`app.py`**: The core Flask backend server. It handles real-time live webcam processing, authenticates users, serves the interactive chat APIs, and broadcasts telemetry to the frontend.
- **`templates/` & `static/`**: Houses the premium Dark/Orange glassmorphism UI, Audio Synthesizers, and the secure Authentication Portal.

## Prerequisites

Ensure you have Python 3.8+ installed on your system. You can install all required dependencies using `pip`:

```bash
pip install opencv-python mediapipe pandas scikit-learn numpy flask google-generativeai yt-dlp
```

> **Note:** Set your Google Gemini API Key in your terminal (`$env:GEMINI_API_KEY="YOUR_KEY"`) before running the app to enable the live chatbot!

## How to use

### 1. Generate Your Dataset
To teach the AI, you must collect data for the gestures you want it to know. The easiest way is to run the automated mass-collector:
```bash
python collect_all_signs.py
```
Follow the on-screen prompts to record massive arrays of data for all 26 letters of the alphabet automatically.

### 2. Train the Deep Learning Model
Once the massive dataset (`hand_data.csv`) is generated, train your Neural Network:
```bash
python train_model.py
```
This builds and tests the `MLPClassifier` pipeline and outputs the final mathematical model to `model.p`.

### 3. Run the Live Web Portal
Boot up the secure local Flask server:
```bash
python app.py
```
Open `http://localhost:5000` in your web browser. You will be greeted by the **AI Security Portal**. Use the interactive Google Auth or Mobile OTP simulators to securely log into the system, and the AI will begin tracking and translating your sign language in real time!

## Latest Advanced Upgrades
- **Deep Learning Upgrade:** Replaced the legacy Random Forest algorithm with a powerful Deep Neural Network (MLP) Pipeline for drastically improved prediction accuracy.
- **Mass Dataset Collector:** Introduced `collect_all_signs.py` to recursively collect training data for the entire alphabet natively.
- **AI Authentication Portal:** Access is fully protected by a custom-styled Login interface natively supporting interactive Google Auth and Mobile SMS verification workflows.
- **Gemini Chat Hub:** The community interaction board is directly wired into the `gemini-1.5-flash` LLM, allowing you to converse with the AI in real time right from the web panel!
- **Web Audio TTS & Synth:** Powered by native browser audio contexts, the app visually synthesizes the Harry Potter Theme Song mathematically in the background, features interactive UI "bloop" sounds, and includes a **Voice Assistant** that reads translated gestures out loud.
- **Premium UX Redesign:** Features an entirely new Black & Vibrant Orange Glassmorphism UI, tracking bounding-boxes, floating background particle orbs, smooth CSS transitions, and an integrated FPS telemetry feed.