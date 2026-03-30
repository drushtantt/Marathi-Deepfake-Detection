# Marathi Deepfake Detection

A Marathi audio deepfake detection project with a browser-based frontend and a Flask inference backend powered by a CNN model.

## Overview

This project focuses on classifying Marathi speech as human or synthetic. The repository now includes:

- a research-oriented model and preprocessing codebase
- a frontend website for uploading audio and viewing predictions
- a Flask backend that extracts mel-spectrogram features and runs inference with `CNN_model.h5`
- a diagnostic script for checking CNN preprocessing behavior

The current web app accepts audio uploads, previews the waveform and feature map in the browser, and sends the file to the backend for prediction.

## Dataset

The original project was built on **28,863 Marathi audio clips**:

- `16,022` human speech samples from Mozilla Common Voice
- `12,841` synthesized speech samples generated using Amazon Polly and gTTS

Dataset link:

- [Google Drive dataset](https://drive.google.com/drive/folders/1j-O-AkqNofQRcuvbkfIqUMbfdVROZQrM?usp=drive_link)

## Models Included

- `CNN_model.h5`
- `SVM_model.joblib`
- `GNN_model.pth`

The live website currently uses the CNN model through the Flask backend in [app.py](app.py).

## Web App

The website includes:

- upload and drag-and-drop audio input
- browser-side waveform preview
- spectrogram-style visual preview
- live CNN inference through `/predict`
- result cards with class probabilities and a human vs synthetic verdict

Main frontend files:

- [index.html](index.html)
- [styles.css](styles.css)
- [script.js](script.js)

## Backend

The Flask backend:

- serves the frontend files
- accepts uploaded audio files at `/predict`
- extracts a `128 x 128` mel-spectrogram using Librosa
- loads `CNN_model.h5`
- returns predicted label, confidence, and per-class probabilities

Main backend file:

- [app.py](app.py)

## Installation

Use Python 3.11 or a compatible version.

```bash
pip install -r requirements.txt
```

## Run Locally

Start the Flask server:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Diagnostic Script

To test how different mel-spectrogram preprocessing variants affect CNN predictions:

```bash
python diagnose_cnn.py
```

This is useful when the saved model appears overly biased toward a single class.

## Project Structure

```text
.
|-- app.py
|-- diagnose_cnn.py
|-- index.html
|-- styles.css
|-- script.js
|-- requirements.txt
|-- CNN_model.h5
|-- SVM_model.joblib
|-- GNN_model.pth
|-- audio_to_melspectogram.py
|-- audio_to_MFCC.py
|-- conversion_audio _to_graph.py
|-- text_to_speech_amazon.py
|-- text_to_speech_gtts.py
```

## Notes

- The frontend and backend are working together for live inference.
- The CNN prediction quality still depends on the saved checkpoint and original training setup.
- The class-name order for the CNN output is inferred and may need adjustment if the original training notebook used a different label mapping.

## Future Improvements

- verify class order from the original training pipeline
- expose raw confidence history in the frontend
- add support for SVM and GNN inference endpoints
- deploy the Flask app publicly

## License

Add a license file if you want to make repository usage terms explicit on GitHub.
