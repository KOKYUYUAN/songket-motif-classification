# songket-motif-classification
Final Year Project: Songket Motif Classification using Deep Learning.

## UI Demo for Automatic Classification

This project now includes a user interface to demonstrate the best model
inference workflow for Songket motif classification.

- UI app: songket_ui_app.py
- Best-model inference module: best_model_inference.py
- Best checkpoint used: songket_motif_alexnet_final.pth

## Run the UI

1. Install dependencies:

	pip install -r requirements.txt

2. Start the app:

	streamlit run songket_ui_app.py

3. Upload an image and click Run Classification.

The app displays:

- Predicted motif class
- Confidence score
- Inference time in milliseconds
- Top-class probability chart
- Guidance on whether auto-tagging is suitable or should be human-reviewed
