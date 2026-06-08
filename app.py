import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import io
import csv
import gdown
import zipfile

# Page configuration
st.set_page_config(page_title="Songket Motif Classifier", layout="wide", initial_sidebar_state="expanded")

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = Path(__file__).parent
class_names = ['bunga_pecah_lapan', 'pucuk_rebung', 'tampuk_manggis']

# Google Drive model IDs (fill in your Google Drive file IDs)
GDRIVE_MODEL_IDS = {
    "resnet50": "1bmb5_rooHxAyF-qHOr2GoUL84kyzG-v9",
    "vgg19": "1eTmSKPzIo5bgQUrnSXi9PR6tROTQArtT",
    "googlenet": "1z1plZFq-6i7zKPiZTpHeezgjXXZoF69S",
    "alexnet": "1RRqdUOwn6wW7nveXerXDqdTPAiM1lvKF"
}

# Preprocessing transform (must match training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model(model_name):
    """Load a trained model. Downloads from Google Drive if not found locally."""
    model_path = MODELS_DIR / f"songket_motif_{model_name}_final.pth"
    zip_path = MODELS_DIR / f"songket_motif_{model_name}_final.zip"

    # Download from Google Drive if model doesn't exist locally
    if not model_path.exists():
        gdrive_id = GDRIVE_MODEL_IDS.get(model_name)
        if gdrive_id and gdrive_id != "YOUR_" + model_name.upper() + "_GDRIVE_ID":
            try:
                st.info(f"📥 Downloading {model_name} model from Google Drive...")
                # Try downloading as ZIP first, then as .pth
                if gdrive_id.endswith("_zip"):
                    gdown.download(f"https://drive.google.com/uc?id={gdrive_id[:-4]}", str(zip_path), quiet=False)
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(MODELS_DIR)
                    zip_path.unlink()  # Delete ZIP after extraction
                    st.success(f"✅ Extracted {model_name} model successfully!")
                else:
                    gdown.download(f"https://drive.google.com/uc?id={gdrive_id}", str(model_path), quiet=False)
                    st.success(f"✅ Downloaded {model_name} model successfully!")
            except Exception as e:
                st.error(f"❌ Failed to download {model_name}: {str(e)}")
                return None
        else:
            st.error(f"❌ Model not found: {model_path}")
            st.error("⚠️ Please upload model files or configure Google Drive IDs")
            return None

    try:
        if model_name == "resnet50":
            model = models.resnet50()
            model.fc = nn.Linear(model.fc.in_features, len(class_names))

        elif model_name == "vgg19":
            model = models.vgg19_bn()  # Must use vgg19_bn (Batch Normalization version)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(class_names))

        elif model_name == "googlenet":
            model = models.googlenet()
            model.fc = nn.Linear(model.fc.in_features, len(class_names))

        elif model_name == "alexnet":
            model = models.alexnet()
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(class_names))

        else:
            st.error(f"❌ Unknown model: {model_name}")
            return None

        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model.to(device)
        model.eval()
        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_data
def load_test_dataset():
    """Load test dataset for evaluation."""
    test_dir = MODELS_DIR / "dataset" / "final_split" / "test"
    if not test_dir.exists():
        return None

    test_dataset = datasets.ImageFolder(test_dir, preprocess)
    return test_dataset

def predict_motif(image, model):
    """Predict motif class and get probabilities."""
    img = image.convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    confidence, predicted_idx = torch.max(probabilities, 0)
    probabilities_np = probabilities.cpu().numpy()

    return {
        'predicted_class': class_names[predicted_idx.item()],
        'predicted_idx': predicted_idx.item(),
        'confidence': confidence.item(),
        'probabilities': probabilities_np
    }

def evaluate_model_on_test_set(model, test_dataset):
    """Evaluate model on test dataset and return metrics."""
    all_preds = []
    all_labels = []

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # Per-class metrics
    per_class_report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'per_class_report': per_class_report
    }

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")
st.sidebar.info(
    "📋 **About**\n"
    "This app classifies songket motifs into three categories:\n"
    "- 🌸 Bunga Pecah Lapan\n"
    "- 🌿 Pucuk Rebung\n"
    "- 🥭 Tampuk Manggis"
)

# Main content
st.title("🎨 Songket Motif Classification")
st.markdown("Advanced classification system with performance metrics, model comparison, and batch processing.")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Classify Image", "Model Comparison", "Batch Processing", "Model Info"])

# ==================== TAB 1: CLASSIFY IMAGE ====================
with tab1:
    st.subheader("📤 Single Image Classification")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Upload Image**")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp"],
            key="single_upload",
            help="Upload a songket motif image"
        )

    with col2:
        st.markdown("**Select Model**")
        selected_model = st.selectbox(
            "Choose model",
            ["resnet50", "vgg19", "googlenet", "alexnet"],
            key="single_model"
        )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        model = load_model(selected_model)

        if model:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, use_column_width=True, caption="Uploaded Image")

            with col2:
                st.subheader("🎯 Classification Result")
                result = predict_motif(image, model)
                confidence_pct = result['confidence'] * 100

                st.metric(
                    "Predicted Motif",
                    result['predicted_class'].upper().replace('_', ' '),
                    f"{confidence_pct:.2f}% confidence"
                )

                st.subheader("📊 Confidence Breakdown")
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ['#2ecc71' if i == result['predicted_idx'] else '#95a5a6' for i in range(len(class_names))]
                bars = ax.barh(class_names, result['probabilities'], color=colors)
                ax.set_xlabel("Confidence", fontsize=12)
                ax.set_title("Classification Confidence by Class", fontsize=14, fontweight='bold')
                ax.set_xlim(0, 1)

                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2,
                           f'{result["probabilities"][i]*100:.1f}%',
                           ha='left', va='center', fontsize=10, fontweight='bold')

                plt.tight_layout()
                st.pyplot(fig)

# ==================== TAB 2: MODEL COMPARISON ====================
with tab2:
    st.subheader("⚖️ Compare All Models")
    st.markdown("Upload an image and see predictions from all 4 models side-by-side.")

    uploaded_file_compare = st.file_uploader(
        "Choose an image file for comparison",
        type=["jpg", "jpeg", "png", "bmp"],
        key="compare_upload",
        help="Upload a songket motif image"
    )

    if uploaded_file_compare is not None:
        image_compare = Image.open(uploaded_file_compare)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image_compare, use_column_width=True, caption="Image for Comparison")

        with col2:
            st.markdown("**Model Predictions**")

            # Get predictions from all models
            models_list = ["resnet50", "vgg19", "googlenet", "alexnet"]
            all_results = {}

            for model_name in models_list:
                model = load_model(model_name)
                if model:
                    all_results[model_name] = predict_motif(image_compare, model)

        # Display results as table
        st.markdown("**Comparison Table**")
        comparison_data = []
        for model_name, result in all_results.items():
            comparison_data.append({
                'Model': model_name.upper(),
                'Predicted Class': result['predicted_class'].replace('_', ' ').title(),
                'Confidence': f"{result['confidence']*100:.2f}%",
                'Class 1': f"{result['probabilities'][0]*100:.1f}%",
                'Class 2': f"{result['probabilities'][1]*100:.1f}%",
                'Class 3': f"{result['probabilities'][2]*100:.1f}%"
            })

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

        # Visualization: Side-by-side confidence
        st.markdown("**Confidence Comparison**")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, (model_name, result) in enumerate(all_results.items()):
            colors = ['#2ecc71' if i == result['predicted_idx'] else '#95a5a6' for i in range(len(class_names))]
            axes[idx].barh(class_names, result['probabilities'], color=colors)
            axes[idx].set_xlabel("Confidence")
            axes[idx].set_title(f"{model_name.upper()}", fontweight='bold')
            axes[idx].set_xlim(0, 1)

        plt.tight_layout()
        st.pyplot(fig)

# ==================== TAB 3: BATCH PROCESSING ====================
with tab3:
    st.subheader("📦 Batch Image Processing")
    st.markdown("Upload multiple images and classify them all at once.")

    batch_model = st.selectbox(
        "Select model for batch processing",
        ["resnet50", "vgg19", "googlenet", "alexnet"],
        key="batch_model"
    )

    uploaded_files = st.file_uploader(
        "Choose image files",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="batch_upload"
    )

    if uploaded_files and len(uploaded_files) > 0:
        model = load_model(batch_model)

        if model:
            st.markdown(f"**Processing {len(uploaded_files)} images with {batch_model.upper()}...**")

            batch_results = []
            progress_bar = st.progress(0)

            for idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                result = predict_motif(image, model)

                batch_results.append({
                    'Filename': uploaded_file.name,
                    'Predicted Class': result['predicted_class'].replace('_', ' ').title(),
                    'Confidence': f"{result['confidence']*100:.2f}%",
                    'Bunga Pecah Lapan': f"{result['probabilities'][0]*100:.2f}%",
                    'Pucuk Rebung': f"{result['probabilities'][1]*100:.2f}%",
                    'Tampuk Manggis': f"{result['probabilities'][2]*100:.2f}%"
                })

                progress_bar.progress((idx + 1) / len(uploaded_files))

            # Display results table
            st.markdown("**Classification Results**")
            results_df = pd.DataFrame(batch_results)
            st.dataframe(results_df, use_container_width=True)

            # Download as CSV
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"songket_classification_{batch_model}.csv",
                mime="text/csv"
            )

            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)

            # Extract numeric confidence values for statistics
            confidences = [float(r['Confidence'].rstrip('%')) for r in batch_results]

            with col1:
                st.metric("Total Images", len(batch_results))
            with col2:
                st.metric("Avg Confidence", f"{np.mean(confidences):.2f}%")
            with col3:
                st.metric("Min Confidence", f"{np.min(confidences):.2f}%")
            with col4:
                st.metric("Max Confidence", f"{np.max(confidences):.2f}%")

# ==================== TAB 4: MODEL INFO ====================
with tab4:
    st.subheader("📋 Model Information")

    selected_model_info = st.selectbox(
        "Select model",
        ["resnet50", "vgg19", "googlenet", "alexnet"],
        key="info_model"
    )

    model_info = {
        "resnet50": {
            "description": "ResNet50 is a 50-layer deep residual network with skip connections.",
            "advantages": ["Fast inference", "Good accuracy", "Stable training"],
            "parameters": "~23.5M"
        },
        "vgg19": {
            "description": "VGG19-BN is a 19-layer very deep convolutional network with batch normalization.",
            "advantages": ["Good performance", "Simple architecture", "Stable"],
            "parameters": "~139M"
        },
        "googlenet": {
            "description": "GoogLeNet (Inception) uses multi-scale convolutional modules.",
            "advantages": ["Efficient", "Multi-scale features", "Good for small objects"],
            "parameters": "~7M"
        },
        "alexnet": {
            "description": "AlexNet is a classic deep CNN with 8 layers.",
            "advantages": ["Lightweight", "Fast inference", "Good baseline"],
            "parameters": "~60M"
        }
    }

    info = model_info[selected_model_info]

    st.write(f"**Description:** {info['description']}")
    st.write(f"**Parameters:** {info['parameters']}")
    st.write("**Advantages:**")
    for advantage in info['advantages']:
        st.write(f"  • {advantage}")

    st.subheader("🏷️ Classes")
    for i, class_name in enumerate(class_names):
        st.write(f"{i+1}. {class_name.replace('_', ' ').title()}")

    st.subheader("⚙️ Preprocessing")
    st.code("""
- Input Size: 224 x 224
- Normalization: ImageNet
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- Model: Pre-trained on ImageNet
- Training: Fine-tuned on songket motifs
    """)

st.markdown("---")
st.caption("🎨 Songket Motif Classifier | Built with Streamlit | v2.0 with Advanced Features")
