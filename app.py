
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

IMG_SIZE = 512
LAST_CONV_LAYER = "top_conv"
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

F1_SCORES = {
    'glioma': 0.97,
    'meningioma': 0.93,
    'notumor': 0.99,
    'pituitary': 0.96
}

OVERLAY_INTENSITY = 0.5
HEATMAP_THRESHOLD = 0.4


@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('tumor_cnn_f1_score_hospital_model_512.keras')


def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img_array, tf.float32)
        last_conv_layer_output, preds = grad_model(img_tensor)
        if isinstance(preds, list):
            preds = preds[0]
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def apply_green_overlay(orig_img, heatmap):
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))

    # Smooth the heatmap for highlight
    heatmap_blurred = cv2.GaussianBlur(heatmap_resized, (15, 15), 0)

    # Normalize after blur
    heatmap_blurred = heatmap_blurred / (heatmap_blurred.max() + 1e-8)

    mask = (heatmap_blurred > HEATMAP_THRESHOLD).astype(np.uint8)

    green_overlay = np.zeros_like(orig_img)
    green_overlay[:, :] = [0, 255, 0]

    overlayed = orig_img.copy()
    condition = mask > 0
    overlayed[condition] = cv2.addWeighted(
        orig_img[condition], 1 - OVERLAY_INTENSITY,
        green_overlay[condition], OVERLAY_INTENSITY, 0
    )
    return overlayed


st.markdown("""
    <style>
        /* Hide top-right toolbar (Deploy, Rerun, etc.) */
        header[data-testid="stHeader"] {
            display: none;
        }
        /* Remove the blank space left behind by hidden header */
        .block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Brain Tumor Detector", layout="wide")

st.title("🧠 Brain Tumor Detection Using CNN (EffiecientNetB0)")
st.write("Upload a brain MRI scan to detect tumor type with confidence score and highlighted region.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized)
    img_expanded = np.expand_dims(img_array, axis=0)
    prep_img = tf.keras.applications.efficientnet.preprocess_input(img_expanded.copy())

    if st.button("Analyze Scan"):
        with st.spinner("Running analysis..."):
            model = load_my_model()

            # Prediction
            preds = model.predict(prep_img)
            class_idx = np.argmax(preds[0])
            confidence = preds[0][class_idx] * 100
            predicted_label = CLASS_NAMES[class_idx]
            f1 = F1_SCORES.get(predicted_label.lower(), None)

            # Grad-CAM
            heatmap = get_gradcam_heatmap(prep_img, model, LAST_CONV_LAYER, pred_index=class_idx)

        st.markdown("---")

        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", predicted_label.title())
        m2.metric("Confidence", f"{confidence:.2f}%")
        if f1:
            m3.metric("Model F1 Score (this class)", f"{f1:.2f}")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original MRI Scan", width=400)

        with col2:
            if predicted_label.lower() != "notumor":
                green_img = apply_green_overlay(img_array, heatmap)
                st.image(green_img, caption="Detected Tumor Region", width=400)
                st.warning(f"Potential **{predicted_label.title()}** detected. Clinical review required.")
            else:
                st.image(image, caption="No Tumor Detected", width=400)
                st.success("No significant tumor signs detected by the model.")