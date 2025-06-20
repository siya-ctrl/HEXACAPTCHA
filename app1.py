import streamlit as st
import cv2
import numpy as np
import string
import math
from PIL import Image
import tensorflow as tf
import pandas as pd
import io
import base64
import time
import plotly.express as px

# Custom CSS for styling and accessibility
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        color: #1e3a8a;
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        color: #374151;
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
    }
    .success-box {
        background-color: #d1fae5;
        padding: 10px;
        border-radius: 5px;
        color: #065f46;
    }
    .error-box {
        background-color: #fee2e2;
        padding: 10px;
        border-radius: 5px;
        color: #991b1b;
    }
    .result-box {
        background-color: #e5e7eb;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #1e40af;
    }
    .zoom-container {
        position: relative;
        overflow: hidden;
    }
    .zoom-image {
        transition: transform 0.2s;
    }
    .zoom-image:hover {
        transform: scale(1.5);
    }
    </style>
    <script>
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && document.activeElement.id === 'captcha_input') {
            document.querySelector('button[kind="primary"]').click();
        }
        if (event.ctrlKey && event.key === 'r') {
            event.preventDefault();
            document.querySelector('button[kind="secondary"]').click();
        }
    });
    </script>
""", unsafe_allow_html=True)

# Function to check CAPTCHA strength with breakdown
def check_captcha_strength(captcha_text, image):
    score = 0
    breakdown = {}

    # Check Length
    length = len(captcha_text)
    if length >= 6:
        score += 2
        breakdown["Length"] = "2 (6+ characters)"
    elif length >= 4:
        score += 1
        breakdown["Length"] = "1 (4-5 characters)"
    else:
        breakdown["Length"] = "0 (<4 characters)"
    
    # Check Character Variety
    has_upper = any(c.isupper() for c in captcha_text)
    has_digit = any(c.isdigit() for c in captcha_text)
    has_hex = all(c in string.hexdigits for c in captcha_text.lower())

    if has_upper:
        score += 1
        breakdown["Uppercase"] = "1 (Contains uppercase)"
    else:
        breakdown["Uppercase"] = "0 (No uppercase)"
    if has_digit:
        score += 1
        breakdown["Digits"] = "1 (Contains digits)"
    else:
        breakdown["Digits"] = "0 (No digits)"
    if has_hex:
        score += 1
        breakdown["Hexadecimal"] = "1 (All hex digits)"
    else:
        breakdown["Hexadecimal"] = "0 (Non-hex digits)"

    # Check Image Complexity (Noise)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])

    if edge_density > 0.1:
        score += 2
        breakdown["Image Complexity"] = "2 (High noise)"
    elif edge_density > 0.05:
        score += 1
        breakdown["Image Complexity"] = "1 (Moderate noise)"
    else:
        breakdown["Image Complexity"] = "0 (Low noise)"

    # Calculate Entropy
    char_frequencies = {char: captcha_text.count(char) for char in set(captcha_text)}
    entropy = -sum((freq/length) * math.log2(freq/length) for freq in char_frequencies.values())

    if entropy > 2.5:
        score += 2
        breakdown["Entropy"] = "2 (High entropy)"
    elif entropy > 1.5:
        score += 1
        breakdown["Entropy"] = "1 (Moderate entropy)"
    else:
        breakdown["Entropy"] = "0 (Low entropy)"

    # Final Strength Rating
    if score >= 7:
        strength = "Strong"
    elif score >= 4:
        strength = "Moderate"
    else:
        strength = "Weak"

    return strength, score, breakdown

# Function to check if the CAPTCHA text is a palindrome
def is_palindrome(captcha_text):
    return captcha_text.lower() == captcha_text.lower()[::-1]

# Function to preprocess the image for parity prediction
def preprocess_image(image):
    image_size = (100, 500)
    image = cv2.resize(image, image_size)
    image = np.transpose(image, (1, 0, 2))  # Transpose to match training format
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Cache model loading for performance
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('captcha_model.keras')
        return model
    except Exception as e:
        st.markdown(f'<div class="error-box">Error loading model: {str(e)}. Please ensure captcha_model.keras exists in the directory.</div>', unsafe_allow_html=True)
        st.stop()

model = load_model()

# Streamlit app
st.markdown('<div class="title">Deep Learning-based System for Hexadecimal CAPTCHA Recognition and Parity Classification</div>', unsafe_allow_html=True)
st.markdown("Upload a CAPTCHA image, manually enter the 4-digit hexadecimal text, and analyze its strength, parity, and bidirectional property. Zoom into the image for clarity and review past analyses in the sidebar.", unsafe_allow_html=True)

# Initialize session state
if 'form_reset' not in st.session_state:
    st.session_state.form_reset = False
if 'captcha_text' not in st.session_state:
    st.session_state.captcha_text = ""
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'input_key' not in st.session_state:
    st.session_state.input_key = "captcha_input_0"
if 'history' not in st.session_state:
    st.session_state.history = []

# Handle reset
if st.session_state.form_reset:
    st.session_state.captcha_text = ""
    st.session_state.uploaded_file = None
    st.session_state.input_key = f"captcha_input_{np.random.randint(1000)}"
    st.session_state.form_reset = False
    st.rerun()

# Sidebar for history
with st.sidebar:
    st.header("Analysis History")
    if st.session_state.history:
        for idx, entry in enumerate(st.session_state.history):
            with st.expander(f"Analysis {idx+1}: {entry['captcha_text']}"):
                st.image(entry['thumbnail'], caption="Thumbnail", width=100)
                st.write(f"**Text**: {entry['captcha_text']}")
                st.write(f"**Strength**: {entry['strength']} (Score: {entry['score']}/10)")
                st.write(f"**Parity**: {entry['parity']} ({entry['confidence']:.2f}%)")
                st.write(f"**Bidirectional**: {entry['bidirectional']}")
    else:
        st.write("No analyses yet. Complete an analysis to see results here.")

# File uploader for CAPTCHA image
with st.container():
    st.markdown('<div class="subtitle">Step 1: Upload CAPTCHA Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CAPTCHA image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], help="Upload a clear image containing a 4-digit hexadecimal CAPTCHA.", key="file_uploader")
    
    # Update session state with uploaded file
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

if st.session_state.uploaded_file is not None:
    # Read and display the image with zoom
    image = Image.open(st.session_state.uploaded_file)
    st.markdown('<div class="zoom-container">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded CAPTCHA Image (Hover to zoom)", use_column_width=True, output_format="PNG", clamp=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Create thumbnail for history
    thumbnail = Image.fromarray(cv2.resize(image_cv, (100, 50)))
    
    # Manual CAPTCHA text input
    st.markdown('<div class="subtitle">Step 2: Enter CAPTCHA Text</div>', unsafe_allow_html=True)
    st.markdown("Type the 4-digit hexadecimal text you see in the image (e.g., A1B2). Press Enter to analyze, Ctrl+R to reset.", unsafe_allow_html=True)
    captcha_text = st.text_input("CAPTCHA Text:", value=st.session_state.captcha_text, help="Enter exactly 4 hexadecimal characters (0-9, A-F).", key=st.session_state.input_key, on_change=None)
    
    # Update session state with CAPTCHA text
    st.session_state.captcha_text = captcha_text
    
    # Real-time input validation
    if captcha_text:
        if len(captcha_text) == 4 and all(c in string.hexdigits for c in captcha_text.upper()):
            st.markdown('<div class="success-box">Valid CAPTCHA text entered!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">Invalid input: Please enter a 4-digit hexadecimal text (e.g., A1B2).</div>', unsafe_allow_html=True)
    
    # Analyze button
    if st.button("Analyze CAPTCHA", key="analyze_button", type="primary"):
        if captcha_text and len(captcha_text) == 4 and all(c in string.hexdigits for c in captcha_text.upper()):
            with st.spinner("Analyzing CAPTCHA..."):
                # Preprocess the image for parity prediction
                processed_image = preprocess_image(image_cv)
                
                # Predict parity with confidence
                prediction = model.predict(processed_image)
                predicted_class = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction) * 100
                label_dict = {0: 'EVEN', 1: 'ODD'}
                parity_result = label_dict[predicted_class]
                
                # Confidence visualization
                confidence_df = pd.DataFrame({
                    "Class": ["EVEN", "ODD"],
                    "Confidence": [prediction[0][0] * 100, prediction[0][1] * 100]
                })
                fig = px.bar(confidence_df, x="Class", y="Confidence", title="Parity Prediction Confidence", color="Class", color_discrete_map={"EVEN": "#10b981", "ODD": "#ef4444"})
                st.plotly_chart(fig, use_container_width=True)
                
                # Check CAPTCHA strength
                strength, score, breakdown = check_captcha_strength(captcha_text, image_cv)
                
                # Check bidirectional property
                bidirectional_result = "Yes ✅" if is_palindrome(captcha_text) else "No ❌"
                
                # Save to history
                st.session_state.history.append({
                    "captcha_text": captcha_text,
                    "strength": strength,
                    "score": score,
                    "parity": parity_result,
                    "confidence": confidence,
                    "bidirectional": bidirectional_result,
                    "thumbnail": thumbnail
                })
                
                # Display results
                st.markdown('<div class="subtitle">Analysis Results</div>', unsafe_allow_html=True)
                with st.container():
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown(f"**CAPTCHA Text**: {captcha_text}", unsafe_allow_html=True)
                    
                    # Color-coded strength display
                    strength_color = {"Strong": "#065f46", "Moderate": "#d97706", "Weak": "#991b1b"}
                    st.markdown(f'<p style="color:{strength_color[strength]}">**Strength**: {strength} (Score: {score}/10)</p>', unsafe_allow_html=True)
                    
                    # Progress bar for strength score
                    st.progress(score / 10.0)
                    
                    # Strength breakdown
                    with st.expander("View Strength Breakdown"):
                        for key, value in breakdown.items():
                            st.markdown(f"- **{key}**: {value}")
                    
                    # Parity with confidence
                    st.markdown(f"**Parity**: The hexadecimal number is **{parity_result}** (Confidence: {confidence:.2f}%)", unsafe_allow_html=True)
                    
                    # Bidirectional result
                    st.markdown(f"**Bidirectional (Palindrome)**: {bidirectional_result}", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Export results
                if st.button("Export Results as CSV", key=f"export_{time.time()}"):
                    results_df = pd.DataFrame([{
                        "CAPTCHA Text": captcha_text,
                        "Strength": strength,
                        "Score": score,
                        "Parity": parity_result,
                        "Confidence (%)": confidence,
                        "Bidirectional": bidirectional_result.replace(" ✅", "").replace(" ❌", "")
                    }])
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"captcha_analysis_{time.time()}.csv",
                        mime="text/csv",
                        key=f"download_{time.time()}"
                    )
        else:
            st.markdown('<div class="error-box">Please enter a valid 4-digit hexadecimal CAPTCHA text (e.g., A1B2).</div>', unsafe_allow_html=True)
    
    # Reset button
    if st.button("Reset and Upload New Image", key="reset_button", type="secondary"):
        st.session_state.form_reset = True
        st.rerun()

else:
    st.markdown('<div class="error-box">Please upload a CAPTCHA image to proceed.</div>', unsafe_allow_html=True)
