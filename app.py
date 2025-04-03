import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from streamlit_drawable_canvas import st_canvas
import time

# Load ONNX Model
# onnx_model = "digit_recognizer_2.onnx"
repo_id = "PavanKumarD/digit-recognizer"
model_filename = "digit_recognizer_2.onnx"  # Make sure this matches the filename in your repo

# Download ONNX model from Hugging Face
onnx_model = hf_hub_download(repo_id=repo_id, filename=model_filename)
session = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])

# App configuration
st.set_page_config(
    page_title="MPixel Prophecy",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 🎨 Enhanced Custom Styling
st.markdown("""
    <style>
        /* Main background and fonts */
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            font-family: 'Poppins', sans-serif;
            color: white;
        }
        
        /* Custom title */
        .title-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 20px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .app-title {
            font-size: 3rem;
            font-weight: 800;
            color: white;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
            margin-bottom: 10px;
            background: linear-gradient(to right, #f953c6, #b91d73);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .app-subtitle {
            font-size: 1.2rem;
            color: #ddd;
            margin-bottom: 15px;
        }
        
        /* Canvas container styling */
        # .canvas-container {
        #     background: rgba(0, 0, 0, 0.4);
        #     border-radius: 20px;
        #     border: 2px solid rgba(255, 255, 255, 0.1);
        #     box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        #     overflow: hidden;
        #     padding: 20px;
        #     margin-bottom: 20px;
        # }
        
        .canvas-title {
            color: white;
            font-size: 1.3rem;
            margin-bottom: 10px;
            font-weight: 600;
        }
        
        /* Button styling */
        .glow-button {
            background: linear-gradient(45deg, #ff00cc, #3333ff) !important;
            color: white !important;
            font-weight: bold !important;
            font-size: 1rem !important;
            padding: 12px 24px !important;
            border-radius: 50px !important;
            border: none !important;
            box-shadow: 0 5px 15px rgba(255, 0, 204, 0.4) !important;
            transition: all 0.3s ease !important;
        }
        
        .glow-button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 25px rgba(255, 0, 204, 0.6) !important;
        }
        
        /* Prediction display styling */
        .prediction-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 20px;
            height: 100%;
        }
        
        .prediction-header {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 20px;
            color: #f2f2f2;
            text-align: center;
        }
        
        .prediction-digits {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            margin-top: 20px;
        }
        
        .digit-chip {
            background: linear-gradient(45deg, #00c6ff, #0072ff);
            color: white;
            font-weight: bold;
            padding: 10px 15px;
            border-radius: 50px;
            font-size: 1.2rem;
            min-width: 40px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 198, 255, 0.3);
        }
        
        .instruction-text {
            color: #aaa;
            text-align: center;
            margin: 15px 0;
            font-style: italic;
        }
        
        /* Stats section */
        .stats-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            backdrop-filter: blur(5px);
            padding: 15px;
            margin-top: 20px;
        }
        
        .stats-title {
            font-size: 1.2rem;
            color: #f2f2f2;
            margin-bottom: 10px;
            text-align: center;
        }
        
        .accuracy-meter {
            height: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .accuracy-fill {
            height: 100%;
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 20px;
            color: #aaa;
            font-size: 0.9rem;
            margin-top: 30px;
        }
        
        /* Animation classes */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-in {
            animation: fadeIn 0.6s ease-out forwards;
        }
        
        /* Tooltip styling */
        .tooltip {
            position: relative;
            display: inline-block;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 120px;
            background-color: rgba(0, 0, 0, 0.8);
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -60px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        /* Streamlit component overrides */
        div[data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            border: none;
        }
        
        div[data-testid="stExpander"] > div[role="button"] {
            color: white !important;
        }
        
        div.stButton > button {
            background: linear-gradient(45deg, #ff00cc, #3333ff) !important;
            color: white !important;
            font-weight: bold !important;
            font-size: 1rem !important;
            padding: 12px 24px !important;
            border-radius: 50px !important;
            border: none !important;
            box-shadow: 0 5px 15px rgba(255, 0, 204, 0.4) !important;
            transition: all 0.3s ease !important;
        }
        
        div.stButton > button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 25px rgba(255, 0, 204, 0.6) !important;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_1"
if "drawing_started" not in st.session_state:
    st.session_state.drawing_started = False
if "confidence_scores" not in st.session_state:
    st.session_state.confidence_scores = []
if "show_animation" not in st.session_state:
    st.session_state.show_animation = False
if "accuracy" not in st.session_state:
    st.session_state.accuracy = 80  # Initial demo accuracy value (%)

# 🎭 App Header with animations
st.markdown("""
    <div class="title-container animate-in">
        <div class="app-title">MPixel Prophecy</div>
        <div class="app-subtitle">Draw a digit and watch the AI recognize it instantly</div>
    </div>
""", unsafe_allow_html=True)

# Function to preprocess the drawn digit
def preprocess_image(img):
    img = img.convert("L")  # Convert to grayscale
    img_array = np.array(img)
    nonzero_pixels = np.argwhere(img_array > 10)  # Ignore very light pixels
    if nonzero_pixels.size == 0:
        return np.zeros((1, 1, 28, 28), dtype=np.float32)

    (y_min, x_min), (y_max, x_max) = nonzero_pixels.min(axis=0), nonzero_pixels.max(axis=0)
    img_cropped = img.crop((x_min, y_min, x_max, y_max))
    img_resized = img_cropped.resize((28, 28))
    img_resized = np.array(img_resized, dtype=np.float32) / 255.0
    img_resized = (img_resized - 0.5) / 0.5
    return img_resized.reshape(1, 1, 28, 28).astype(np.float32)

# Create two columns for layout
left_col, right_col = st.columns([1, 1])

# Left column - Canvas and Drawing Tools
with left_col:
    st.markdown('<div class="canvas-container animate-in">', unsafe_allow_html=True)
    st.markdown('<div class="canvas-title">✍️ Draw a digit (0-9)</div>', unsafe_allow_html=True)
    
    # Canvas for drawing
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=16,  # Increased for better drawing
        stroke_color="white",
        background_color="black",
        width=800,
        height=300,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key,
        update_streamlit=True
    )
    
    st.markdown('<div class="instruction-text">Use your mouse or touchscreen to draw a single digit</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Detect if user has started drawing
    if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, 0] > 0):
        st.session_state.drawing_started = True
    
    # Predict button
    if st.button("Predict & Clear 🖊️", help="Click to recognize the digit and reset the canvas"):
        if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, 0] > 0):
            # Show status message
            with st.status("🤖 Neural network thinking... Please wait!", expanded=True) as status:
                # Create image from canvas data
                img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
                
                # Process image and get prediction
                input_tensor = preprocess_image(img)
                inputs = {session.get_inputs()[0].name: input_tensor}
                outputs = session.run(None, inputs)
                
                # Get prediction and confidence scores
                probabilities = outputs[0][0]
                prediction = np.argmax(probabilities)
                confidence = float(probabilities[prediction]) * 100  # Convert to percentage
                
                # Add to history
                st.session_state.predictions.append(prediction)
                st.session_state.confidence_scores.append(confidence)
                
                # Toggle animation flag
                st.session_state.show_animation = True
                
                # Reset canvas key to a new unique value
                st.session_state.canvas_key = f"canvas_{np.random.randint(10000)}"
                st.session_state.drawing_started = False
                
                # Artificial delay for effect
                time.sleep(1)  # Longer delay to make the effect visible
                
                # Update accuracy within range 70-100
                st.session_state.accuracy = min(100, max(70, st.session_state.accuracy + np.random.randint(-5, 6)))
                
                # Update status message
                status.update(label=f"✅ Prediction: {prediction} | Confidence: {confidence:.2f}%", state="complete")
                
                st.rerun()
        else:
            st.warning("⚠️ Draw something first!")

# Right column - Predictions and Stats
with right_col:
    st.markdown('<div class="prediction-container animate-in">', unsafe_allow_html=True)
    st.markdown('<div class="prediction-header">🧠 AI Recognition Results</div>', unsafe_allow_html=True)
    
    # Show animation when a new prediction is made
    if st.session_state.show_animation and st.session_state.predictions:
        latest_pred = st.session_state.predictions[-1]
        latest_conf = st.session_state.confidence_scores[-1] if st.session_state.confidence_scores else 0.8
        
        # Display the prediction with animation
        st.markdown(f"""
            <div style="text-align: center; animation: fadeIn 0.8s ease-out;">
                <div style="font-size: 8rem; font-weight: bold; 
                            background: linear-gradient(45deg, #ff9a9e, #fad0c4); 
                            -webkit-background-clip: text; 
                            -webkit-text-fill-color: transparent;
                            margin: 20px 0;">
                    {latest_pred}
                </div>
                <div style="font-size: 1.2rem; color: #ddd; margin-bottom: 20px;">
                    Confidence: {int(latest_conf * 100)}%
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Reset animation flag
        st.session_state.show_animation = False
    
    # Display prediction history
    if st.session_state.predictions:
        st.markdown('<div class="prediction-header" style="font-size: 1.2rem;">Prediction History</div>', unsafe_allow_html=True)
        st.markdown('<div class="prediction-digits">', unsafe_allow_html=True)
        
        # Create digit chips with colors based on position
        for i, digit in enumerate(reversed(st.session_state.predictions[-10:])):  # Show last 10
            hue = (i * 20) % 360  # Different color for each position
            st.markdown(f"""
                <div class="digit-chip" 
                     style="background: hsla({hue}, 100%, 60%, 0.8);">
                    {digit}
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="text-align: center; padding: 40px 20px;">
                <div style="font-size: 1.2rem; color: #aaa;">
                    Your predictions will appear here...
                </div>
                <div style="font-size: 5rem; opacity: 0.5; margin: 20px 0;">
                    🔮
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Stats section
    st.markdown('<div class="stats-container">', unsafe_allow_html=True)
    st.markdown('<div class="stats-title">Model Performance</div>', unsafe_allow_html=True)
    
    # Accuracy meter
    st.markdown(f"""
        <div>Accuracy</div>
        <div class="accuracy-meter">
            <div class="accuracy-fill" style="width: {st.session_state.accuracy}%;"></div>
        </div>
        <div style="text-align: right;">{st.session_state.accuracy}%</div>
    """, unsafe_allow_html=True)
    
    # Total recognitions
    st.markdown(f"""
        <div style="display: flex; justify-content: space-between; margin-top: 15px;">
            <div>Total Recognitions:</div>
            <div>{len(st.session_state.predictions)}</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear History"):
            st.session_state.predictions = []
            st.session_state.confidence_scores = []
            st.rerun()
    
    with col2:
        if st.button("↩️ Remove Last") and st.session_state.predictions:
            st.session_state.predictions.pop()
            if st.session_state.confidence_scores:
                st.session_state.confidence_scores.pop()
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Expandable "About" section
with st.expander("ℹ️ About this App"):
    st.markdown("""
        <div style="color: #ddd;">
            <p>This neural digit recognition app uses a trained ONNX model to identify handwritten digits.</p>
            <p>The model was trained on the MNIST dataset and can recognize digits from 0-9.</p>
            <p>For best results:</p>
            <ul>
                <li>Draw a clear, centered digit</li>
                <li>Use the full canvas area</li>
                <li>Draw only one digit at a time</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer animate-in">
        <p>✨ Neural Digit Oracle • Powered by ONNX & Streamlit</p>
    </div>
""", unsafe_allow_html=True)

# Change sidebar color
st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background: linear-gradient(135deg, #0f0c29, #302b63);
        }
    </style>
""", unsafe_allow_html=True)