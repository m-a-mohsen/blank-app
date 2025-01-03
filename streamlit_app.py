import streamlit as st
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import time
import random

def load_dicom_image(dicom_file):
    """
    Load and convert DICOM file to displayable image
    """
    try:
        # Read DICOM file
        dicom_data = pydicom.dcmread(dicom_file)
        
        # Convert to numpy array and normalize
        image_array = dicom_data.pixel_array
        image_normalized = ((image_array - image_array.min()) / 
                            (image_array.max() - image_array.min()) * 255).astype(np.uint8)
        
        return image_normalized
    except Exception as e:
        st.error(f"Error processing DICOM file: {e}")
        return None

def display_dicom_image(image_array):
    """
    Create visualization of DICOM image
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display image
    ax.imshow(image_array, cmap='gray')
    ax.set_title('DICOM Brain CT Scan')
    ax.axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

def simulate_prediction():
    """
    Simulate a prediction process with a spinner and random result
    """
    hemorrhage_types = [
        'Epidural Hemorrhage', 
        'Subdural Hemorrhage', 
        'Subarachnoid Hemorrhage', 
        'Intraventricular Hemorrhage', 
        'Intracerebral Hemorrhage'
    ]
    
    with st.spinner('AI is analyzing the brain scan...'):
        # Simulate processing time
        time.sleep(3)
        
        # Randomly select a hemorrhage type
        predicted_type = random.choice(hemorrhage_types)
        confidence = round(random.uniform(0.6, 0.95), 2)
    
    # Display simulated results
    st.success(f"Predicted Hemorrhage Type: {predicted_type}")
    st.info(f"Confidence: {confidence * 100:.2f}%")

def main():
    st.set_page_config(
        page_title="Brain CT Scan Analyzer", 
        page_icon=":brain:", 
        layout="wide"
    )
    
    st.title("🧠 Brain CT Scan Analyzer")
    
    # Sidebar instructions
    st.sidebar.header("📋 Instructions")
    st.sidebar.markdown("""
    1. Upload a Brain CT Scan (.dcm file)
    2. View the medical image
    3. Get AI-powered analysis
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload DICOM Brain CT Scan", 
        type=['dcm'], 
        help="Upload a DICOM format brain CT scan"
    )
    
    # Image display and prediction section
    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Load and display DICOM image
            dicom_image = load_dicom_image(uploaded_file)
            
            if dicom_image is not None:
                st.subheader("📸 Brain CT Scan")
                display_dicom_image(dicom_image)
        
        with col2:
            # Prediction simulation
            st.subheader("🤖 AI Analysis")
            if st.button("Analyze Scan"):
                simulate_prediction()

    # Additional information
    st.markdown("---")
    st.markdown("""
    ### ℹ️ About This Tool
    - This is a demonstration of medical image analysis
    - Actual predictions require a trained AI model
    - Always consult medical professionals for accurate diagnoses
    """)

if __name__ == "__main__":
    main()

# import streamlit as st
# import requests
# import io
# import base64
# import pydicom
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# # Configuration
# API_ENDPOINT = st.secrets.get("API_ENDPOINT", "http://localhost:8000/predict_base64")

# def load_dicom_image(dicom_file):
#     """
#     Load and convert DICOM file to displayable image
#     """
#     try:
#         dicom_data = pydicom.dcmread(dicom_file)
#         image_array = dicom_data.pixel_array
        
#         # Normalize image
#         image_normalized = ((image_array - image_array.min()) / 
#                             (image_array.max() - image_array.min()) * 255).astype(np.uint8)
        
#         return image_normalized
#     except Exception as e:
#         st.error(f"Error processing DICOM file: {e}")
#         return None

# def convert_image_to_base64(image_array):
#     """
#     Convert numpy image array to base64 for API transmission
#     """
#     pil_image = Image.fromarray(image_array)
#     pil_image_resized = pil_image.resize((224, 224))
    
#     buffered = io.BytesIO()
#     pil_image_resized.save(buffered, format="PNG")
    
#     return base64.b64encode(buffered.getvalue()).decode('utf-8')

# def send_to_ai_model(base64_image):
#     """
#     Send image to FastAPI model
#     """
#     try:
#         response = requests.post(
#             API_ENDPOINT, 
#             json={"image": base64_image},
#             headers={"Content-Type": "application/json"}
#         )
        
#         if response.status_code == 200:
#             return response.json()
#         else:
#             st.error(f"Model API error: {response.status_code}")
#             return None
    
#     except requests.exceptions.RequestException as e:
#         st.error(f"Network error: {e}")
#         return None

# def display_results(original_image, ai_results):
#     """
#     Create visualization of results
#     """
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
#     # Original image
#     ax1.imshow(original_image, cmap='gray')
#     ax1.set_title('Original DICOM Image')
#     ax1.axis('off')
    
#     # Results bar chart
#     if ai_results:
#         hemorrhage_types = [
#             'Epidural', 
#             'Subdural', 
#             'Subarachnoid', 
#             'Intraventricular', 
#             'Intracerebral'
#         ]
#         probabilities = ai_results.get('probabilities', [0]*len(hemorrhage_types))
        
#         ax2.bar(hemorrhage_types, probabilities)
#         ax2.set_title('Hemorrhage Type Probabilities')
#         ax2.set_ylabel('Probability')
#         ax2.set_xticklabels(hemorrhage_types, rotation=45)
    
#     plt.tight_layout()
#     st.pyplot(fig)

# def main():
#     st.title("🧠 Brain Hemorrhage Classifier")
    
#     # Sidebar 
#     st.sidebar.header("Instructions")
#     st.sidebar.info(
#         "1. Upload a Brain CT Scan (.dcm file)\n"
#         "2. Wait for AI analysis\n"
#         "3. View hemorrhage type probabilities"
#     )
    
#     # File uploader
#     uploaded_file = st.file_uploader(
#         "Upload DICOM Brain CT Scan", 
#         type=['dcm'], 
#         help="Upload a DICOM format brain CT scan"
#     )
    
#     if uploaded_file is not None:
#         with st.spinner('Processing image...'):
#             # Load DICOM image
#             dicom_image = load_dicom_image(uploaded_file)
            
#             if dicom_image is not None:
#                 # Convert to base64
#                 base64_image = convert_image_to_base64(dicom_image)
                
#                 # Send to AI model
#                 ai_results = send_to_ai_model(base64_image)
                
#                 if ai_results:
#                     # Display results
#                     display_results(dicom_image, ai_results)
                    
#                     # Detailed probabilities
#                     st.subheader("Detailed Analysis")
#                     hemorrhage_types = [
#                         'Epidural', 
#                         'Subdural', 
#                         'Subarachnoid', 
#                         'Intraventricular', 
#                         'Intracerebral'
#                     ]
                    
#                     for type, prob in zip(hemorrhage_types, ai_results.get('probabilities', [0]*5)):
#                         st.metric(label=type, value=f"{prob*100:.2f}%")
                    
#                     # Prediction summary
#                     st.success(f"Predicted Hemorrhage Type: {ai_results.get('predicted_type')}")
#                     st.info(f"Confidence: {ai_results.get('confidence', 0)*100:.2f}%")

# if __name__ == "__main__":
#     main()