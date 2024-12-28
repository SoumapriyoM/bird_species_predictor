#1.
# import streamlit as st
# import numpy as np
# import librosa
# import matplotlib.cm as cm
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
# from PIL import Image
# import io

# # Array of bird species labels
# class_labels = np.array(['aldfly', 'amegfi', 'astfly', 'balori', 'bewwre', 'bkhgro',
#                          'bkpwar', 'blugrb1', 'brdowl', 'brespa', 'brnthr', 'buhvir',
#                          'bulori', 'cangoo', 'canwar', 'canwre', 'carwre', 'comrav',
#                          'daejun', 'eastow', 'eawpew', 'foxspa', 'gnttow', 'hamfly',
#                          'herthr', 'hoowar', 'houfin', 'houspa', 'indbun', 'lesgol',
#                          'louwat', 'magwar', 'marwre', 'norcar', 'normoc', 'olsfly',
#                          'pasfly', 'reevir1', 'rewbla', 'scoori', 'spotow', 'swathr',
#                          'vesspa', 'warvir', 'wesmea', 'westan', 'wewpew', 'whbnut',
#                          'woothr', 'yebfly'])

# # Function to process audio file as RGB mel spectrogram
# def process_audio_as_rgb(audio_file):
#     # Convert uploaded file to a numpy array
#     audio_data, sample_rate = librosa.load(audio_file, duration=10)  # Load 10 seconds of audio
#     mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    
#     # Convert to dB scale (log scale)
#     mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
#     # Normalize mel spectrogram to range [0, 1]
#     mel_spec -= mel_spec.min()
#     mel_spec /= mel_spec.max()

#     # Apply a colormap to convert the mel spectrogram to an RGB image
#     colormap = cm.get_cmap('viridis')  # You can choose any colormap you prefer
#     mel_spec_rgb = colormap(mel_spec)  # Apply colormap
    
#     # Convert to 8-bit RGB and remove the alpha channel
#     mel_spec_rgb = (mel_spec_rgb[..., :3] * 255).astype(np.uint8)  # Keep only the RGB channels
    
#     return mel_spec_rgb

# # Load the pre-trained model (ensure you have your model saved in the proper directory)
# model = load_model('my_model.h5')  # Replace with the actual path to your model

# # Streamlit UI components
# st.title('Bird Species Prediction from Audio')

# # Use a markdown description with instructions
# st.markdown("""
#     ## How to Use:
#     1. Upload an audio file (max 10 seconds).
#     2. The system will process the audio and predict the bird species.
#     3. The result will be displayed below.
# """)

# # File uploader widget with a label
# audio_file = st.file_uploader("Choose an audio file (ogg, mp3, wav)", type=["ogg", "mp3", "wav"])

# # Display the file name when uploaded
# if audio_file is not None:
#     st.write(f"File uploaded: {audio_file.name}")
    
#     # Display the audio file preview (First 10 seconds of audio)
#     audio_bytes = audio_file.getvalue()
#     st.audio(audio_bytes, format="audio/wav")

#     # Process the audio file
#     processed_audio = process_audio_as_rgb(audio_file)
    
#     # Show mel spectrogram as image
#     st.image(processed_audio, caption="Mel Spectrogram", use_column_width=True)

#     # Ensure the shape matches the expected input shape (128, 431, 3)
#     expected_shape = (128, 431, 3)
#     if processed_audio.shape != expected_shape:
#         # Resize or pad if necessary to match the expected shape
#         audio_data_img = array_to_img(processed_audio)
#         audio_data_img = audio_data_img.resize((431, 128))  # Resize to (431, 128)
#         processed_audio = img_to_array(audio_data_img)
    
#     # Add batch dimension and ensure data type
#     processed_audio = np.expand_dims(processed_audio, axis=0)  # Add batch dimension
#     processed_audio = processed_audio.astype(np.float32)

#     # Model prediction with a loading spinner
#     with st.spinner('Processing the audio...'):
#         try:
#             prediction = model.predict(processed_audio)
#             pred_label = np.argmax(prediction, axis=1)
#             st.success(f"Predicted bird species: {class_labels[pred_label[0]]}")
#         except Exception as e:
#             st.error(f"Error during prediction: {e}")
# else:
#     st.warning("Please upload an audio file to proceed.")
#............................................................................................
#2
# import streamlit as st
# import numpy as np
# import librosa
# import matplotlib.cm as cm
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
# from PIL import Image
# import io

# # Array of bird species labels
# class_labels = np.array(['aldfly', 'amegfi', 'astfly', 'balori', 'bewwre', 'bkhgro',
#                          'bkpwar', 'blugrb1', 'brdowl', 'brespa', 'brnthr', 'buhvir',
#                          'bulori', 'cangoo', 'canwar', 'canwre', 'carwre', 'comrav',
#                          'daejun', 'eastow', 'eawpew', 'foxspa', 'gnttow', 'hamfly',
#                          'herthr', 'hoowar', 'houfin', 'houspa', 'indbun', 'lesgol',
#                          'louwat', 'magwar', 'marwre', 'norcar', 'normoc', 'olsfly',
#                          'pasfly', 'reevir1', 'rewbla', 'scoori', 'spotow', 'swathr',
#                          'vesspa', 'warvir', 'wesmea', 'westan', 'wewpew', 'whbnut',
#                          'woothr', 'yebfly'])

# # Function to process audio file as RGB mel spectrogram
# def process_audio_as_rgb(audio_file):
#     audio_data, sample_rate = librosa.load(audio_file, duration=10)  # Load 10 seconds of audio
#     mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
#     mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB scale
#     mel_spec -= mel_spec.min()
#     mel_spec /= mel_spec.max()
#     colormap = cm.get_cmap('viridis')  # Apply colormap
#     mel_spec_rgb = colormap(mel_spec)[..., :3] * 255  # Convert to RGB
#     return mel_spec_rgb.astype(np.uint8)

# # Load the pre-trained model
# model = load_model('my_model.h5')  # Replace with the actual path to your model

# # Apply custom CSS for a better design
# st.markdown("""
#     <style>
#         .main {
#             background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
#             color: #333333;
#         }
#         .title {
#             font-family: 'Trebuchet MS', sans-serif;
#             font-size: 2.5rem;
#             color: #1a73e8;
#             text-align: center;
#             margin-bottom: 1rem;
#         }
#         .instructions {
#             font-family: 'Arial', sans-serif;
#             color: #444444;
#             margin: 1rem auto;
#             padding: 10px;
#             border-left: 5px solid #1a73e8;
#             background-color: #f9f9f9;
#             box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
#         }
#         .prediction {
#             font-size: 1.5rem;
#             color: #0a944f;
#             font-weight: bold;
#             text-align: center;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Title Section
# st.markdown('<h1 class="title">🐦 Bird Species Prediction from Audio</h1>', unsafe_allow_html=True)

# # Instructions Section
# st.markdown("""
# <div class="instructions">
#     <p>Welcome to the Bird Species Prediction App! Follow these steps:</p>
#     <ol>
#         <li>Upload an audio file (max 10 seconds).</li>
#         <li>View the generated Mel spectrogram.</li>
#         <li>Get the predicted bird species!</li>
#     </ol>
# </div>
# """, unsafe_allow_html=True)

# # File uploader widget
# audio_file = st.file_uploader("🎵 Upload an audio file (ogg, mp3, wav):", type=["ogg", "mp3", "wav"])

# # File Processing Section
# if audio_file:
#     st.write(f"**File uploaded:** `{audio_file.name}`")
#     st.audio(audio_file, format="audio/wav")
    
#     # Process the audio file
#     processed_audio = process_audio_as_rgb(audio_file)
#     st.image(processed_audio, caption="🎶 Mel Spectrogram", use_column_width=True)
    
#     # Resize to match the model's expected input shape
#     expected_shape = (128, 431, 3)
#     if processed_audio.shape != expected_shape:
#         processed_audio = img_to_array(array_to_img(processed_audio).resize((431, 128)))
#     processed_audio = np.expand_dims(processed_audio, axis=0).astype(np.float32)
    
#     # Predict with the model
#     with st.spinner('🔄 Analyzing audio...'):
#         try:
#             prediction = model.predict(processed_audio)
#             pred_label = np.argmax(prediction, axis=1)
#             st.markdown(f'<p class="prediction">Predicted Bird Species: {class_labels[pred_label[0]]}</p>', unsafe_allow_html=True)
#         except Exception as e:
#             st.error(f"An error occurred during prediction: {e}")
# else:
#     st.warning("⚠️ Please upload an audio file to proceed.")
#3
import streamlit as st
import numpy as np
import librosa
import matplotlib.cm as cm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image
import time

# Array of bird species labels
class_labels = np.array(['aldfly', 'amegfi', 'astfly', 'balori', 'bewwre', 'bkhgro',
                         'bkpwar', 'blugrb1', 'brdowl', 'brespa', 'brnthr', 'buhvir',
                         'bulori', 'cangoo', 'canwar', 'canwre', 'carwre', 'comrav',
                         'daejun', 'eastow', 'eawpew', 'foxspa', 'gnttow', 'hamfly',
                         'herthr', 'hoowar', 'houfin', 'houspa', 'indbun', 'lesgol',
                         'louwat', 'magwar', 'marwre', 'norcar', 'normoc', 'olsfly',
                         'pasfly', 'reevir1', 'rewbla', 'scoori', 'spotow', 'swathr',
                         'vesspa', 'warvir', 'wesmea', 'westan', 'wewpew', 'whbnut',
                         'woothr', 'yebfly'])

# Function to process audio file as RGB mel spectrogram
def process_audio_as_rgb(audio_file):
    audio_data, sample_rate = librosa.load(audio_file, duration=10)  # Load 10 seconds of audio
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB scale
    mel_spec -= mel_spec.min()
    mel_spec /= mel_spec.max()
    colormap = cm.get_cmap('viridis')  # Apply colormap
    mel_spec_rgb = colormap(mel_spec)[..., :3] * 255  # Convert to RGB
    return mel_spec_rgb.astype(np.uint8)

# Load the pre-trained model
model = load_model('my_model.h5')  # Replace with the actual path to your model

# Apply custom CSS for a more beautiful interface
st.markdown("""
    <style>
        body {
            background: linear-gradient(120deg, #89f7fe, #66a6ff);
            color: white;
        }
        .main {
            background: rgba(255, 255, 255, 0.8);
            padding: 1rem;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
        }
        .title {
            font-family: 'Arial Black', sans-serif;
            font-size: 3rem;
            color: #1c1c1c;
            text-shadow: 2px 2px #66a6ff;
            text-align: center;
        }
        .instructions {
            font-family: 'Verdana', sans-serif;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .btn-upload {
            font-family: 'Verdana', sans-serif;
            font-size: 1rem;
            padding: 10px 20px;
            background: #66a6ff;
            color: white;
            border-radius: 20px;
            text-transform: uppercase;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
        }
        .btn-upload:hover {
            background: #1c1c1c;
            color: #66a6ff;
            transition: 0.3s;
        }
        .prediction {
            font-size: 2rem;
            color: #1c1c1c;
            font-weight: bold;
            text-align: center;
        }
        .spectrogram {
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown('<h1 class="title">🐦 Bird Species Audio Classifier</h1>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("Use this sidebar to explore the app:")
st.sidebar.markdown("- **Upload Audio**: Upload your audio file.")
st.sidebar.markdown("- **View Spectrogram**: See the processed spectrogram.")
st.sidebar.markdown("- **Prediction**: View the bird species prediction.")

# File Uploader Section
st.markdown('<p class="instructions">🎵 Upload an audio file (ogg, mp3, wav) below to get started.</p>', unsafe_allow_html=True)
audio_file = st.file_uploader("", type=["ogg", "mp3", "wav"])

if audio_file:
    st.write(f"**Uploaded File:** `{audio_file.name}`")
    st.audio(audio_file, format="audio/wav")
    
    # Process the audio
    with st.spinner('🔄 Generating Mel Spectrogram...'):
        processed_audio = process_audio_as_rgb(audio_file)
        time.sleep(2)  # Simulate processing time

    # Display the spectrogram
    st.image(processed_audio, caption="🎶 Mel Spectrogram", use_column_width=True, output_format="PNG")

    # Ensure correct shape for model input
    expected_shape = (128, 431, 3)
    if processed_audio.shape != expected_shape:
        processed_audio = img_to_array(array_to_img(processed_audio).resize((431, 128)))
    processed_audio = np.expand_dims(processed_audio, axis=0).astype(np.float32)

    # Predict Bird Species
    with st.spinner('🔍 Analyzing audio for bird species...'):
        try:
            prediction = model.predict(processed_audio)
            pred_label = np.argmax(prediction, axis=1)
            time.sleep(2)  # Simulate prediction time
            st.markdown(f'<p class="prediction">Predicted Bird Species: 🐦 <strong>{class_labels[pred_label[0]]}</strong></p>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
else:
    st.markdown('<p class="instructions" style="text-align: center;">Please upload an audio file to proceed.</p>', unsafe_allow_html=True)



