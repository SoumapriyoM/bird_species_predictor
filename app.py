import streamlit as st
import numpy as np
import librosa
import matplotlib.cm as cm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image
import io

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
    # Convert uploaded file to a numpy array
    audio_data, sample_rate = librosa.load(audio_file, duration=10)  # Load 10 seconds of audio
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    
    # Convert to dB scale (log scale)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize mel spectrogram to range [0, 1]
    mel_spec -= mel_spec.min()
    mel_spec /= mel_spec.max()

    # Apply a colormap to convert the mel spectrogram to an RGB image
    colormap = cm.get_cmap('viridis')  # You can choose any colormap you prefer
    mel_spec_rgb = colormap(mel_spec)  # Apply colormap
    
    # Convert to 8-bit RGB and remove the alpha channel
    mel_spec_rgb = (mel_spec_rgb[..., :3] * 255).astype(np.uint8)  # Keep only the RGB channels
    
    return mel_spec_rgb

# Load the pre-trained model (ensure you have your model saved in the proper directory)
model = load_model('my_model.h5')  # Replace with the actual path to your model

# Streamlit UI components
st.title('Bird Species Prediction from Audio')

# Use a markdown description with instructions
st.markdown("""
    ## How to Use:
    1. Upload an audio file (max 10 seconds).
    2. The system will process the audio and predict the bird species.
    3. The result will be displayed below.
""")

# File uploader widget with a label
audio_file = st.file_uploader("Choose an audio file (ogg, mp3, wav)", type=["ogg", "mp3", "wav"])

# Display the file name when uploaded
if audio_file is not None:
    st.write(f"File uploaded: {audio_file.name}")
    
    # Display the audio file preview (First 10 seconds of audio)
    audio_bytes = audio_file.getvalue()
    st.audio(audio_bytes, format="audio/wav")

    # Process the audio file
    processed_audio = process_audio_as_rgb(audio_file)
    
    # Show mel spectrogram as image
    st.image(processed_audio, caption="Mel Spectrogram", use_column_width=True)

    # Ensure the shape matches the expected input shape (128, 431, 3)
    expected_shape = (128, 431, 3)
    if processed_audio.shape != expected_shape:
        # Resize or pad if necessary to match the expected shape
        audio_data_img = array_to_img(processed_audio)
        audio_data_img = audio_data_img.resize((431, 128))  # Resize to (431, 128)
        processed_audio = img_to_array(audio_data_img)
    
    # Add batch dimension and ensure data type
    processed_audio = np.expand_dims(processed_audio, axis=0)  # Add batch dimension
    processed_audio = processed_audio.astype(np.float32)

    # Model prediction with a loading spinner
    with st.spinner('Processing the audio...'):
        try:
            prediction = model.predict(processed_audio)
            pred_label = np.argmax(prediction, axis=1)
            st.success(f"Predicted bird species: {class_labels[pred_label[0]]}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.warning("Please upload an audio file to proceed.")
