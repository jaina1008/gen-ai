import streamlit as st
import torch
import pyttsx3
import sounddevice as sd
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import io
import tempfile
import scipy.io.wavfile as wav

# Initialize Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Initialize pyttsx3 TTS engine
engine = pyttsx3.init()

# Function to record audio from the microphone
def record_audio(duration=5, samplerate=16000):
    st.write("Recording audio... Please speak.")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    return audio.flatten().astype(np.float32) / 32768.0, samplerate  # Normalize to [-1, 1]

# Streamlit interface
st.title("Real-Time Speech-to-Text with AI Response")

# Button to start recording
if st.button("Start Talking"):
    audio_data, samplerate = record_audio()

    # Save the audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        wav.write(tmpfile.name, samplerate, audio_data)
        st.write(f"Audio recorded. Transcribing...")

        # Process audio using Wav2Vec 2.0
        input_values = processor(audio_data, return_tensors="pt", sampling_rate=samplerate).input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        # Show transcription
        st.write(f"Transcription: {transcription}")

        # Respond to the user (simple example response)
        response = f"You said: {transcription}. I'm here to help!"
        
        # Text-to-Speech: Convert the response to speech and play it
        engine.say(response)
        engine.runAndWait()
        
        st.write(f"AI says: {response}")
