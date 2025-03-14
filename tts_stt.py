import whisper
from TTS.api import TTS
import librosa

# Load audio file using librosa
def transcribe_audio(file_path):
    model = whisper.load_model("base")
    audio, _ = librosa.load(file_path, sr=16000)  # Load audio and resample to 16kHz
    result = model.transcribe(audio)
    return result["text"]

# Generate speech from text
def generate_speech(text, output_path="output.wav"):
    """Convert text to speech and save as an audio file"""
    tts = TTS("tts_models/en/ljspeech/glow-tts")
    tts.tts_to_file(text, file_path=output_path)
    return output_path

if __name__ == "__main__":
    # Example usage
    print(transcribe_audio("Conference.wav"))  # Transcribe a sample file
    generate_speech("Hey, how are you? can i include my wife in the insurance plan?", "output.wav")  # Generate speech
