import os
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel

import voice_service as vs
from rag.AIVoiceAssistant import AIVoiceAssistant

DEFAULT_MODEL_SIZE = "medium"
DEFAULT_CHUNK_LENGTH = 2  # Shorter chunks for more responsive detection

ai_assistant = AIVoiceAssistant()


def is_silence(data, max_amplitude_threshold=8000):
    """Check if audio data contains silence."""
    max_amplitude = np.max(np.abs(data))
    print(f"Audio amplitude: {max_amplitude} (threshold: {max_amplitude_threshold})")
    return max_amplitude <= max_amplitude_threshold


def record_audio_chunk(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    frames = []
    
    # Record with error handling for buffer overflow
    try:
        for _ in range(0, int(16000 / 1024 * chunk_length)):
            try:
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)
            except OSError as e:
                if "Input overflowed" in str(e):
                    print("Buffer overflow - continuing...")
                    continue
                else:
                    raise e
    except Exception as e:
        print(f"Error during recording: {e}")
        return True  # Return True (silence) on error to continue listening

    if not frames:
        return True  # No audio data, treat as silence

    temp_file_path = 'temp_audio_chunk.wav'
    try:
        with wave.open(temp_file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(frames))

        # Check if the recorded chunk contains silence
        samplerate, data = wavfile.read(temp_file_path)
        is_silent = is_silence(data)
        
        if is_silent:
            os.remove(temp_file_path)
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error while processing audio file: {e}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return True  # Treat as silence on error

    

def transcribe_audio(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription


def main():
    
    model_size = DEFAULT_MODEL_SIZE + ".en"
    model = WhisperModel(model_size, device="cpu", compute_type="int8", num_workers=4)
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    customer_input_transcription = ""

    try:
        while True:
            chunk_file = "temp_audio_chunk.wav"
            
            # Record audio chunk
            print("_")
            if not record_audio_chunk(audio, stream):
                # Transcribe audio
                transcription = transcribe_audio(model, chunk_file)
                os.remove(chunk_file)
                print("Customer:{}".format(transcription))
                
                # Add customer input to transcript
                customer_input_transcription += "Customer: " + transcription + "\n"
                
                # Process customer input and get response from AI assistant
                output = ai_assistant.interact_with_llm(transcription)
                if output:
                    output = output.lstrip()
                    vs.play_text_to_speech(output)
                    print("AI Assistant:{}".format(output))


    
    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()