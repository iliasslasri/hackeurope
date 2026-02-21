import av
import queue
import wave
import io
import os
import requests

def transcribe_chunk(audio_data: bytes):
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": os.getenv("ELEVENLABS_API_KEY", "")}
    files = {"file": ("audio.wav", audio_data, "audio/wav")}
    data = {"model_id": "scribe_v1", "diarize": "true"}
    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        if response.status_code == 200:
            words = response.json().get("words", [])
            return " ".join([w.get("text", "") for w in words])
        return ""
    except Exception:
        return ""

class AudioProcessor:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.frames_buffer = []
        self.sample_rate = 48000
        self.channels = 2
        self.sample_width = 2
        
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Save audio parameters
        self.sample_rate = frame.sample_rate
        self.channels = len(frame.layout.channels)
        self.sample_width = frame.format.bytes

        raw_data = frame.to_ndarray().tobytes()
        self.frames_buffer.append(raw_data)
        
        # Suppose 48000 frames/sec * 2 bytes * 2 channels = 192000 bytes/sec
        bytes_per_sec = self.sample_rate * self.channels * self.sample_width
        
        total_bytes = sum(len(b) for b in self.frames_buffer)
        
        # Every ~3 seconds, emit chunk
        if total_bytes >= bytes_per_sec * 3:
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.sample_width)
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.frames_buffer))
            
            self.audio_queue.put(wav_io.getvalue())
            self.frames_buffer = []
            
        return frame
