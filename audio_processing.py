import os
import io
import queue
import wave
import requests
import av
import torch
import numpy as np
from scipy.spatial.distance import cdist

# Fallback STT using ElevenLabs (no internal diarization, we handle it via Pyannote)
def transcribe_audio_chunk(audio_data: bytes):
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": os.getenv("ELEVENLABS_API_KEY", "")}
    files = {"file": ("audio.wav", audio_data, "audio/wav")}
    # Diarize is FALSE because we will use Pyannote to map IDs across chunks
    data = {"model_id": "scribe_v1", "diarize": "false"}
    
    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        if response.status_code == 200:
            return response.json().get("text", "")
        else:
            print(f"[Transcription Failed: {response.text}]")
            return ""
    except Exception as e:
        print(f"[Transcription Error: {e}]")
        return ""

# ---- Pyannote Offline Diarization & Speaker Tracking ----
# Requires HF_TOKEN in .env for pyannote models
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Load PyTorch Embedding Model
try:
    from pyannote.audio import Model
    from pyannote.audio import Inference
    
    # We use the embedding model to get voice fingerprints
    print("Loading Pyannote Embedding Model...")
    embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN)
    speaker_inference = Inference(embedding_model, window="whole")
except Exception as e:
    print(f"Warning: Failed to load pyannote embedding model. Are you missing HF_TOKEN? Error: {e}")
    speaker_inference = None

# Global dictionary to lock in Doctor and Patient IDs across the entire session
known_speaker_embeddings = {}

def get_speaker_role(waveform, sample_rate):
    """
    Given a raw waveform of a speaker, computes their embedding and matches it
    to either 'Doctor' or 'Patient' permanently for the session.
    """
    if speaker_inference is None:
        return "Unknown"
        
    try:
        # waveform should be (channels, frames) tensor
        wav_tensor = torch.from_numpy(waveform).float()
        if wav_tensor.ndim == 1:
             wav_tensor = wav_tensor.unsqueeze(0)
             
        # Extract voice fingerprint for this chunk
        emb = speaker_inference({"waveform": wav_tensor, "sample_rate": sample_rate})
        
        # Ensure embedding is 1D for distance calculations
        emb = np.squeeze(emb)
        
        global known_speaker_embeddings
        
        # If no speakers are known yet, the first person to speak is assigned 'Doctor'
        if not known_speaker_embeddings:
            known_speaker_embeddings["Doctor"] = emb
            return "Doctor"
            
        # Compare with known profiles using Cosine distance
        best_match = None
        best_dist = float('inf')
        
        for role, profile_emb in known_speaker_embeddings.items():
            dist = cdist([emb], [profile_emb], metric="cosine")[0][0]
            if dist < best_dist:
                best_dist = dist
                best_match = role
                
        # If the distance is significant (> 0.4) and we don't have a Patient yet, assign 'Patient'
        if best_dist > 0.4 and "Patient" not in known_speaker_embeddings:
            known_speaker_embeddings["Patient"] = emb
            return "Patient"
            
        # Slowly update the running average of the matched speaker's voice profile
        known_speaker_embeddings[best_match] = 0.8 * known_speaker_embeddings[best_match] + 0.2 * emb
        return best_match
        
    except Exception as e:
        print(f"Embedding mapping error: {e}")
        return "Unknown"

# ---- WebRTC Audio Streaming Processor ----
class AudioStreamingProcessor:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.frames_buffer = []
        self.sample_rate = 48000
        self.channels = 1
        self.sample_width = 2
        
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.sample_rate = frame.sample_rate
        self.channels = len(frame.layout.channels)
        self.sample_width = frame.format.bytes

        # Convert to mono for pyannote compatibility
        if self.channels > 1:
            frame_nd = frame.to_ndarray()
            mono_data = np.mean(frame_nd, axis=0).astype(np.int16)
            raw_data = mono_data.tobytes()
            self.channels = 1
        else:
            raw_data = frame.to_ndarray().tobytes()
            
        self.frames_buffer.append(raw_data)
        
        bytes_per_sec = self.sample_rate * self.channels * self.sample_width
        total_bytes = sum(len(b) for b in self.frames_buffer)
        
        # Process every ~3.5 seconds
        if total_bytes >= bytes_per_sec * 3.5:
            combined_raw = b''.join(self.frames_buffer)
            
            # Export to pure WAV byte stream for ElevenLabs
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.sample_width)
                wf.setframerate(self.sample_rate)
                wf.writeframes(combined_raw)
            wav_bytes = wav_io.getvalue()
            
            # Create numpy waveform for Pyannote
            audio_np = np.frombuffer(combined_raw, dtype=np.int16).astype(np.float32) / 32768.0
            
            self.audio_queue.put({
                "wav_bytes": wav_bytes,
                "waveform": audio_np,
                "sample_rate": self.sample_rate
            })
            
            self.frames_buffer = []
            
        return frame
