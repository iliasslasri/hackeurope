import os
import io
import queue
import wave
import requests
import av
import numpy as np
import websocket
import json
import threading
import base64
import ssl

# ---- WebRTC Audio Streaming Processor (Gradium WebSocket) ----
class AudioStreamingProcessor:
    def __init__(self):
        self.text_queue = queue.Queue()
        
        # Standardize audio strictly to 24kHz, 16-bit PCM, Mono for Gradium
        self.resampler = av.AudioResampler(
            format='s16',
            layout='mono',
            rate=24000
        )
        self.ws_url = "wss://eu.api.gradium.ai/api/speech/asr"
        self.api_key = os.getenv("GRADIUM_API_KEY", "")
        self.ws = None
        self.ws_thread = None
        
        self.current_speaker = "Doctor" # Assumed first speaker
        self.text_buffer = [] # Holds the text for current turn
        self.is_ready = False # Prevent sending audio before server is ready
        self.frames_buffer = []

        self._connect_ws()

    def __del__(self):
        if hasattr(self, 'ws') and self.ws:
            try:
                self.ws.close()
            except:
                pass
        
    def _connect_ws(self):
        if not self.api_key:
            print("[Gradium] Missing GRADIUM_API_KEY in environment!")
            return
            
        def on_message(ws, message):
            try:
                msg = json.loads(message)
                msg_type = msg.get("type")
                
                if msg_type == "ready":
                    print("[Gradium] Server Ready! Can now send audio.")
                    self.is_ready = True
                    return
                    
                if msg_type == "text":
                    text = msg.get("text", "").strip()
                    if text:
                        # Stream text instantly instead of waiting for VAD turn
                        self.text_queue.put(text)
                            
            except Exception as e:
                print(f"[Gradium] Message parse error: {e}")
                
        def on_error(ws, error):
            print(f"[Gradium] WS Error details: {type(error).__name__} - {error}")
            
        def on_close(ws, close_status_code, close_msg):
            print(f"[Gradium] WS Closed. Code: {close_status_code}, Msg: {close_msg}")
            
        def on_open(ws):
            print("[Gradium] WS Connected! Sending setup...")
            setup_msg = { 
                "type": "setup", 
                "model_name": "default", 
                "input_format": "pcm",
                "language": "en"
            }
            ws.send(json.dumps(setup_msg))
            print("[Gradium] Setup sent.")
            
        def run_ws():
            import time
            retries = 3
            while retries > 0:
                try:
                    # Enable websocket trace logging for debugging
                    # websocket.enableTrace(True)
                    self.ws = websocket.WebSocketApp(
                        self.ws_url,
                        header=[f"x-api-key: {self.api_key}"],
                        on_open=on_open,
                        on_message=on_message,
                        on_error=on_error,
                        on_close=on_close
                    )
                    print("[Gradium] Starting WebSocket run_forever loop...")
                    self.ws.run_forever(
                        ping_interval=10, 
                        ping_timeout=5,
                        sslopt={"cert_reqs": ssl.CERT_NONE}
                    )
                    print("[Gradium] WebSocket run_forever exited.")
                    break
                except Exception as e:
                    print(f"[Gradium] Thread exception: {e}")
                    retries -= 1
                    time.sleep(2)
            
        self.ws_thread = threading.Thread(target=run_ws)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        if self.ws and self.ws.sock and self.ws.sock.connected and self.is_ready:
            try:
                # Resample frame into 24kHz standard format
                resampled_frames = self.resampler.resample(frame)
                
                for r_frame in resampled_frames:
                    raw_data = r_frame.to_ndarray().tobytes()
                    self.frames_buffer.append(raw_data)
                    
                # Gradium API allows small chunks, sending 40ms to drastically improve latency (960 samples @ 24kHz = 1920 bytes)
                bytes_per_chunk = 1920
                total_bytes = sum(len(b) for b in self.frames_buffer)
                
                if total_bytes >= bytes_per_chunk:
                    combined_raw = b''.join(self.frames_buffer)
                    self.frames_buffer = []
                    
                    b64_audio = base64.b64encode(combined_raw).decode('utf-8')
                    msg = {"type": "audio", "audio": b64_audio}
                    self.ws.send(json.dumps(msg))
                    
            except Exception as e:
                print(f"[Gradium] WS send error: {e}")
                
        return frame
