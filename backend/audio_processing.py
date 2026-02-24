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

# ---- WebRTC Audio Streaming Processor (Mistral Realtime) ----
import asyncio

class AudioStreamingProcessor:
    def __init__(self):
        self.text_queue = queue.Queue()
        
        # Standardize audio to 16kHz, 16-bit PCM, Mono for Mistral Realtime
        self.sample_rate = 16000
        self.resampler = av.AudioResampler(
            format='s16',
            layout='mono',
            rate=self.sample_rate
        )
        
        self.api_key = os.getenv("MISTRAL_API_KEY", "")
        self.client = None
        self.is_ready = False
        self.audio_queue = None
        self.loop = None
        self._background_thread = None

        if self.api_key:
            try:
                from mistralai import Mistral
                self.client = Mistral(api_key=self.api_key)
                print("[Mistral] Mistral client initialized.")
                self.start_streaming()
            except ImportError as e:
                print(f"[Mistral] Error initializing: {e}. Please run 'uv add mistralai[realtime]'.")
        else:
            print("[Mistral] Warning: MISTRAL_API_KEY missing in environment!")

    def start_streaming(self):
        self.loop = asyncio.new_event_loop()
        self.audio_queue = asyncio.Queue()
        self.is_ready = True
        
        self._background_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._background_thread.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._transcribe_task())

    def __del__(self):
        if self.is_ready and self.audio_queue and self.loop:
            # Send sentinel to stop the async generator cleanly
            asyncio.run_coroutine_threadsafe(self.audio_queue.put(None), self.loop)

    async def _queue_audio_iter(self):
        """Yield audio chunks from a queue until a None sentinel is received."""
        while True:
            chunk = await self.audio_queue.get()
            if chunk is None:
                break
            yield chunk

    async def _transcribe_task(self):
        if not self.client:
            return
            
        from mistralai.extra.realtime import UnknownRealtimeEvent
        from mistralai.models import (
            AudioFormat,
            RealtimeTranscriptionError,
            RealtimeTranscriptionSessionCreated,
            TranscriptionStreamDone,
            TranscriptionStreamTextDelta,
        )

        audio_format = AudioFormat(encoding="pcm_s16le", sample_rate=self.sample_rate)
        audio_stream = self._queue_audio_iter()

        try:
            print("[Mistral] Starting realtime transcription stream...")
            async for event in self.client.audio.realtime.transcribe_stream(
                audio_stream=audio_stream,
                model="voxtral-mini-transcribe-realtime-2602",
                audio_format=audio_format,
            ):
                if isinstance(event, RealtimeTranscriptionSessionCreated):
                    print(f"[Mistral Realtime] Session created.")
                elif isinstance(event, TranscriptionStreamTextDelta):
                    text = event.text.strip()
                    if text:
                        print(f"[Mistral Realtime Delta]: {text}")
                        self.text_queue.put(text)
                elif isinstance(event, TranscriptionStreamDone):
                    print("[Mistral Realtime] Transcription stream done.")
                    break
                elif isinstance(event, RealtimeTranscriptionError):
                    print(f"[Mistral Realtime Error]: {event}")
                    break
                elif isinstance(event, UnknownRealtimeEvent):
                    continue
        except Exception as e:
            print(f"[Mistral Realtime Task Exception]: {e}")

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        if self.is_ready and self.audio_queue and self.loop:
            try:
                # Resample frame into 16kHz standard format
                resampled_frames = self.resampler.resample(frame)
                
                for r_frame in resampled_frames:
                    raw_data = r_frame.to_ndarray().tobytes()
                    # Push to async queue safely from sync WebRTC thread
                    asyncio.run_coroutine_threadsafe(self.audio_queue.put(raw_data), self.loop)
                    
            except Exception as e:
                print(f"[Mistral] recv processing error: {e}")
                
        return frame
