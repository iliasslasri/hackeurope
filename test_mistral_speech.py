import os
import av
import wave
import queue
import time
import asyncio
from backend.audio_processing import AudioStreamingProcessor
from dotenv import load_dotenv

load_dotenv()

async def main():
    processor = AudioStreamingProcessor()
    
    print("Initializing...")
    await asyncio.sleep(1) # wait for ws connect
    
    print("Generating speech audio with gTTS...")
    from gtts import gTTS
    tts = gTTS("Hello doctor, I have been feeling a sharp pain in my chest since yesterday morning.", lang='en')
    tts.save("test_speech.mp3")
    
    print("Reading and sending audio...")
    try:
        container = av.open('test_speech.mp3')
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            processor.recv(frame)
            # Small delay per frame to simulate realtime pacing
            await asyncio.sleep(frame.samples / frame.sample_rate)
    except Exception as e:
        print("Error reading audio:", e)
    
    print("Waiting for transcription...")
    for _ in range(10):
        while not processor.text_queue.empty():
            print("=>", processor.text_queue.get())
        await asyncio.sleep(1)
        
if __name__ == "__main__":
    asyncio.run(main())
