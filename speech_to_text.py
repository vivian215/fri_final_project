import numpy as np
import sounddevice as sd
import whisper
import threading

command = ""

#openAI's whisper model
model = whisper.load_model("small")

SAMPLE_RATE = 16000  #16 kHz sample rate
BUFFER_DURATION = 2  #how long in seconds it waits before transcribing

audio_buffer = np.array([], dtype=np.float32)

stop_recording = False
finished_processing = False

#processes audio
def audio_processing():
    global audio_buffer
    while not stop_recording:
        if len(audio_buffer) >= SAMPLE_RATE * BUFFER_DURATION:
            #transcribe current chunk of audio
            audio_np = audio_buffer.copy() 
            audio_buffer = np.array([], dtype=np.float32) #clear buffer
            result = model.transcribe(audio_np, temperature=0)

            print("I heard:", result['text'])
            global command
            command = result['text']

def callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(status)

    #convert captured audio to float32
    audio_chunk = indata.copy().flatten().astype(np.float32)
    audio_buffer = np.append(audio_buffer, audio_chunk)

def main():
    #starts live recording
    with sd.InputStream(callback=callback, samplerate=SAMPLE_RATE, channels=1, dtype='float32'):
        print("Listening for a command...")

        processing_thread = threading.Thread(target=audio_processing)
        processing_thread.start()

        input("Press Enter to stop recording...")
        global stop_recording, finished_processing
        stop_recording = True
        print("Processing command...")
        # Wait for the audio processing thread to finish
        processing_thread.join()
        finished_processing = True

if __name__ == "__main__":
    main()