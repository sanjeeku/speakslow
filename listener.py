import os
import time
import queue
import threading
import tkinter as tk
from collections import deque
import json

import pyaudio
from vosk import Model, KaldiRecognizer

# --------------------------------------------------
# Configuration
# --------------------------------------------------
#MODEL_PATH = r"/Users/sanjeev/devtools/vosk-model-en-us-0.22"  # <-- Update this to your local Vosk model folder
MODEL_PATH = r"/Users/sanjeev/devtools/vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000
CHUNK = 1024

# Speech rate parameters
ROLLING_WINDOW_SIZE = 10  # Number of seconds to keep in rolling buffer
WORDS_PER_MINUTE_THRESHOLD = 180

# Stammering parameters
STAMMER_REPEAT_THRESHOLD = 2  # repeated words

# Confidence-based "garbled" tagging
GARBLED_CONFIDENCE_THRESHOLD = 0.6  # Adjust as needed

# --------------------------------------------------
# Global Variables
# --------------------------------------------------
audio_queue = queue.Queue()  # Thread-safe queue to hold audio data
running = True

# Rolling buffer to store (word, timestamp)
word_timestamp_deque = deque()

# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def compute_rolling_wpm():
    """
    Computes approximate words per minute based on
    the timestamps of words in the deque within the last ROLLING_WINDOW_SIZE seconds.
    """
    current_time = time.time()
    # Remove old entries from deque
    while word_timestamp_deque and (current_time - word_timestamp_deque[0][1]) > ROLLING_WINDOW_SIZE:
        word_timestamp_deque.popleft()

    num_words = len(word_timestamp_deque)
    # words / seconds * 60 = words per minute
    words_per_minute = (num_words / ROLLING_WINDOW_SIZE) * 60
    return words_per_minute

def analyze_stammering_and_articulation(words):
    """
    Naive approach:
      1. Detect repeated consecutive words to flag potential stammering.
      2. Check if no words recognized => possible unclear articulation.
    """
    stammer_count = 0
    for i in range(1, len(words)):
        if words[i].lower() == words[i-1].lower():
            stammer_count += 1

    stammer_alert = stammer_count >= STAMMER_REPEAT_THRESHOLD
    articulation_alert = (len(words) == 0)  # extremely naive

    return stammer_alert, stammer_count, articulation_alert


def parse_vosk_json(json_str):
    """
    Given a JSON string from rec.Result(), return:
      ( display_text, display_words, plain_words )

    - display_text: A string with garbled words tagged as [garbled: ...]
    - display_words: A list of the display-version words (including garbled tags)
    - plain_words: A list of the original words (no tags), used for stammer checks, etc.
    """
    try:
        data = json.loads(json_str)
        recognized_text = data.get("text", "")

        display_words = []
        plain_words = []

        if "result" in data:
            for item in data["result"]:
                w = item.get("word", "")
                conf = item.get("conf", 1.0)

                plain_words.append(w)

                # Tag if confidence is below threshold
                if conf < GARBLED_CONFIDENCE_THRESHOLD:
                    tagged = f"[garbled:{w}]"
                    display_words.append(tagged)
                else:
                    display_words.append(w)

        display_text = " ".join(display_words)
        return display_text, display_words, plain_words
    except:
        return "", [], []

def parse_partial_json(json_str):
    """
    Given a JSON string from rec.PartialResult(), return partial text if available.
    Example JSON:
      {"partial": "hello"}
    """
    try:
        data = json.loads(json_str)
        return data.get("partial", "")
    except:
        return ""


# --------------------------------------------------
# Audio Capture Thread
# --------------------------------------------------
def audio_capture_thread():
    """
    Captures audio from the default microphone and pushes raw audio into audio_queue.
    """
    global running

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    while running:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_queue.put(data)

    stream.stop_stream()
    stream.close()
    p.terminate()


# --------------------------------------------------
# Speech Recognition / Processing Thread
# --------------------------------------------------
def recognition_thread(gui_callback):
    """
    Pulls audio data from audio_queue, feeds into Vosk recognizer,
    and calls 'gui_callback' to update the UI with recognized text / alerts.
    """
    global running

    if not os.path.exists(MODEL_PATH):
        gui_callback("[ERROR] Vosk model path not found. Check MODEL_PATH.", alert=True)
        return

    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)  # Attempt to get word-level info if model supports it

    partial_text_accumulator = ""

    while running:
        try:
            data = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        if rec.AcceptWaveform(data):
            # Full utterance recognized
            result = rec.Result()

            # Parse recognized text and words (with garbled tags)
            display_text, display_words, plain_words = parse_vosk_json(result)

            if display_text.strip():
                now = time.time()
                for w in plain_words:
                    word_timestamp_deque.append((w, now))

                stammer_alert, stammer_count, articulation_alert = analyze_stammering_and_articulation(plain_words)
                wpm = compute_rolling_wpm()

                message = f"Recognized: {display_text}\n"
                message += f"Rolling WPM: {wpm:.2f}\n"

                alert_messages = []
                if wpm > WORDS_PER_MINUTE_THRESHOLD:
                    alert_messages.append(f"You might be speaking too fast! (>{WORDS_PER_MINUTE_THRESHOLD} WPM)")

                if stammer_alert:
                    alert_messages.append(f"Possible stammering. Repeated words: {stammer_count}")

                if articulation_alert:
                    alert_messages.append("Articulation might be unclear (no words recognized).")

                if alert_messages:
                    message += "\n".join(f"[ALERT] {m}" for m in alert_messages)

                gui_callback(message, alert=bool(alert_messages))
        else:
            # Partial result
            partial = rec.PartialResult()
            partial_text = parse_partial_json(partial)
            if partial_text and partial_text != partial_text_accumulator:
                partial_text_accumulator = partial_text
                gui_callback(f"[Partial] {partial_text}", alert=False)


# --------------------------------------------------
# GUI
# --------------------------------------------------
class SpeechMonitorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Enhanced Speech Monitor (Large Font)")

        # (1) Make the window larger by default
        self.root.geometry("900x600")

        # (2) Create a Text widget with larger fonts
        self.text_area = tk.Text(
            self.root, 
            width=80, 
            height=10, 
            wrap="word", 
            font=("Helvetica", 24)  # <-- Larger font for easy visibility
        )
        self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Tag for alerts with an even larger, bold font
        self.text_area.tag_config(
            "alert", 
            foreground="red", 
            font=("Helvetica", 20, "bold")
        )

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def run(self):
        self.root.mainloop()

    def on_close(self):
        global running
        running = False
        self.root.quit()
        self.root.destroy()

    def update_text(self, msg, alert=False):
        """
        Updates the text area with the new message. If 'alert' is True,
        highlight in red with a bigger bold font.
        """
        self.text_area.insert(tk.END, "\n" + ("=" * 60) + "\n")
        if alert:
            self.text_area.insert(tk.END, msg + "\n", "alert")
        else:
            self.text_area.insert(tk.END, msg + "\n")

        # Auto-scroll
        self.text_area.see(tk.END)


# --------------------------------------------------
# Main Function
# --------------------------------------------------
def main():
    gui = SpeechMonitorGUI()

    # Start audio capture thread
    t_audio = threading.Thread(target=audio_capture_thread, daemon=True)
    t_audio.start()

    # Start recognition thread
    t_rec = threading.Thread(
        target=recognition_thread,
        daemon=True,
        args=(gui.update_text,)
    )
    t_rec.start()

    # Start the GUI mainloop
    gui.run()


if __name__ == "__main__":
    main()
