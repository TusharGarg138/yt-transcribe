import os
import re
import torch
from yt_dlp import YoutubeDL
from moviepy.editor import VideoFileClip
import whisper
from transformers import BartTokenizer, BartForConditionalGeneration
import gradio as gr
from datetime import datetime
from gtts import gTTS


# Setup folders
DOWNLOAD_FOLDER = "downloads"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# ========== STEP 1: YouTube Download ==========
def download_video(url, path=DOWNLOAD_FOLDER):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(path, '%(title)s.%(ext)s'),
        'merge_output_format': 'mp4',
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_filename = os.path.join(path, f"{info['title']}.mp4")
    return video_filename

# ========== STEP 4: Preprocess Transcript ==========
def preprocess_transcript(transcript_path):
    with open(transcript_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,!? ]+', '', text)
    text = text.strip()
    preprocessed_path = transcript_path.replace(".txt", "_preprocessed.txt")
    with open(preprocessed_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text, preprocessed_path

# ========== STEP 2: Extract Audio ==========
def extract_audio(video_path):
    video_clip = VideoFileClip(video_path)
    audio_path = video_path.replace(".mp4", ".mp3")
    video_clip.audio.write_audiofile(audio_path)
    return audio_path


# ========== STEP 3: Transcribe ==========
def generate_transcript(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    transcript_path = audio_path.replace(".mp3", ".txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    return transcript_path
