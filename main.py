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

# ========== STEP 5: Summarize ==========
def chunk_text(text, max_length=1024):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def summarize_transcript(text, model_name="facebook/bart-large-cnn"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    text_chunks = chunk_text(text)
    summaries = []
    for chunk in text_chunks:
        tokens = tokenizer(chunk, truncation=True, padding="longest", max_length=1024, return_tensors="pt").to(device)
        summary_ids = model.generate(**tokens, max_length=200, num_beams=5, length_penalty=2.0, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    full_summary = " ".join(summaries)
    bullet_summary = "\n".join([f"• {sentence.strip()}" for sentence in full_summary.split(". ") if sentence])
    print("[INFO] Summary generated successfully.")
    summary = bullet_summary
    return summary
