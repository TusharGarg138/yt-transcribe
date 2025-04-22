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
