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
