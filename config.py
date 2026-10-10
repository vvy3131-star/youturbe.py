import os

# ==========================
# API
# ==========================

# Khi deploy Streamlit Cloud sẽ đọc từ secrets.toml
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ==========================
# MODEL
# ==========================

WHISPER_MODEL = "large-v3"

DEVICE = "cpu"
# Nếu có GPU NVIDIA:
# DEVICE = "cuda"

COMPUTE_TYPE = "int8"

# ==========================
# THƯ MỤC
# ==========================

UPLOAD_FOLDER = "uploads"
TEMP_FOLDER = "temp"
OUTPUT_FOLDER = "outputs"

# ==========================
# VIDEO
# ==========================

SUPPORTED_VIDEO = [
    "mp4",
    "mov",
    "avi",
    "mkv"
]

# ==========================
# PHỤ ĐỀ
# ==========================

FONT_SIZE = 32

FONT_COLOR = "white"

OUTLINE_COLOR = "black"

OUTLINE_WIDTH = 2

FONT_NAME = "Arial"

POSITION = "bottom"

# ==========================
# AUDIO
# ==========================

SAMPLE_RATE = 16000

CHANNEL = 1

# ==========================
# WHISPER
# ==========================

LANGUAGE = "zh"

BEAM_SIZE = 5

VAD_FILTER = True

# ==========================
# TỰ ĐỘNG TẠO THƯ MỤC
# ==========================

for folder in [
    UPLOAD_FOLDER,
    TEMP_FOLDER,
    OUTPUT_FOLDER
]:
    os.makedirs(folder, exist_ok=True)
[server]
maxUploadSize = 2048
maxMessageSize = 2048

[browser]
gatherUsageStats = false

[client]
toolbarMode = "minimal"

[theme]
base = "dark"
primaryColor = "#8b5cf6"
backgroundColor = "#0b1020"
secondaryBackgroundColor = "#141a33"
textColor = "#e8ecf8"
borderColor = "#2a3358"
font = "sans serif"
baseRadius = "large"
