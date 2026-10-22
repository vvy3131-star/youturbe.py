"""
Dịch & lồng tiếng video Trung → Việt — BẢN TỰ ĐỘNG HOÁ + VIRAL BOOST + VIENEU-TTS.

Người dùng chỉ cần: tải video → chọn cỡ chữ → chọn màu chữ → chọn giọng đọc.
Phần còn lại (phát hiện & làm mờ phụ đề gốc, lồng tiếng AI, hạ âm lượng gốc,
màu viền tương phản, tốc độ xử lý...) được xử lý tự động.

BẢN NÀY CÓ GÌ MỚI
    - THÊM GIỌNG AI VieNeu-TTS (github.com/pnnbao97/VieNeu-TTS): giọng Việt tự nhiên, chạy offline
      trên CPU (ONNX), nằm chung danh sách giọng với edge-tts. Có nút quét toàn bộ giọng có sẵn.
      Nếu VieNeu chưa cài / lỗi → tự chuyển sang edge-tts rồi gTTS nên video không bao giờ hỏng.
    - SỬA LỖI KHÔNG DỊCH RA TIẾNG VIỆT: dịch bằng 3 nguồn dự phòng (Google trực tiếp →
      deep_translator → MyMemory), kiểm tra kết quả thật sự là tiếng Việt, tự thử lại,
      báo lỗi rõ ràng thay vì âm thầm giữ chữ Trung.
    - SỬA LỖI KHÔNG LỒNG TIẾNG AI: chạy edge-tts ổn định hơn, hiển thị lỗi thật,
      tự chuyển sang gTTS nếu edge-tts hỏng, ghép track giọng đọc bằng numpy.
    - NHẬN DIỆN & LỌC ÂM THANH TỐT HƠN: lọc nhiễu/cân bằng âm trước khi nhận diện,
      model mặc định "small", tham số Whisper/VAD tối ưu cho tiếng Trung, lọc câu "ảo giác",
      gộp câu quá ngắn.
    - 12 tính năng "Viral Boost": lật gương, zoom nhẹ, tăng màu & nét, tăng tốc nhẹ,
      tiêu đề hook 3 giây đầu, watermark tên kênh, che logo góc, thanh tiến trình,
      fade vào/ra, chuẩn hoá âm lượng, khung dọc 9:16 Shorts, thumbnail tự động,
      gói YouTube (tiêu đề / mô tả / chương / hashtag).

CÀI ĐẶT:
    pip install -r requirements.txt        # cần streamlit >= 1.50
    pip install -U edge-tts gTTS deep-translator
    pip install -U vieneu                  # (tuỳ chọn) giọng VieNeu-TTS, lần đầu chạy sẽ tự tải model
    # cần ffmpeg: sudo apt install ffmpeg / brew install ffmpeg

CHẠY:
    streamlit run app.py

CẤU TRÚC THƯ MỤC:
    app.py
    requirements.txt            (thêm dòng: vieneu)
    packages.txt                (Streamlit Cloud: ffmpeg, fonts-dejavu-core)
    .streamlit/config.toml      (theme tối + nâng giới hạn upload lên 2GB)
"""

import asyncio
import hashlib
import html as html_lib
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import threading
import time
import urllib.parse
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

st.set_page_config(
    page_title="Dịch & Lồng tiếng video Trung → Việt",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# GIAO DIỆN (CSS) — chữ to, đậm, tương phản cao
# ============================================================
APP_CSS = """
<style>
:root{
  --accent:#8b5cf6; --accent2:#22d3ee; --accent3:#f472b6;
  --card:rgba(255,255,255,.055); --card-border:rgba(255,255,255,.14);
  --text:#f4f6ff; --muted:#c9d1ee;
  color-scheme: dark;
}
html, body, .stApp{color:var(--text); font-size:17px;}
.stApp{
  background:
    radial-gradient(900px 500px at 8% -5%, rgba(139,92,246,.28), transparent 60%),
    radial-gradient(800px 500px at 100% 0%, rgba(34,211,238,.18), transparent 55%),
    radial-gradient(700px 500px at 50% 110%, rgba(244,114,182,.16), transparent 60%),
    #0b1020;
  background-attachment: fixed;
}
header[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:880px; padding-top:2rem; padding-bottom:4rem;}

/* ----- Chữ: ép màu sáng + cỡ to để luôn dễ đọc, kể cả khi Streamlit đang ở theme sáng ----- */
.stApp p, .stApp li, .stApp label, .stApp span,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMarkdownContainer"] *{
  color:var(--text);
}
.stApp [data-testid="stWidgetLabel"] p, .stApp [data-testid="stWidgetLabel"] label{
  font-size:1.05rem !important; font-weight:650 !important; color:#ffffff !important;
}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p{
  color:var(--muted) !important; font-size:.95rem !important; line-height:1.5;
}
.stApp [data-testid="stTooltipIcon"] svg{color:var(--accent2) !important;}
.stApp [data-testid="stExpander"] summary p{font-size:1.08rem !important; font-weight:650 !important; color:#fff !important;}
.stApp .stTabs [data-baseweb="tab"] p{font-size:1.02rem !important; font-weight:650 !important;}
.stApp [data-testid="stButtonGroup"] button p{font-size:.98rem !important; font-weight:600 !important;}
.stApp [data-testid="stMetricValue"]{color:#fff !important; font-weight:700;}
.stApp [data-testid="stMetricLabel"] p{color:var(--muted) !important; font-size:.92rem !important;}
.stApp [data-testid="stSliderThumbValue"]{color:#fff !important; font-weight:700;}
.stApp [data-testid="stSliderTickBarMin"], .stApp [data-testid="stSliderTickBarMax"]{color:var(--muted) !important;}

/* ----- Ô nhập liệu / dropdown: nền tối, chữ trắng ----- */
.stApp [data-baseweb="input"] > div, .stApp [data-baseweb="base-input"],
.stApp [data-baseweb="select"] > div, .stApp [data-baseweb="textarea"]{
  background:rgba(255,255,255,.08) !important; border-color:rgba(255,255,255,.25) !important; border-radius:12px !important;
}
.stApp input, .stApp textarea, .stApp [data-baseweb="select"] *{
  color:#ffffff !important; -webkit-text-fill-color:#ffffff !important; font-size:1rem !important;
}
.stApp input::placeholder, .stApp textarea::placeholder{color:#9ea8cf !important; -webkit-text-fill-color:#9ea8cf !important;}
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] [role="listbox"]{background:#171e3d !important;}
div[data-baseweb="popover"] li, div[data-baseweb="popover"] li *{color:#f4f6ff !important; font-size:1rem !important;}
div[data-baseweb="popover"] li:hover{background:rgba(139,92,246,.35) !important;}

/* ---------- HERO ---------- */
.hero{
  position:relative; overflow:hidden; border-radius:26px; padding:34px 30px 28px;
  background:linear-gradient(135deg, rgba(139,92,246,.38), rgba(34,211,238,.20) 55%, rgba(244,114,182,.24));
  border:1px solid rgba(255,255,255,.18);
  box-shadow:0 20px 60px -20px rgba(139,92,246,.55);
  margin-bottom:1.4rem;
}
.hero:before{
  content:""; position:absolute; inset:-40% -10% auto auto; width:360px; height:360px; border-radius:50%;
  background:radial-gradient(circle, rgba(255,255,255,.18), transparent 65%);
}
.hero .badge{
  display:inline-block; font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; font-weight:800;
  padding:5px 12px; border-radius:999px; background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.28);
  margin-bottom:12px; color:#fff;
}
.hero h1{
  margin:0 0 10px; font-size:2.3rem; line-height:1.15; font-weight:800; letter-spacing:-.02em;
  background:linear-gradient(90deg,#fff,#d8ccff 45%,#8bf0ff); -webkit-background-clip:text; background-clip:text;
  color:transparent !important; -webkit-text-fill-color:transparent;
}
.hero p{margin:0; color:#e6eaff !important; font-size:1.05rem; max-width:640px; line-height:1.55;}
.hero .chips{display:flex; flex-wrap:wrap; gap:8px; margin-top:16px;}
.hero .chip{
  font-size:.9rem; font-weight:600; padding:7px 13px; border-radius:999px; color:#fff !important;
  background:rgba(8,12,28,.55); border:1px solid rgba(255,255,255,.22);
}

/* ---------- CARD ---------- */
[class*="st-key-card_"]{
  background:var(--card); border:1px solid var(--card-border); border-radius:22px;
  padding:22px 22px 18px; margin-bottom:1.1rem;
  backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  box-shadow:0 10px 34px -18px rgba(0,0,0,.7);
  transition:border-color .25s ease;
}
[class*="st-key-card_"]:hover{border-color:rgba(139,92,246,.5);}

.step-head{display:flex; align-items:center; gap:14px; margin-bottom:10px;}
.step-num{
  flex:0 0 auto; width:42px; height:42px; border-radius:13px; display:grid; place-items:center;
  font-weight:800; font-size:1.1rem; color:#fff; background:linear-gradient(135deg,var(--accent),var(--accent2));
  box-shadow:0 8px 20px -6px rgba(139,92,246,.7);
}
.step-title{font-size:1.35rem; font-weight:800; margin:0; line-height:1.2; color:#fff !important;}
.step-desc{color:var(--muted) !important; font-size:.98rem; margin:3px 0 0;}

/* ---------- WIDGETS ---------- */
[data-testid="stFileUploaderDropzone"]{
  border:2px dashed rgba(139,92,246,.65); border-radius:18px; padding:26px;
  background:linear-gradient(135deg, rgba(139,92,246,.12), rgba(34,211,238,.07));
  transition:all .25s ease;
}
[data-testid="stFileUploaderDropzone"]:hover{
  border-color:var(--accent2); background:linear-gradient(135deg, rgba(139,92,246,.2), rgba(34,211,238,.13));
}
[data-testid="stFileUploaderDropzone"] *{color:#f4f6ff !important;}
.stButton>button, .stDownloadButton>button{
  border-radius:14px; font-weight:700; font-size:1rem; padding:.7rem 1.1rem; transition:all .2s ease;
  border:1px solid rgba(255,255,255,.22);
}
.stButton>button:hover, .stDownloadButton>button:hover{
  transform:translateY(-2px); border-color:var(--accent2); box-shadow:0 10px 24px -10px rgba(34,211,238,.6);
}
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"]{
  border:none; color:#fff; font-size:1.12rem; padding:.95rem 1.2rem;
  background:linear-gradient(90deg,#8b5cf6,#6366f1 50%,#22d3ee);
  box-shadow:0 14px 34px -12px rgba(99,102,241,.9);
}
.stButton>button[kind="primary"]:hover{filter:brightness(1.1); box-shadow:0 18px 40px -12px rgba(34,211,238,.8);}
.stButton>button:disabled{opacity:.45; transform:none; box-shadow:none;}
[data-testid="stImage"] img, [data-testid="stVideo"] video{
  border-radius:16px; border:1px solid rgba(255,255,255,.14); box-shadow:0 18px 40px -22px #000;
}
[data-testid="stMetric"]{
  background:rgba(255,255,255,.06); border:1px solid var(--card-border); border-radius:16px; padding:12px 16px;
}
[data-testid="stExpander"]{border-radius:16px; border:1px solid var(--card-border); background:rgba(255,255,255,.035);}
[data-testid="stStatus"]{border-radius:16px;}
.stTabs [data-baseweb="tab-list"]{gap:6px;}
.stTabs [data-baseweb="tab"]{border-radius:12px 12px 0 0; padding:.55rem 1rem;}
.legend{display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 2px;}
.legend span{
  font-size:.88rem; font-weight:600; padding:6px 12px; border-radius:999px; background:rgba(255,255,255,.08);
  border:1px solid var(--card-border); color:#f4f6ff !important;
}
.note{
  font-size:.92rem; color:#ffe9a8 !important; background:rgba(255,200,60,.10);
  border:1px solid rgba(255,200,60,.35); border-radius:12px; padding:9px 13px; margin-top:8px;
}
.footer{text-align:center; color:var(--muted) !important; font-size:.85rem; margin-top:2rem;}

@media (max-width:640px){
  .hero{padding:24px 18px;} .hero h1{font-size:1.65rem;}
  [class*="st-key-card_"]{padding:16px 14px 12px;}
}
</style>
"""
st.html(APP_CSS)


def step_header(num, title: str, desc: str = ""):
    desc_html = f'<p class="step-desc">{desc}</p>' if desc else ""
    st.html(
        f'<div class="step-head"><div class="step-num">{num}</div>'
        f'<div><p class="step-title">{title}</p>{desc_html}</div></div>'
    )


st.html("""
<div class="hero">
  <span class="badge">AI · Video Studio</span>
  <h1>Dịch & lồng tiếng video<br>Trung → Việt</h1>
  <p>Tải video lên, chọn vài tuỳ chọn — phần còn lại AI lo: nhận diện giọng nói, dịch, che phụ đề gốc,
  ghi phụ đề mới, lồng tiếng Việt và tăng sức hút cho video.</p>
  <div class="chips">
    <span class="chip">🔎 Tự phát hiện phụ đề gốc</span>
    <span class="chip">🎙️ Lồng tiếng AI</span>
    <span class="chip">🦜 Giọng VieNeu-TTS</span>
    <span class="chip">🚀 Viral Boost</span>
    <span class="chip">🖼️ Thumbnail & mô tả YouTube</span>
  </div>
</div>
""")

# ---------- DANH SÁCH GIỌNG ĐỌC (edge-tts) ----------
VOICE_STYLE_OPTIONS = {
    "👩 Hoài My - Chuẩn":            {"voice": "vi-VN-HoaiMyNeural",  "rate": 0,   "pitch": 0},
    "👩 Hoài My - Nhẹ nhàng, chậm":   {"voice": "vi-VN-HoaiMyNeural",  "rate": -15, "pitch": -3},
    "👩 Hoài My - Vui tươi, nhanh":   {"voice": "vi-VN-HoaiMyNeural",  "rate": 15,  "pitch": 3},
    "👩 Hoài My - Trầm ấm":          {"voice": "vi-VN-HoaiMyNeural",  "rate": -5,  "pitch": -6},
    "👨 Nam Minh - Chuẩn":           {"voice": "vi-VN-NamMinhNeural", "rate": 0,   "pitch": 0},
    "👨 Nam Minh - Trầm, chậm rãi":  {"voice": "vi-VN-NamMinhNeural", "rate": -15, "pitch": -6},
    "👨 Nam Minh - Nhanh, dứt khoát": {"voice": "vi-VN-NamMinhNeural", "rate": 15,  "pitch": 2},
    "👨 Nam Minh - Trẻ trung, cao":  {"voice": "vi-VN-NamMinhNeural", "rate": 5,   "pitch": 6},
}

# ---------- GIỌNG VieNeu-TTS (github.com/pnnbao97/VieNeu-TTS) ----------
# Danh sách mặc định (đã có sẵn trong gói vieneu). Bấm nút "Quét giọng VieNeu" trong phần
# Tuỳ chỉnh nâng cao để lấy toàn bộ giọng có thật trong bản vieneu đang cài trên máy.
VIENEU_STATIC_VOICES = [("Hải Đăng", "Hải Đăng"), ("Minh Quân", "Minh Quân")]
EDGE_FALLBACK_VOICE = "vi-VN-HoaiMyNeural"   # giọng dự phòng khi VieNeu chưa cài / lỗi


def get_voice_options():
    """Giọng edge-tts + giọng VieNeu-TTS (nếu đã quét thì dùng danh sách quét được)."""
    options = dict(VOICE_STYLE_OPTIONS)
    vieneu_voices = st.session_state.get("vieneu_voices") or VIENEU_STATIC_VOICES
    for label, voice_id in vieneu_voices:
        options[f"🦜 VieNeu - {label}"] = {"engine": "vieneu", "voice": voice_id, "rate": 0, "pitch": 0}
    return options


POSITION_OPTIONS = {"Dưới (mặc định)": 2, "Giữa màn hình": 5, "Trên": 8}

# beam_size lớn hơn = nhận diện chính xác hơn; tts_concurrency vừa phải để không bị Microsoft chặn.
SPEED_PRESETS = {
    "⚡ Nhanh nhất": {"beam_size": 3, "video_preset": "ultrafast", "tts_concurrency": 8},
    "⚖️ Cân bằng": {"beam_size": 5, "video_preset": "veryfast", "tts_concurrency": 6},
    "🎯 Chính xác nhất": {"beam_size": 8, "video_preset": "medium", "tts_concurrency": 4},
}
DEFAULT_SPEED_LABEL = "⚖️ Cân bằng"
DEFAULT_POSITION_LABEL = "Dưới (mặc định)"

LOGO_CORNERS = {
    "Không che": None,
    "↖ Trên trái": "tl",
    "↗ Trên phải": "tr",
    "↙ Dưới trái": "bl",
    "↘ Dưới phải": "br",
}

CPU_COUNT = os.cpu_count() or 4

SAMPLE_SUBTITLE_TEXT = "Đây là câu phụ đề mẫu để xem trước"
SAMPLE_TTS_TEXT = "Xin chào, đây là giọng đọc mẫu để bạn nghe thử trước khi xử lý toàn bộ video."

AUTO_BG_VOLUME_PCT = 8
AUTO_NO_DUB_VOLUME_PCT = 100

MAX_TTS_SPEEDUP = 1.15
TTS_SAMPLE_RATE = 24000

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def find_font_path():
    for c in FONT_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


# ============================================================
# CÁC HÀM XỬ LÝ CHUNG
# ============================================================

class FFmpegError(RuntimeError):
    """Lỗi ffmpeg kèm vài dòng cuối của stderr để hiển thị cho người dùng."""
    pass


def _run_ffmpeg(cmd, label: str = "ffmpeg"):
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        stderr_tail = "\n".join(result.stderr.strip().splitlines()[-15:]) if result.stderr else "(không có thông tin lỗi)"
        raise FFmpegError(f"Bước '{label}' thất bại (mã lỗi {result.returncode}).\n\nChi tiết ffmpeg:\n{stderr_tail}")
    return result


# Chuỗi lọc âm thanh trước khi nhận diện: cắt tiếng ù/rè thấp, cắt hiss cao, khử nhiễu nền, cân bằng âm lượng.
AUDIO_ENHANCE_FILTERS = [
    "highpass=f=90,lowpass=f=7600,afftdn=nf=-25,dynaudnorm=f=250:g=13:p=0.9",
    "highpass=f=90,lowpass=f=7600,dynaudnorm=f=250:g=13:p=0.9",
]


def extract_audio(video_path: str, audio_path: str, enhance: bool = True):
    """Tách âm thanh 16kHz mono. Nếu enhance=True: lọc nhiễu + cân bằng âm lượng để nhận diện chính xác hơn.
    Nếu ffmpeg thiếu bộ lọc, tự động thử bộ lọc đơn giản hơn rồi mới tới bản không lọc."""
    if not has_audio_stream(video_path):
        raise FFmpegError("Video này không có âm thanh nên không thể nhận diện giọng nói.")

    base = ["ffmpeg", "-y", "-i", video_path, "-vn", "-map", "0:a:0"]
    tail = ["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]

    attempts = []
    if enhance:
        for af in AUDIO_ENHANCE_FILTERS:
            attempts.append(base + ["-af", af] + tail)
    attempts.append(base + tail)

    last_err = None
    for cmd in attempts:
        try:
            _run_ffmpeg(cmd, label="tách âm thanh")
            return
        except FFmpegError as e:
            last_err = e
    raise last_err


def ffprobe_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def has_audio_stream(path: str) -> bool:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", path],
            capture_output=True, text=True,
        )
        return "audio" in r.stdout
    except Exception:
        return False


def get_video_resolution(path: str):
    """Trả về (width, height), hoặc None nếu ffprobe không đọc được."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
             "-of", "csv=s=x:p=0", path],
            capture_output=True, text=True,
        )
        w_str, h_str = result.stdout.strip().split("x")
        return int(w_str), int(h_str)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=CPU_COUNT)


def load_wav_as_float32(path: str) -> "np.ndarray":
    """Đọc file wav 16kHz mono (do ffmpeg tạo) thành mảng float32 [-1, 1].

    SỬA LỖI "open() got an unexpected keyword argument 'metadata_errors'":
    lỗi này xảy ra vì faster-whisper gọi av.open(..., metadata_errors=...) khi đưa vào
    đường dẫn file, mà bản PyAV cài trên máy quá cũ không có tham số đó. Ở đây ta đã có
    sẵn wav chuẩn từ ffmpeg nên đọc trực tiếp và truyền mảng numpy — bỏ qua hoàn toàn PyAV.
    """
    with wave.open(path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# Các câu "ảo giác" quen thuộc của Whisper khi gặp đoạn im lặng / nhạc nền.
_HALLUCINATION_HINTS = (
    "字幕", "订阅", "訂閱", "点赞", "點贊", "谢谢观看", "謝謝觀看", "感谢观看", "感謝觀看",
    "请不吝", "請不吝", "Amara", "打赏", "打賞", "下集再见", "下期再见", "明镜", "明鏡",
)


def _is_hallucination(seg) -> bool:
    text = seg.text.strip()
    if not text:
        return True
    no_speech = getattr(seg, "no_speech_prob", 0.0) or 0.0
    avg_lp = getattr(seg, "avg_logprob", 0.0) or 0.0
    if no_speech > 0.8 and avg_lp < -1.0:
        return True
    if any(h in text for h in _HALLUCINATION_HINTS) and (no_speech > 0.4 or avg_lp < -0.8):
        return True
    return False


def merge_short_segments(segs, max_gap: float = 0.35, max_len: float = 8.0, min_dur: float = 1.5):
    """Gộp các câu quá ngắn, sát nhau để dịch đủ ngữ cảnh và giọng đọc tự nhiên hơn."""
    out = []
    for s in segs:
        if out:
            p = out[-1]
            gap = s["start"] - p["end"]
            p_dur = p["end"] - p["start"]
            s_dur = s["end"] - s["start"]
            if gap <= max_gap and (p_dur < min_dur or s_dur < min_dur * 0.6) and (s["end"] - p["start"]) <= max_len:
                p["end"] = s["end"]
                p["text"] = (p["text"] + s["text"]).strip()
                continue
        out.append(dict(s))
    return out


def transcribe_audio(audio_path: str, model_size: str, beam_size: int = 5, progress_cb=None):
    model = load_whisper_model(model_size)
    audio_array = load_wav_as_float32(audio_path)
    segments, info = model.transcribe(
        audio_array,
        language="zh",
        task="transcribe",
        beam_size=beam_size,
        best_of=5,
        temperature=[0.0, 0.2, 0.4],
        vad_filter=True,
        vad_parameters=dict(threshold=0.35, min_silence_duration_ms=400,
                            speech_pad_ms=250, min_speech_duration_ms=200),
        condition_on_previous_text=False,       # tránh lặp câu / trôi lệch kéo dài
        initial_prompt="以下是普通话的句子，使用简体中文。",   # gợi ý tiếng Trung giản thể, có dấu câu
        no_speech_threshold=0.5,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )

    results = []
    prev_text = None
    for seg in segments:
        if _is_hallucination(seg):
            continue
        text = seg.text.strip()
        if text == prev_text and (getattr(seg, "avg_logprob", 0.0) or 0.0) < -0.6:
            continue   # lặp lại y hệt câu trước với độ tin cậy thấp
        prev_text = text
        results.append({"start": seg.start, "end": seg.end, "text": text})
        if progress_cb:
            progress_cb(seg.end, text)

    return merge_short_segments(results)


# ============================================================
# DỊCH TRUNG → VIỆT (nhiều nguồn dự phòng, có kiểm tra kết quả)
# ============================================================

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _is_good_translation(src: str, out: str) -> bool:
    """Kết quả dịch hợp lệ = không rỗng và thật sự không còn là chữ Trung."""
    if not out or not out.strip():
        return False
    if not _CJK_RE.search(src):
        return True
    if out.strip() == src.strip():
        return False
    cjk = len(_CJK_RE.findall(out))
    return cjk / max(len(out.strip()), 1) < 0.3


def _tr_google_web(text: str) -> str:
    q = urllib.parse.urlencode({"client": "gtx", "sl": "zh-CN", "tl": "vi", "dt": "t", "q": text})
    req = urllib.request.Request(
        "https://translate.googleapis.com/translate_a/single?" + q,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode("utf-8"))
    return "".join(p[0] for p in data[0] if p and p[0])


def _tr_deep_google(text: str) -> str:
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source="zh-CN", target="vi").translate(text)


def _tr_mymemory(text: str) -> str:
    q = urllib.parse.urlencode({"q": text[:480], "langpair": "zh-CN|vi"})
    req = urllib.request.Request("https://api.mymemory.translated.net/get?" + q,
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode("utf-8"))
    out = (data.get("responseData") or {}).get("translatedText", "")
    if not out or "MYMEMORY WARNING" in out.upper() or "INVALID" in out.upper():
        raise RuntimeError(out or "MyMemory không trả kết quả")
    return out


_TRANSLATE_BACKENDS = (_tr_google_web, _tr_deep_google, _tr_mymemory)


def _translate_one(text: str, errors: list):
    """Dịch 1 câu, thử lần lượt các nguồn. Trả về None nếu tất cả đều thất bại."""
    if not text.strip():
        return text
    for fn in _TRANSLATE_BACKENDS:
        for attempt in range(2):
            try:
                out = html_lib.unescape((fn(text) or "").strip())
                if _is_good_translation(text, out):
                    return out
                errors.append(f"{fn.__name__}: kết quả không phải tiếng Việt")
                break
            except Exception as e:
                errors.append(f"{fn.__name__}: {type(e).__name__}: {e}")
                time.sleep(0.5 * (attempt + 1))
    return None


def translate_segments(segments, progress_cb=None, workers: int = 8):
    """Dịch toàn bộ câu sang tiếng Việt (song song). Câu nào dịch hỏng sẽ có translated_ok=False.
    Nếu hỏng toàn bộ → raise lỗi có nguyên nhân cụ thể (không âm thầm giữ chữ Trung nữa)."""
    n = len(segments)
    errors = []
    results = [None] * n
    done = 0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(_translate_one, seg["text"], errors): i for i, seg in enumerate(segments)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")
                results[i] = None
            done += 1
            if progress_cb:
                progress_cb(done, n)

    ok_count = 0
    for seg, tr in zip(segments, results):
        if tr:
            seg["translated"] = tr
            seg["translated_ok"] = True
            ok_count += 1
        else:
            seg["translated"] = seg["text"]
            seg["translated_ok"] = False

    if n and ok_count == 0:
        last = errors[-1] if errors else "không rõ nguyên nhân"
        raise RuntimeError(
            "Không dịch được câu nào sang tiếng Việt — thường do máy không có Internet hoặc bị chặn "
            "truy cập dịch vụ dịch. Hãy kiểm tra kết nối mạng rồi thử lại.\n"
            f"Chi tiết lỗi cuối: {last}"
        )
    return segments


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt(segments, bilingual: bool = False) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
        if bilingual:
            lines.append(seg["text"])
        lines.append(seg["translated"])
        lines.append("")
    return "\n".join(lines)


def hex_to_ass_color(hex_color: str, alpha: str = "00") -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        hex_color = "FFFFFF"
    r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
    return f"&H{alpha}{b}{g}{r}&".upper()


def hex_to_ffmpeg_color(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        hex_color = "000000"
    return f"0x{hex_color}"


def auto_outline_color(text_color_hex: str) -> str:
    """Tự chọn màu viền (đen/trắng) tương phản với màu chữ."""
    hex_color = text_color_hex.lstrip("#")
    if len(hex_color) != 6:
        return "#000000"
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000000" if luminance > 0.55 else "#FFFFFF"


def compute_free_box(width: int, height: int, x_pct: float, y_pct: float, w_pct: float, h_pct: float):
    x = int(width * x_pct / 100)
    y = int(height * y_pct / 100)
    w = int(width * w_pct / 100)
    h = int(height * h_pct / 100)
    x = max(0, min(x, width - 2))
    y = max(0, min(y, height - 2))
    w = max(4, min(w, width - x))
    h = max(4, min(h, height - y))
    return x, y, w, h


def compute_logo_box(width: int, height: int, corner: str):
    bw, bh = int(width * 0.22), int(height * 0.13)
    cx = 0 if corner[1] == "l" else width - bw
    cy = 0 if corner[0] == "t" else height - bh
    return cx, cy, bw, bh


def process_video(video_path: str, output_path: str, *,
                   remove_old_sub: bool = False, old_sub_method: str = "blur",
                   old_sub_box=None, old_sub_color: str = "0x000000", blur_strength: int = 25,
                   burn_new_sub: bool = False, srt_path: str = None,
                   font_size: int = 20, primary_color: str = "&H00FFFFFF&",
                   outline_color: str = "&H00000000&", alignment: int = 2, margin_v: int = 25,
                   video_preset: str = "veryfast", hflip: bool = False, frame_width: int = None):
    """Gộp lật gương + xóa phụ đề gốc + ghi phụ đề mới vào MỘT lần encode video duy nhất."""
    filters = []
    stage = "[0:v]"
    counter = 0

    if hflip:
        # Lật gương trước; toạ độ vùng che phụ đề gốc được lật theo để vẫn trùng đúng chỗ,
        # còn phụ đề mới ghi SAU bước này nên không bị ngược chữ.
        filters.append(f"{stage}hflip[vflip{counter}]")
        stage = f"[vflip{counter}]"
        if old_sub_box and frame_width:
            bx, by, bw, bh = old_sub_box
            old_sub_box = (frame_width - bx - bw, by, bw, bh)

    if remove_old_sub and old_sub_box:
        x, y, w, h = old_sub_box
        if old_sub_method == "blur":
            luma_radius = max(1, min(int(blur_strength), 40))
            chroma_radius = max(1, min(luma_radius, 17))
            filters.append(f"{stage}split=2[vmain{counter}][vcrop{counter}]")
            filters.append(
                f"[vcrop{counter}]crop={w}:{h}:{x}:{y},"
                f"boxblur=luma_radius={luma_radius}:luma_power=2:"
                f"chroma_radius={chroma_radius}:chroma_power=2[vblur{counter}]"
            )
            filters.append(f"[vmain{counter}][vblur{counter}]overlay={x}:{y}[v{counter}]")
        elif old_sub_method == "delogo":
            filters.append(f"{stage}delogo=x={x}:y={y}:w={w}:h={h}:show=0[v{counter}]")
        else:  # solid
            filters.append(f"{stage}drawbox=x={x}:y={y}:w={w}:h={h}:color={old_sub_color}@1:t=fill[v{counter}]")
        stage = f"[v{counter}]"
        counter += 1

    if burn_new_sub and srt_path:
        srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
        style = (
            f"FontName=Arial,FontSize={font_size},PrimaryColour={primary_color},"
            f"OutlineColour={outline_color},BorderStyle=1,Outline=1.2,Shadow=0.5,"
            f"Alignment={alignment},MarginV={margin_v}"
        )
        filters.append(f"{stage}subtitles={srt_escaped}:force_style='{style}'[vout]")
        stage = "[vout]"

    # libx264 yêu cầu chiều rộng/cao là số chẵn.
    filters.append(f"{stage}scale=trunc(iw/2)*2:trunc(ih/2)*2[veven]")
    stage = "[veven]"

    filter_complex = ";".join(filters)

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-filter_complex", filter_complex,
        "-map", stage, "-map", "0:a?",
        "-c:v", "libx264", "-preset", video_preset, "-crf", "20",
        "-threads", "0",
        "-c:a", "copy",
        output_path,
    ]
    _run_ffmpeg(cmd, label="xử lý hình ảnh video (lật gương / xoá phụ đề gốc / ghi phụ đề mới)")


def adjust_video_volume(video_path: str, output_path: str, volume_factor: float):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-filter:a", f"volume={volume_factor}",
        "-c:v", "copy",
        output_path,
    ]
    _run_ffmpeg(cmd, label="điều chỉnh âm lượng gốc")


# ============================================================
# VIRAL BOOST — lượt xử lý hoàn thiện cuối (1 lần encode)
# ============================================================

def _ff_escape_path(p: str) -> str:
    return p.replace("\\", "/").replace(":", "\\:")


def finish_video(in_path: str, out_path: str, tmp_dir: str, *, speed: float = 1.0, zoom_pct: int = 0,
                  color_boost: bool = False, logo_corner=None, vertical: bool = False,
                  hook_text: str = "", watermark_text: str = "", progress_bar: bool = False,
                  fade: bool = False, loudnorm: bool = False, video_preset: str = "veryfast"):
    """Áp các hiệu ứng Viral Boost lên video cuối cùng trong MỘT lần encode.
    Trả về danh sách ghi chú (vd: thiếu font nên bỏ qua chữ chèn)."""
    notes = []
    res = get_video_resolution(in_path)
    if res is None:
        raise FFmpegError("Không đọc được độ phân giải video để áp hiệu ứng Viral Boost.")
    W, H = res
    D_in = ffprobe_duration(in_path)
    D = D_in / speed
    font = find_font_path()
    audio_ok = has_audio_stream(in_path)

    chain, extra_inputs = [], []
    state = {"stage": "[0:v]", "n": 0}

    def nxt():
        state["n"] += 1
        return state["n"]

    def add(flt):
        k = nxt()
        label = f"[f{k}]"
        chain.append(f"{state['stage']}{flt}{label}")
        state["stage"] = label

    # 1) Che logo/watermark gốc ở góc
    if logo_corner:
        cx, cy, bw, bh = compute_logo_box(W, H, logo_corner)
        k = nxt()
        src = state["stage"]
        chain.append(f"{src}split=2[la{k}][lb{k}]")
        chain.append(
            f"[lb{k}]crop={bw}:{bh}:{cx}:{cy},"
            f"boxblur=luma_radius=20:luma_power=2:chroma_radius=10:chroma_power=2[lc{k}]"
        )
        chain.append(f"[la{k}][lc{k}]overlay={cx}:{cy}[f{k}]")
        state["stage"] = f"[f{k}]"

    # 2) Zoom nhẹ (cắt giữa rồi phóng lại về đúng kích thước)
    if zoom_pct > 0:
        z = 1 + zoom_pct / 100
        add(f"crop=trunc(iw/{z:.4f}/2)*2:trunc(ih/{z:.4f}/2)*2,scale={W}:{H}")

    # 3) Tăng màu & độ nét
    if color_boost:
        add("eq=contrast=1.06:saturation=1.18:brightness=0.02,unsharp=5:5:0.7:5:5:0.0")

    # 4) Tăng tốc nhẹ
    if abs(speed - 1.0) > 0.001:
        add(f"setpts=PTS/{speed:.4f}")

    OW, OH = W, H
    # 5) Khung dọc 9:16 (Shorts/Reels/TikTok) với nền mờ
    if vertical:
        k = nxt()
        src = state["stage"]
        chain.append(f"{src}split=2[bs{k}][fs{k}]")
        chain.append(
            f"[bs{k}]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
            f"boxblur=luma_radius=25:luma_power=2:chroma_radius=12:chroma_power=2[bg{k}]"
        )
        chain.append(f"[fs{k}]scale=1080:-2[fg{k}]")
        chain.append(f"[bg{k}][fg{k}]overlay=(W-w)/2:(H-h)/2[f{k}]")
        state["stage"] = f"[f{k}]"
        OW, OH = 1080, 1920

    # 6) Chữ hook 3-4 giây đầu + watermark tên kênh
    if (hook_text.strip() or watermark_text.strip()) and not font:
        notes.append("Không tìm thấy font trên máy nên bỏ qua chữ hook/watermark "
                     "(cài font DejaVu: sudo apt install fonts-dejavu-core).")
    elif font:
        ff = _ff_escape_path(font)
        if hook_text.strip():
            wrapped = "\n".join(textwrap.wrap(hook_text.strip(), width=18 if vertical else 28))
            hp = os.path.join(tmp_dir, "hook.txt")
            with open(hp, "w", encoding="utf-8") as f:
                f.write(wrapped)
            fs = max(int(OH / (22 if vertical else 18)), 18)
            add(
                f"drawtext=fontfile='{ff}':textfile='{_ff_escape_path(hp)}':expansion=none:"
                f"fontcolor=0xFFE600:fontsize={fs}:borderw={max(fs // 12, 2)}:bordercolor=black:"
                f"line_spacing=8:x=(w-text_w)/2:y=h*0.10:enable='between(t,0,{min(4.0, D):.1f})'"
            )
        if watermark_text.strip():
            wp = os.path.join(tmp_dir, "watermark.txt")
            with open(wp, "w", encoding="utf-8") as f:
                f.write(watermark_text.strip())
            fs2 = max(int(OH / 40), 14)
            pad = max(int(OW * 0.03), 10)
            add(
                f"drawtext=fontfile='{ff}':textfile='{_ff_escape_path(wp)}':expansion=none:"
                f"fontcolor=white@0.7:fontsize={fs2}:borderw=2:bordercolor=black@0.6:x=w-text_w-{pad}:y={pad}"
            )

    # 7) Thanh tiến trình chạy dưới đáy
    if progress_bar:
        bar_h = max(int(OH / 150), 4)
        extra_inputs += ["-f", "lavfi", "-i", f"color=c=0xFF3B30:s={OW}x{bar_h}:r=30:d={D + 1:.2f}"]
        k = nxt()
        chain.append(f"{state['stage']}[1:v]overlay=x='-w+w*t/{D:.3f}':y=H-h:shortest=1[f{k}]")
        state["stage"] = f"[f{k}]"

    # 8) Fade vào/ra
    if fade and D > 2:
        add(f"fade=t=in:st=0:d=0.6,fade=t=out:st={D - 0.6:.3f}:d=0.6")

    # Âm thanh: tăng tốc, chuẩn hoá âm lượng, fade
    a_parts = []
    if audio_ok:
        if abs(speed - 1.0) > 0.001:
            a_parts.append(f"atempo={speed:.4f}")
        if loudnorm:
            a_parts.append("loudnorm=I=-14:TP=-1.5:LRA=11")
        if fade and D > 2:
            a_parts.append(f"afade=t=in:st=0:d=0.6,afade=t=out:st={D - 0.6:.3f}:d=0.6")
        if a_parts:
            chain.append(f"[0:a]{','.join(a_parts)}[aout]")

    vid_filtered = state["stage"] != "[0:v]"
    cmd = ["ffmpeg", "-y", "-i", in_path] + extra_inputs
    if chain:
        cmd += ["-filter_complex", ";".join(chain)]
    cmd += ["-map", state["stage"] if vid_filtered else "0:v"]
    if audio_ok:
        cmd += ["-map", "[aout]" if a_parts else "0:a?"]
    if vid_filtered:
        cmd += ["-c:v", "libx264", "-preset", video_preset, "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "0"]
    else:
        cmd += ["-c:v", "copy"]
    if audio_ok:
        cmd += ["-c:a", "aac", "-b:a", "192k"] if a_parts else ["-c:a", "copy"]
    cmd += ["-movflags", "+faststart", out_path]
    _run_ffmpeg(cmd, label="Viral Boost (hiệu ứng hoàn thiện video)")
    return notes


# ============================================================
# THUMBNAIL + GÓI YOUTUBE
# ============================================================

def make_thumbnail(video_path: str, title: str, out_path: str, vertical: bool = False):
    """Tạo thumbnail bắt mắt: khung hình đẹp + tăng màu + chữ vàng viền đen. Trả về True nếu thành công."""
    try:
        dur = ffprobe_duration(video_path)
    except Exception:
        dur = 0
    frame_path = out_path + ".frame.jpg"
    ok = False
    for ts in (dur * 0.25, dur * 0.5, 0):
        ok, _ = _try_grab_frame(
            ["ffmpeg", "-y", "-ss", f"{max(ts, 0):.2f}", "-i", video_path, "-frames:v", "1", "-q:v", "2", frame_path],
            frame_path,
        )
        if ok:
            break
    if not ok:
        return False

    target = (1080, 1920) if vertical else (1280, 720)
    img = Image.open(frame_path).convert("RGB")
    img = ImageOps.fit(img, target, Image.LANCZOS)
    img = ImageEnhance.Color(img).enhance(1.25)
    img = ImageEnhance.Contrast(img).enhance(1.08)

    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for y in range(h):
        alpha = int(190 * max(0.0, (y / h - 0.40) / 0.60))
        if alpha:
            od.line([(0, y), (w, y)], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    text = " ".join(title.upper().split())
    if text:
        size = int(min(w, h) / (7 if not vertical else 9))
        font = _load_preview_font(size)
        lines = textwrap.wrap(text, width=16 if not vertical else 12)[:3]
        draw = ImageDraw.Draw(img)
        line_h = int(size * 1.18)
        y = h - int(h * 0.06) - line_h * len(lines)
        x = int(w * 0.05)
        for line in lines:
            draw.text((x, y), line, font=font, fill="#FFE600", stroke_width=max(size // 8, 4), stroke_fill="black")
            y += line_h

    img.save(out_path, "JPEG", quality=92)
    try:
        os.remove(frame_path)
    except OSError:
        pass
    return True


def _fmt_clock(sec: float) -> str:
    sec = int(max(sec, 0))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _shorten(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def build_youtube_pack(segments, total_duration: float, speed: float = 1.0, channel_name: str = "",
                        vertical: bool = False, custom_title: str = "") -> str:
    """Gợi ý tiêu đề, mô tả, mốc chương và hashtag dựa trên nội dung đã dịch (không cần gọi API)."""
    first = _shorten(segments[0]["translated"], 55) if segments else "Video hay"
    base = custom_title.strip() or first
    titles = [
        f"[Vietsub] {base}",
        f"😱 {base} — Xem đến cuối!",
        f"Không thể tin nổi! {base}",
        f"{base} | Lồng tiếng Việt",
        f"🔥 {base} (Cực hot)",
    ]
    titles = [_shorten(t, 100) for t in titles]

    summary = _shorten(" ".join(s["translated"].strip() for s in segments[:3]), 240)

    out_dur = total_duration / speed
    chapters = [("00:00", "Mở đầu")]
    last = 0.0
    gap = max(15.0, out_dur / 8)
    for seg in segments:
        t = seg["start"] / speed
        if t - last >= gap and len(chapters) < 8:
            chapters.append((_fmt_clock(t), _shorten(seg["translated"], 45)))
            last = t
    if len(chapters) < 3:
        chapters = []

    tags = ["#vietsub", "#longtieng", "#viral", "#xuhuong"]
    if vertical:
        tags.append("#shorts")
    chan_tag = re.sub(r"\W", "", channel_name or "")
    if chan_tag:
        tags.append("#" + chan_tag)

    lines = ["=== GỢI Ý TIÊU ĐỀ (chọn 1, nhớ chỉnh cho đúng nội dung) ==="]
    lines += [f"{i}. {t}" for i, t in enumerate(titles, 1)]
    lines += ["", "=== MÔ TẢ ===", f"🎬 {base}", summary, ""]
    if chapters:
        lines.append("📌 Mốc thời gian:")
        lines += [f"{t} {label}" for t, label in chapters]
        lines.append("")
    chan_text = f" {channel_name}" if channel_name else ""
    lines += [
        f"👍 Thấy hay thì like, chia sẻ và đăng ký kênh{chan_text} để xem thêm video mới mỗi ngày!",
        "🌐 Video được dịch & lồng tiếng sang tiếng Việt.",
        "🔗 Nguồn video gốc: (điền tên tác giả / link gốc tại đây)",
        "",
        " ".join(tags),
    ]
    return "\n".join(lines)


# ============================================================
# TỰ ĐỘNG PHÁT HIỆN VÙNG PHỤ ĐỀ GỐC (Pillow + numpy)
# ============================================================

def _extract_sample_frames(video_path: str, out_dir: str, count: int = 10):
    try:
        dur = ffprobe_duration(video_path)
    except Exception:
        return []
    if dur <= 0.5:
        return []
    if count > 1:
        timestamps = [dur * (0.08 + 0.84 * i / (count - 1)) for i in range(count)]
    else:
        timestamps = [dur * 0.5]

    paths = []
    for i, t in enumerate(timestamps):
        p = os.path.join(out_dir, f"det_frame_{i}.jpg")
        try:
            cmd = ["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", video_path, "-frames:v", "1", "-q:v", "3", p]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(p):
                paths.append(p)
        except Exception:
            pass
    return paths


def _edge_energy_map(gray_arr: "np.ndarray") -> "np.ndarray":
    gx = np.abs(np.diff(gray_arr, axis=1))
    gy = np.abs(np.diff(gray_arr, axis=0))
    gx = np.pad(gx, ((0, 0), (0, 1)))
    gy = np.pad(gy, ((0, 1), (0, 0)))
    return gx + gy


def _default_subtitle_box(width: int, height: int):
    x = int(width * 0.05)
    y = int(height * 0.80)
    w = int(width * 0.90)
    h = int(height * 0.14)
    return x, y, w, h


def detect_subtitle_region(video_path: str, width: int, height: int, tmp_dir: str, sample_count: int = 10):
    """Trả về (x, y, w, h) theo pixel là vùng nhiều khả năng chứa phụ đề gốc ghi cứng."""
    try:
        frame_paths = _extract_sample_frames(video_path, tmp_dir, count=sample_count)
        if len(frame_paths) < 3:
            return _default_subtitle_box(width, height)

        row_profiles = []
        edge_maps = []
        for p in frame_paths:
            img = Image.open(p).convert("L")
            if img.size != (width, height):
                img = img.resize((width, height))
            arr = np.asarray(img, dtype=np.float32)
            edge = _edge_energy_map(arr)
            edge_maps.append(edge)
            row_energy = edge.sum(axis=1)
            m = row_energy.max()
            row_profiles.append(row_energy / m if m > 0 else row_energy)

        avg_row = np.mean(row_profiles, axis=0)

        bottom_start = int(height * 0.45)
        top_end = int(height * 0.22)
        bottom_zone = avg_row[bottom_start:]
        top_zone = avg_row[:top_end]

        bottom_peak = float(bottom_zone.max()) if len(bottom_zone) else 0.0
        top_peak = float(top_zone.max()) if len(top_zone) else 0.0

        if bottom_peak <= 0.05 and top_peak <= 0.05:
            return _default_subtitle_box(width, height)

        if bottom_peak >= top_peak:
            zone_offset, zone, peak = bottom_start, bottom_zone, bottom_peak
        else:
            zone_offset, zone, peak = 0, top_zone, top_peak

        peak_idx = int(np.argmax(zone)) + zone_offset
        threshold = peak * 0.4

        y_start = peak_idx
        while y_start > 0 and avg_row[y_start - 1] >= threshold:
            y_start -= 1
        y_end = peak_idx
        while y_end < height - 1 and avg_row[y_end + 1] >= threshold:
            y_end += 1

        min_h, max_h = int(height * 0.04), int(height * 0.22)
        if (y_end - y_start) < min_h:
            pad = (min_h - (y_end - y_start)) // 2 + 1
            y_start, y_end = max(0, y_start - pad), min(height - 1, y_end + pad)
        if (y_end - y_start) > max_h:
            center = (y_start + y_end) // 2
            y_start, y_end = max(0, center - max_h // 2), min(height - 1, center + max_h // 2)

        col_profiles = []
        for edge in edge_maps:
            band = edge[y_start:y_end + 1, :]
            col_energy = band.sum(axis=0)
            m = col_energy.max()
            col_profiles.append(col_energy / m if m > 0 else col_energy)
        avg_col = np.mean(col_profiles, axis=0)
        col_peak = float(avg_col.max())
        col_threshold = col_peak * 0.25

        idx = np.where(avg_col >= col_threshold)[0]
        if len(idx) == 0:
            x_start, x_end = int(width * 0.05), int(width * 0.95)
        else:
            x_start, x_end = int(idx.min()), int(idx.max())

        pad_x, pad_y = int(width * 0.02), int(height * 0.015)
        x_start, x_end = max(0, x_start - pad_x), min(width - 1, x_end + pad_x)
        y_start, y_end = max(0, y_start - pad_y), min(height - 1, y_end + pad_y)

        x, y, w, h = x_start, y_start, (x_end - x_start), (y_end - y_start)
        if w < width * 0.15 or h < height * 0.03:
            return _default_subtitle_box(width, height)
        return (x, y, w, h)
    except Exception:
        return _default_subtitle_box(width, height)


# ============================================================
# XEM TRƯỚC (PREVIEW)
# ============================================================

def ensure_workdir() -> str:
    if "workdir" not in st.session_state:
        st.session_state.workdir = tempfile.mkdtemp(prefix="vidtrans_")
    return st.session_state.workdir


def get_persistent_video_path(uploaded_file) -> str:
    """Lưu file upload vào thư mục tạm bền vững (qua các lần rerun). Hash chỉ tính 1 lần cho mỗi file."""
    ensure_workdir()

    file_key = getattr(uploaded_file, "file_id", None) or f"{uploaded_file.name}-{uploaded_file.size}"
    cache = st.session_state.setdefault("_hash_cache", {})
    if file_key not in cache:
        cache[file_key] = hashlib.md5(uploaded_file.getbuffer()).hexdigest()[:10]
    file_hash = cache[file_key]

    ext = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    path = os.path.join(st.session_state.workdir, f"src_{file_hash}{ext}")

    if not os.path.exists(path):
        for old in os.listdir(st.session_state.workdir):
            if old.startswith("src_") and old != os.path.basename(path):
                try:
                    os.remove(os.path.join(st.session_state.workdir, old))
                except OSError:
                    pass
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return path


def _try_grab_frame(cmd, out_path: str):
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except OSError:
            pass
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    ok = result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0
    stderr_tail = "\n".join(result.stderr.strip().splitlines()[-10:]) if result.stderr else ""
    return ok, stderr_tail


def extract_preview_frame(video_path: str, out_path: str):
    """Trích 1 khung hình xem trước, thử lần lượt nhiều cách để tối đa khả năng thành công."""
    try:
        dur = ffprobe_duration(video_path)
    except Exception:
        dur = 0

    timestamps = []
    if dur > 0.2:
        timestamps.append(max(min(dur * 0.3, dur - 0.1), 0))
    if 0 not in timestamps:
        timestamps.append(0)

    attempts = []
    for ts in timestamps:
        attempts.append((f"seek nhanh @ {ts:.1f}s",
                          ["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", video_path,
                           "-frames:v", "1", "-q:v", "2", out_path]))
        attempts.append((f"seek chính xác @ {ts:.1f}s",
                          ["ffmpeg", "-y", "-i", video_path, "-ss", f"{ts:.2f}",
                           "-frames:v", "1", "-q:v", "2", out_path]))

    attempts.append(("select khung đầu tiên",
                      ["ffmpeg", "-y", "-i", video_path, "-vf", "select=eq(n\\,0)",
                       "-frames:v", "1", "-q:v", "2", "-vsync", "0", out_path]))
    attempts.append(("map track video 0",
                      ["ffmpeg", "-y", "-i", video_path, "-map", "0:v:0",
                       "-frames:v", "1", "-q:v", "2", "-vsync", "0", out_path]))

    last_error = "(không có thông tin lỗi)"
    for label, cmd in attempts:
        ok, stderr_tail = _try_grab_frame(cmd, out_path)
        if ok:
            return True
        if stderr_tail:
            last_error = f"[{label}]\n{stderr_tail}"

    st.session_state["_last_preview_frame_error"] = last_error
    return False


def ensure_preview_frame(video_path: str):
    if st.session_state.get("preview_frame_for") == video_path and st.session_state.get("preview_frame_path"):
        return st.session_state.preview_frame_path, st.session_state.preview_video_res

    frame_path = os.path.join(st.session_state.workdir, "preview_frame.jpg")
    st.session_state["preview_frame_error"] = None
    try:
        ok = extract_preview_frame(video_path, frame_path)
        if not ok:
            raise RuntimeError(st.session_state.get("_last_preview_frame_error", "Không rõ nguyên nhân."))

        resolution = get_video_resolution(video_path)
        if resolution is None:
            with Image.open(frame_path) as im:
                resolution = im.size

        st.session_state.preview_frame_for = video_path
        st.session_state.preview_frame_path = frame_path
        st.session_state.preview_video_res = resolution
        return frame_path, resolution
    except Exception as e:
        st.session_state.preview_frame_for = video_path
        st.session_state.preview_frame_path = None
        st.session_state.preview_video_res = None
        st.session_state["preview_frame_error"] = str(e)
        return None, None


def ensure_detected_box(video_path: str, width: int, height: int):
    """Chạy phát hiện vùng phụ đề gốc 1 lần cho mỗi video."""
    if st.session_state.get("detected_box_for") == video_path and st.session_state.get("detected_box"):
        return st.session_state.detected_box
    box = detect_subtitle_region(video_path, width, height, st.session_state.workdir)
    st.session_state.detected_box_for = video_path
    st.session_state.detected_box = box
    return box


def _load_preview_font(size: int):
    p = find_font_path()
    if p:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def render_subtitle_style_preview(base_image: "Image.Image", sample_text: str, font_size: int,
                                   text_color_hex: str, outline_color_hex: str,
                                   alignment: int, margin_v: int) -> "Image.Image":
    img = base_image.copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)

    scaled_font_size = max(int(font_size * (h / 360)), 10)
    font = _load_preview_font(scaled_font_size)

    bbox = draw.textbbox((0, 0), sample_text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x = (w - text_w) / 2
    scaled_margin = int(margin_v * (h / 360))
    if alignment == 8:
        y = scaled_margin
    elif alignment == 5:
        y = (h - text_h) / 2
    else:
        y = h - text_h - scaled_margin - h * 0.02

    outline_w = max(scaled_font_size // 12, 1)
    draw.text((x, y), sample_text, font=font, fill=text_color_hex,
              stroke_width=outline_w, stroke_fill=outline_color_hex)
    return img


def render_old_sub_box_preview(base_image: "Image.Image", box, method: str,
                                color_hex: str, blur_strength: int) -> "Image.Image":
    img = base_image.copy()
    if not box:
        return img
    x, y, w, h = box
    x2, y2 = x + w, y + h

    if method == "blur":
        region = img.crop((x, y, x2, y2)).filter(ImageFilter.GaussianBlur(max(blur_strength / 4, 1)))
        img.paste(region, (x, y))
    elif method == "delogo":
        region = img.crop((x, y, x2, y2)).filter(ImageFilter.GaussianBlur(max(blur_strength / 6, 1)))
        img.paste(region, (x, y))
    else:  # solid
        overlay = Image.new("RGB", (w, h), color_hex)
        img.paste(overlay, (x, y))

    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x2 - 1, y2 - 1], outline="#FF3B30", width=max(int(h * 0.06), 2))
    return img


def render_zoom_preview(img: "Image.Image", zoom_pct: int) -> "Image.Image":
    z = 1 + zoom_pct / 100
    w, h = img.size
    cw, ch = int(w / z), int(h / z)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)


def render_text_overlays_preview(img: "Image.Image", hook_text: str, watermark_text: str) -> "Image.Image":
    img = img.copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)
    if hook_text.strip():
        fs = max(int(h / 18), 14)
        font = _load_preview_font(fs)
        y = int(h * 0.10)
        for line in textwrap.wrap(hook_text.strip(), width=28):
            bb = draw.textbbox((0, 0), line, font=font)
            draw.text(((w - (bb[2] - bb[0])) / 2, y), line, font=font, fill="#FFE600",
                      stroke_width=max(fs // 12, 2), stroke_fill="black")
            y += int(fs * 1.25)
    if watermark_text.strip():
        fs2 = max(int(h / 40), 12)
        font2 = _load_preview_font(fs2)
        bb = draw.textbbox((0, 0), watermark_text.strip(), font=font2)
        pad = max(int(w * 0.03), 8)
        draw.text((w - (bb[2] - bb[0]) - pad, pad), watermark_text.strip(), font=font2, fill="#E8E8E8",
                  stroke_width=1, stroke_fill="black")
    return img


def generate_tts_preview_audio(voice: str, rate: str, volume: str, pitch: str, out_path: str,
                                sample_text: str = SAMPLE_TTS_TEXT, engine: str = "edge"):
    if engine == "vieneu":
        vieneu_engine = load_vieneu_engine()
        audio = _vieneu_infer(vieneu_engine, sample_text, voice)
        vieneu_engine.save(audio, out_path)
        return
    import edge_tts
    communicate = edge_tts.Communicate(sample_text, voice=voice, rate=rate, volume=volume, pitch=pitch)
    _run_async(communicate.save(out_path))


# ============================================================
# LỒNG TIẾNG (TTS)
# ============================================================

def _run_async(coro):
    """Chạy coroutine trong một thread riêng với event loop mới (ổn định trong Streamlit,
    không bao giờ đụng loop đang chạy). Thread được gắn ngữ cảnh Streamlit để progress bar cập nhật được."""
    box = {}

    def runner():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            box["value"] = loop.run_until_complete(coro)
        except BaseException as e:  # noqa: BLE001
            box["error"] = e
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    t = threading.Thread(target=runner, daemon=True)
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx is not None:
            add_script_run_ctx(t, ctx)
    except Exception:
        pass
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


# ---------- VieNeu-TTS (github.com/pnnbao97/VieNeu-TTS) ----------

_VIENEU_LOCK = threading.Lock()   # engine ONNX dùng chung → chỉ cho 1 luồng đọc tại một thời điểm


@st.cache_resource(show_spinner=False)
def load_vieneu_engine():
    """Khởi tạo VieNeu-TTS một lần (mặc định v3 Turbo 48kHz, chạy CPU/ONNX; lần đầu sẽ tự tải model)."""
    from vieneu import Vieneu
    return Vieneu()


def _parse_vieneu_voices(raw):
    """Chuẩn hoá kết quả list_preset_voices() thành [(nhãn hiển thị, id giọng)]."""
    voices = []
    for v in raw or []:
        if isinstance(v, (tuple, list)) and v:
            label = str(v[0])
            voice_id = str(v[1]) if len(v) > 1 and v[1] else label
        elif isinstance(v, dict):
            label = str(v.get("label") or v.get("name") or v.get("id") or "")
            voice_id = str(v.get("id") or v.get("voice_id") or label)
        else:
            label = voice_id = str(v)
        if label:
            voices.append((label, voice_id))
    return voices


def _vieneu_infer(engine, text: str, voice_id: str):
    """Đọc 1 câu bằng giọng preset. Nếu truyền tên không được thì thử lại bằng get_preset_voice()."""
    with _VIENEU_LOCK:
        try:
            return engine.infer(text, voice=voice_id)
        except Exception as first_err:  # noqa: BLE001
            try:
                return engine.infer(text, voice=engine.get_preset_voice(voice_id))
            except Exception:  # noqa: BLE001
                raise first_err


def synthesize_all_vieneu(items, voice_id: str, progress_cb=None):
    """Tạo giọng đọc bằng VieNeu-TTS (lần lượt từng câu, ghi file .wav).
    Trả về (tập chỉ số câu lỗi, thông báo lỗi cuối cùng)."""
    all_idx = {idx for idx, _, _ in items}
    try:
        engine = load_vieneu_engine()
    except ImportError:
        return all_idx, "Chưa cài thư viện vieneu (pip install -U vieneu)."
    except Exception as e:  # noqa: BLE001
        return all_idx, f"Không khởi tạo được VieNeu-TTS: {type(e).__name__}: {e}"

    failed_indices = set()
    errors = []
    for n, (idx, text, raw_path) in enumerate(items, start=1):
        success = False
        for _attempt in range(2):
            try:
                audio = _vieneu_infer(engine, text, voice_id)
                engine.save(audio, raw_path)
                if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
                    success = True
                    break
                errors.append("VieNeu trả về file rỗng")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}")
        if not success:
            failed_indices.add(idx)
        if progress_cb:
            try:
                progress_cb(n, len(items))
            except Exception:
                pass
    return failed_indices, (errors[-1] if errors else "")


def synthesize_all_tts(items, voice: str, rate: str, volume: str, pitch: str,
                        max_retries: int = 3, concurrency: int = 6, progress_cb=None):
    """Tạo giọng đọc bằng edge-tts. Trả về (tập chỉ số câu lỗi, thông báo lỗi cuối cùng)."""
    all_idx = {idx for idx, _, _ in items}
    try:
        import edge_tts
    except ImportError:
        return all_idx, "Chưa cài thư viện edge-tts (pip install -U edge-tts)."

    failed_indices = set()
    errors = []
    done_count = 0

    async def _synthesize_one(sem, idx, text, raw_path):
        nonlocal done_count
        async with sem:
            success = False
            for attempt in range(max_retries):
                try:
                    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, volume=volume, pitch=pitch)
                    await communicate.save(raw_path)
                    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
                        success = True
                        break
                    errors.append("edge-tts trả về file rỗng")
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{type(e).__name__}: {e}")
                await asyncio.sleep(0.8 * (attempt + 1))
            if not success:
                failed_indices.add(idx)
        done_count += 1
        if progress_cb:
            try:
                progress_cb(done_count, len(items))
            except Exception:
                pass

    async def _synthesize_all():
        sem = asyncio.Semaphore(concurrency)
        await asyncio.gather(*[_synthesize_one(sem, idx, text, raw_path) for idx, text, raw_path in items])

    try:
        _run_async(_synthesize_all())
    except Exception as e:  # noqa: BLE001
        errors.append(f"{type(e).__name__}: {e}")
        failed_indices = set(all_idx)

    last_error = errors[-1] if errors else ""
    return failed_indices, last_error


def _gtts_fallback(items):
    """Dự phòng: dùng gTTS (Google Translate TTS, giọng Việt) cho các câu edge-tts không tạo được.
    Trả về (tập chỉ số vẫn lỗi, thông báo lỗi)."""
    try:
        from gtts import gTTS
    except ImportError:
        return {idx for idx, _, _ in items}, "Chưa cài gTTS (pip install gTTS) nên không có giọng dự phòng."

    errors = []

    def _one(item):
        idx, text, raw_path = item
        for attempt in range(2):
            try:
                gTTS(text=text, lang="vi").save(raw_path)
                if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
                    return idx, True
            except Exception as e:  # noqa: BLE001
                errors.append(f"gTTS {type(e).__name__}: {e}")
                time.sleep(0.8)
        return idx, False

    still_failed = set()
    with ThreadPoolExecutor(max_workers=4) as ex:
        for idx, ok in ex.map(_one, items):
            if not ok:
                still_failed.add(idx)
    return still_failed, (errors[-1] if errors else "")


def fit_audio_to_duration(in_path: str, out_path: str, target_duration: float):
    """Khớp audio TTS vào khung thời gian của câu, tăng tốc tối đa MAX_TTS_SPEEDUP lần
    để giọng đọc luôn tự nhiên; phần tràn được xử lý ở build_dub_track() (không chồng tiếng)."""
    dur = ffprobe_duration(in_path)
    base = ["ffmpeg", "-y", "-i", in_path]
    tail = ["-ar", str(TTS_SAMPLE_RATE), "-ac", "1", out_path]
    quiet = dict(check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if dur <= target_duration or target_duration <= 0:
        subprocess.run(base + tail, **quiet)
        return

    factor = min(dur / target_duration, MAX_TTS_SPEEDUP)
    if factor <= 1.02:
        subprocess.run(base + tail, **quiet)
        return

    subprocess.run(base + ["-filter:a", f"atempo={factor:.3f}"] + tail, **quiet)


def make_silence(out_path: str, duration: float):
    duration = max(duration, 0.1)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r={TTS_SAMPLE_RATE}:cl=mono:d={duration}", out_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _read_wav_mono_float(path: str) -> "np.ndarray":
    with wave.open(path, "rb") as wf:
        nch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        arr = arr.reshape(-1, nch).mean(axis=1)
    return arr


def _to_mp3_items(items, failed_indices, tmp_dir: str, raw_paths: dict):
    """Tạo danh sách (idx, text, đường dẫn .mp3) cho các câu cần đọc lại bằng giọng dự phòng
    (edge-tts / gTTS), đồng thời cập nhật raw_paths để bước ghép biết file nào dùng cho câu nào."""
    retry = []
    for idx, text, _ in items:
        if idx in failed_indices:
            p = os.path.join(tmp_dir, f"tts_raw_{idx}.mp3")
            raw_paths[idx] = p
            retry.append((idx, text, p))
    return retry


def build_dub_track(segments, voice: str, rate: str, volume: str, pitch: str,
                     tmp_dir: str, total_duration: float, concurrency: int = 6, progress_cb=None,
                     engine: str = "edge"):
    """Trả về (đường dẫn track lồng tiếng hoặc None nếu không tạo được câu nào, tập câu lỗi, thông báo lỗi).
    engine = "edge" (edge-tts) hoặc "vieneu" (VieNeu-TTS). Với VieNeu, câu nào hỏng sẽ tự chuyển sang
    edge-tts rồi gTTS; ghi chú về việc dùng giọng dự phòng được lưu ở st.session_state["_dub_notice"]."""
    ext = ".wav" if engine == "vieneu" else ".mp3"
    items = []
    for i, seg in enumerate(segments):
        text = seg["translated"].strip()
        # Câu chưa dịch được (vẫn là chữ Trung) thì không đưa cho giọng Việt đọc.
        if text and seg.get("translated_ok", True):
            raw_path = os.path.join(tmp_dir, f"tts_raw_{i}{ext}")
            items.append((i, text, raw_path))

    if not items:
        return None, set(), "Không có câu nào đã được dịch sang tiếng Việt để lồng tiếng."

    raw_paths = {idx: p for idx, _, p in items}
    notice = None

    if engine == "vieneu":
        failed_indices, last_error = synthesize_all_vieneu(items, voice, progress_cb=progress_cb)
        if failed_indices:
            # VieNeu hỏng (chưa cài / lỗi) → đọc lại các câu đó bằng edge-tts.
            retry_items = _to_mp3_items(items, failed_indices, tmp_dir, raw_paths)
            notice = (f"VieNeu-TTS không đọc được {len(failed_indices)}/{len(items)} câu "
                      f"nên các câu đó dùng giọng dự phòng ({EDGE_FALLBACK_VOICE}). "
                      f"Lý do: {last_error or 'không rõ'}")
            failed_indices, edge_err = synthesize_all_tts(retry_items, EDGE_FALLBACK_VOICE, "+0%", "+0%", "+0Hz",
                                                           concurrency=concurrency)
            if failed_indices and edge_err:
                last_error = f"{last_error} | {edge_err}" if last_error else edge_err
    else:
        failed_indices, last_error = synthesize_all_tts(items, voice, rate, volume, pitch,
                                                         concurrency=concurrency, progress_cb=progress_cb)

    # Vẫn còn câu hỏng → thử gTTS cho câu đó.
    if failed_indices:
        retry_items = _to_mp3_items(items, failed_indices, tmp_dir, raw_paths)
        failed_indices, gtts_err = _gtts_fallback(retry_items)
        if failed_indices and gtts_err:
            last_error = f"{last_error} | {gtts_err}" if last_error else gtts_err

    st.session_state["_dub_notice"] = notice

    if len(failed_indices) >= len(items):
        hint = (" Gợi ý: chạy 'pip install -U edge-tts gTTS vieneu' và kiểm tra kết nối Internet.")
        return None, failed_indices, (last_error or "Không rõ nguyên nhân") + hint

    # Chống chồng tiếng: câu sau không bao giờ bắt đầu sớm hơn thời điểm câu trước kết thúc ("cursor").
    placed = []   # (start_time, numpy audio)
    cursor = 0.0
    for idx, text, _unused_path in items:
        if idx in failed_indices:
            continue
        raw_path = raw_paths[idx]
        seg = segments[idx]
        slot_duration = max(seg["end"] - seg["start"], 0.3)
        fit_path = os.path.join(tmp_dir, f"tts_fit_{idx}.wav")
        try:
            fit_audio_to_duration(raw_path, fit_path, slot_duration)
            audio = _read_wav_mono_float(fit_path)
        except Exception:
            failed_indices.add(idx)
            continue
        actual_dur = len(audio) / TTS_SAMPLE_RATE
        actual_start = max(seg["start"], cursor)
        cursor = actual_start + actual_dur
        placed.append((actual_start, audio))

    dub_track_duration = max(total_duration, cursor)
    track = np.zeros(int(dub_track_duration * TTS_SAMPLE_RATE) + TTS_SAMPLE_RATE, dtype=np.float32)
    for start, audio in placed:
        s = int(start * TTS_SAMPLE_RATE)
        e = s + len(audio)
        if e > len(track):
            track = np.pad(track, (0, e - len(track)))
        track[s:e] += audio
    track = np.clip(track, -1.0, 1.0)

    dub_track_path = os.path.join(tmp_dir, "dub_track.wav")
    with wave.open(dub_track_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TTS_SAMPLE_RATE)
        wf.writeframes((track * 32767.0).astype(np.int16).tobytes())
    return dub_track_path, failed_indices, last_error


def combine_video_with_dub(video_path: str, dub_track_path: str, output_path: str,
                            keep_bg: bool, bg_volume: float = 0.08):
    if keep_bg and not has_audio_stream(video_path):
        keep_bg = False   # video gốc không có tiếng nền để giữ

    if keep_bg:
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-i", dub_track_path,
            "-filter_complex", f"[0:a]volume={bg_volume}[bg];[1:a]anull[dub];[bg][dub]amix=inputs=2:duration=first:normalize=0[aout]",
            "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-i", dub_track_path,
            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", output_path,
        ]
    _run_ffmpeg(cmd, label="ghép giọng lồng tiếng AI vào video")


# ============================================================
# FRAGMENT: nghe thử giọng (bấm nút chỉ rerun riêng khối này, không chạy lại cả app)
# ============================================================

@st.fragment
def tts_preview_fragment(label: str, voice: str, rate: str, volume: str, pitch: str, engine: str = "edge"):
    st.markdown(f"**🔊 Nghe thử giọng đọc:** {label}")
    if st.button("▶️ Nghe thử giọng đọc mẫu", key="btn_tts_preview", width="stretch"):
        spinner_text = ("Đang tạo giọng đọc mẫu (lần đầu dùng VieNeu sẽ tải model, hơi lâu)..."
                        if engine == "vieneu" else "Đang tạo giọng đọc mẫu...")
        with st.spinner(spinner_text):
            try:
                preview_ext = "wav" if engine == "vieneu" else "mp3"
                tts_preview_path = os.path.join(st.session_state.workdir, f"tts_preview.{preview_ext}")
                generate_tts_preview_audio(voice, rate, volume, pitch, tts_preview_path, engine=engine)
                with open(tts_preview_path, "rb") as f:
                    st.session_state.tts_preview_bytes = f.read()
                st.session_state.tts_preview_format = f"audio/{preview_ext}"
            except Exception as e:
                hint = "pip install -U vieneu" if engine == "vieneu" else "pip install -U edge-tts"
                st.error(f"Không tạo được giọng đọc mẫu: {e}  — thử chạy: {hint}")
                st.session_state.tts_preview_bytes = None
    if st.session_state.get("tts_preview_bytes"):
        st.audio(st.session_state.tts_preview_bytes, format=st.session_state.get("tts_preview_format", "audio/mp3"))


# ============================================================
# GIAO DIỆN CHÍNH
# ============================================================

# ---------- BƯỚC 1: TẢI VIDEO ----------
with st.container(key="card_upload"):
    step_header(1, "Tải video lên", "Hỗ trợ mp4, mkv, mov, avi — giới hạn dung lượng chỉnh trong .streamlit/config.toml")
    uploaded_file = st.file_uploader(
        "Chọn hoặc kéo thả file video", type=["mp4", "mkv", "mov", "avi"], label_visibility="collapsed"
    )

    active_video_path = None
    active_video_name = None
    if uploaded_file is not None:
        active_video_path = get_persistent_video_path(uploaded_file)
        active_video_name = uploaded_file.name

    preview_frame_path = None
    preview_video_res = None
    detected_box = None

    if active_video_path is not None:
        with st.spinner("Đang chuẩn bị khung hình xem trước..."):
            preview_frame_path, preview_video_res = ensure_preview_frame(active_video_path)
        if preview_frame_path is None:
            st.warning("Không trích được khung hình xem trước — đã thử nhiều cách nhưng đều thất bại. "
                       "Vẫn xử lý được, chỉ là không xem trước được.")
            preview_err = st.session_state.get("preview_frame_error")
            if preview_err:
                with st.expander("🔍 Xem chi tiết lỗi kỹ thuật (để biết vì sao)"):
                    st.code(preview_err)
                st.caption(
                    "💡 Nguyên nhân thường gặp: file hỏng/tải chưa xong, codec hiếm mà ffmpeg đang dùng "
                    "không hỗ trợ, hoặc file không phải video. Thử tải lại, đổi sang .mp4 H.264 (vd bằng HandBrake), "
                    "hoặc kiểm tra ffmpeg đã cài đủ codec."
                )
        else:
            pw0, ph0 = preview_video_res
            with st.spinner("🔎 Đang tự động phát hiện vùng phụ đề gốc..."):
                detected_box = ensure_detected_box(active_video_path, pw0, ph0)

            m1, m2, m3 = st.columns(3)
            m1.metric("Độ phân giải", f"{pw0}×{ph0}")
            m2.metric("Tệp", (active_video_name[:18] + "…") if len(active_video_name) > 19 else active_video_name)
            m3.metric("Vùng phụ đề gốc", "Đã phát hiện ✅" if detected_box else "Mặc định")
    else:
        st.caption("👆 Tải video lên để bắt đầu — bản xem trước sẽ xuất hiện ngay bên dưới.")

pw, ph = preview_video_res if preview_video_res else (None, None)

default_font_size = 20
if ph:
    default_font_size = max(14, min(48, int(ph / 18)))

# ---------- BƯỚC 2: 3 LỰA CHỌN ----------
with st.container(key="card_style"):
    step_header(2, "Chọn phong cách", "Chỉ 3 điều cần chọn — mọi thứ còn lại đã tự động tối ưu")

    c1, c2 = st.columns([3, 2])
    with c1:
        font_size = st.slider("🔤 Cỡ chữ phụ đề", 10, 60, default_font_size, step=1)
    with c2:
        text_color = st.color_picker("🎨 Màu chữ phụ đề", "#FFFFFF")

    voice_options = get_voice_options()
    voice_style_label = st.selectbox(
        "🎙️ Giọng đọc lồng tiếng AI", list(voice_options.keys()),
        help="👩👨 = edge-tts (cần Internet). 🦜 = VieNeu-TTS (chạy offline trên máy, cần: pip install vieneu; "
             "lần đầu dùng sẽ tải model). Nếu VieNeu lỗi, app tự chuyển sang giọng Hoài My để video không bị hỏng.",
    )
    voice_style = voice_options[voice_style_label]

outline_color = auto_outline_color(text_color)

# ---------- BƯỚC 3: VIRAL BOOST ----------
with st.container(key="card_viral"):
    step_header(3, "🚀 Viral Boost", "Tăng sức hút cho video — bật cái nào dùng cái đó, tất cả đều không bắt buộc")

    tab_img, tab_txt, tab_aud, tab_pub = st.tabs(["🎞️ Hình ảnh", "🪝 Chữ & nhận diện", "🔊 Nhịp & âm thanh", "📦 Xuất bản"])

    with tab_img:
        v_hflip = st.toggle("🪞 Lật gương video", value=False,
                            help="Lật ngang toàn bộ hình. Phụ đề mới vẫn đọc đúng chiều.")
        v_zoom = st.slider("🔍 Zoom nhẹ vào giữa (%)", 0, 10, 0, step=1,
                           help="Phóng to nhẹ để khung hình gọn hơn và loại bỏ viền thừa. 0 = tắt.")
        v_color = st.toggle("🎨 Tăng màu & độ nét", value=False,
                            help="Tăng nhẹ tương phản, độ bão hoà và độ nét để hình bắt mắt hơn.")
        v_vertical = st.toggle("📱 Khung dọc 9:16 (Shorts / Reels / TikTok)", value=False,
                               help="Xuất 1080×1920, video nằm giữa, nền là chính video đó làm mờ. Không hiện trong ảnh xem trước.")
        v_logo_label = st.segmented_control(
            "🙈 Che logo / watermark gốc ở góc", list(LOGO_CORNERS.keys()), default="Không che",
            selection_mode="single",
        ) or "Không che"
        v_logo = LOGO_CORNERS[v_logo_label]

    with tab_txt:
        v_hook = st.text_input("🪝 Chữ hook 3–4 giây đầu", value="", max_chars=80,
                               placeholder="VD: Không ai ngờ chuyện này xảy ra...",
                               help="Dòng chữ vàng to hiện ở đầu video để giữ chân người xem.")
        v_watermark = st.text_input("🏷️ Watermark tên kênh (góc trên phải)", value="", max_chars=40,
                                    placeholder="VD: @TenKenhCuaBan")
        v_progress = st.toggle("📊 Thanh tiến trình chạy dưới đáy video", value=False,
                               help="Thanh đỏ chạy dần giúp người xem muốn xem đến hết video.")

    with tab_aud:
        v_speed = st.slider("⏩ Tăng tốc nhẹ", 1.00, 1.15, 1.00, step=0.01, format="%.2fx",
                            help="Tăng tốc cả hình và tiếng. 1.05x gần như không nhận ra nhưng giúp video gọn, nhịp nhanh hơn.")
        v_fade = st.toggle("🌅 Fade vào / ra đầu cuối video", value=False)
        v_loud = st.toggle("🔊 Chuẩn hoá âm lượng (-14 LUFS chuẩn YouTube)", value=True,
                           help="Đưa âm lượng về mức chuẩn để không quá nhỏ hoặc quá to so với video khác.")

    with tab_pub:
        v_channel = st.text_input("📺 Tên kênh (dùng trong mô tả & hashtag)", value="", max_chars=40)
        v_thumb = st.toggle("🖼️ Tạo thumbnail tự động", value=True,
                            help="Lấy khung hình đẹp, tăng màu, thêm chữ vàng viền đen.")
        v_thumb_title = st.text_input("Chữ trên thumbnail", value="", max_chars=60,
                                      placeholder="Để trống = dùng chữ hook, hoặc câu đầu tiên của video",
                                      disabled=not v_thumb)
        v_pack = st.toggle("📝 Tạo gói YouTube (tiêu đề, mô tả, mốc thời gian, hashtag)", value=True,
                           help="Gợi ý dựa trên nội dung đã dịch. Nhớ chỉnh lại cho đúng ý trước khi đăng.")

    st.html('<div class="note">⚠️ Chỉ đăng lại video bạn có quyền sử dụng hoặc đã được tác giả đồng ý, và nên ghi nguồn. '
            'YouTube ưu tiên nội dung có đóng góp mới (dịch, lồng tiếng, bình luận) — các hiệu ứng ở đây không đảm bảo '
            'video qua được kiểm tra bản quyền.</div>')

# ---------- TUỲ CHỈNH NÂNG CAO ----------
with st.expander("⚙️ Tuỳ chỉnh nâng cao (không bắt buộc)"):
    st.caption("Chỉ mở phần này nếu bạn muốn tự tay ghi đè các lựa chọn tự động.")

    t1, t2 = st.columns(2)
    with t1:
        adv_burn_sub = st.toggle("Ghi phụ đề chữ lên video", value=True)
    with t2:
        adv_bilingual = st.toggle("Song ngữ (Trung + Việt) trong .srt", value=False)

    adv_position_label = st.segmented_control(
        "Vị trí phụ đề mới", list(POSITION_OPTIONS.keys()), default=DEFAULT_POSITION_LABEL,
        selection_mode="single",
    ) or DEFAULT_POSITION_LABEL
    adv_margin_v = st.slider("Cách mép (px)", 0, 150, int(25 * ((ph or 360) / 360)), step=5)

    st.divider()
    adv_manual_old_box = st.toggle("Tự chỉnh tay vùng che phụ đề gốc (thay vì tự động phát hiện)", value=False)
    adv_disable_remove_old = st.toggle("KHÔNG xoá/che phụ đề gốc", value=False)
    adv_old_method_label = st.segmented_control(
        "Cách che phụ đề gốc",
        ["Làm mờ (blur)", "Khung màu đặc", "Làm mượt (delogo)"],
        default="Làm mờ (blur)", selection_mode="single",
    ) or "Làm mờ (blur)"
    if "delogo" in adv_old_method_label:
        adv_old_method_key = "delogo"
    elif "màu" in adv_old_method_label:
        adv_old_method_key = "solid"
    else:
        adv_old_method_key = "blur"

    manual_box_pct = None
    if adv_manual_old_box:
        old_x_pct = st.slider("Vị trí ngang (%)", 0, 100, 5, step=1, key="adv_old_x_pct")
        old_y_pct = st.slider("Vị trí dọc (%)", 0, 100, 80, step=1, key="adv_old_y_pct")
        old_w_pct = st.slider("Chiều rộng vùng che (%)", 4, 100, 90, step=1, key="adv_old_w_pct")
        old_h_pct = st.slider("Chiều cao vùng che (%)", 3, 60, 14, step=1, key="adv_old_h_pct")
        manual_box_pct = (old_x_pct, old_y_pct, old_w_pct, old_h_pct)

    adv_box_color = st.color_picker("Màu khung che (chỉ cho 'khung màu đặc')", "#000000",
                                     disabled=adv_old_method_key != "solid")
    adv_blur_strength = st.slider("Độ mờ (chỉ cho 'làm mờ')", 5, 50, 25, disabled=adv_old_method_key != "blur")

    st.divider()
    adv_disable_dub = st.toggle("KHÔNG lồng tiếng AI (chỉ dịch phụ đề chữ)", value=False)
    adv_keep_bg = st.toggle("Giữ tiếng nền/nhạc nền gốc phía sau giọng đọc AI", value=True)
    adv_override_bg_volume = st.toggle("Tự chỉnh tay âm lượng nền gốc (mặc định tự động ở mức thấp)", value=False)
    adv_bg_volume_pct = st.slider("Âm lượng nền gốc (%)", 0, 100, AUTO_BG_VOLUME_PCT, step=1,
                                   disabled=not adv_override_bg_volume)

    st.divider()
    st.markdown("**🦜 Giọng VieNeu-TTS**")
    st.caption("Cài: `pip install -U vieneu` (lần đầu dùng sẽ tự tải model). Bấm nút bên dưới để quét toàn bộ "
               "giọng có sẵn và thêm vào danh sách giọng ở bước 2.")
    if st.button("🔄 Quét tất cả giọng VieNeu có sẵn", key="btn_scan_vieneu"):
        with st.spinner("Đang khởi tạo VieNeu-TTS và lấy danh sách giọng (lần đầu có thể lâu)..."):
            try:
                scanned = _parse_vieneu_voices(load_vieneu_engine().list_preset_voices())
                if scanned:
                    st.session_state.vieneu_voices = scanned
                    st.toast(f"Đã thêm {len(scanned)} giọng VieNeu vào danh sách giọng!", icon="🦜")
                    st.rerun()
                else:
                    st.warning("VieNeu không trả về giọng nào, đang dùng danh sách mặc định.")
            except ImportError:
                st.error("Chưa cài VieNeu-TTS. Hãy chạy: pip install -U vieneu rồi khởi động lại app.")
            except Exception as e:
                st.error(f"Không lấy được danh sách giọng VieNeu: {type(e).__name__}: {e}")
    if st.session_state.get("vieneu_voices"):
        st.caption(f"Đang dùng {len(st.session_state.vieneu_voices)} giọng VieNeu đã quét.")

    st.divider()
    adv_enhance_audio = st.toggle(
        "Lọc nhiễu & cân bằng âm thanh trước khi nhận diện", value=True,
        help="Cắt tiếng ù/rè, khử nhiễu nền và cân bằng âm lượng để nhận diện giọng nói chính xác hơn. "
             "Chỉ ảnh hưởng bước nhận diện, không đổi âm thanh của video kết quả.",
    )
    adv_model_size = st.selectbox(
        "Model nhận diện giọng nói", ["tiny", "base", "small", "medium", "large-v3"], index=2,
        help="Model lớn hơn nhận diện chính xác hơn nhưng chậm hơn. 'small' là mức khởi điểm tốt cho tiếng Trung; "
             "video nhiều tạp âm/nhạc nền nên chọn 'medium' hoặc 'large-v3'.",
    )
    adv_speed_label = st.segmented_control(
        "Tốc độ xử lý", list(SPEED_PRESETS.keys()), default=DEFAULT_SPEED_LABEL, selection_mode="single",
    ) or DEFAULT_SPEED_LABEL

# ---------- Gộp giá trị ----------
burn_sub = adv_burn_sub
bilingual = adv_bilingual
position_label = adv_position_label
margin_v = adv_margin_v
model_size = adv_model_size
speed_label = adv_speed_label
speed_cfg = SPEED_PRESETS[speed_label]

remove_old_sub = not adv_disable_remove_old
old_sub_method_key = adv_old_method_key
box_color = adv_box_color
blur_strength = adv_blur_strength

enable_dub = not adv_disable_dub
keep_bg = adv_keep_bg
voice = voice_style["voice"]
tts_engine = voice_style.get("engine", "edge")
tts_rate_pct = voice_style["rate"]
tts_volume_pct = 0
tts_pitch_hz = voice_style["pitch"]

if enable_dub:
    original_volume_pct = adv_bg_volume_pct if adv_override_bg_volume else AUTO_BG_VOLUME_PCT
else:
    original_volume_pct = AUTO_NO_DUB_VOLUME_PCT

old_sub_box = None
if remove_old_sub and pw and ph:
    if adv_manual_old_box and manual_box_pct:
        old_sub_box = compute_free_box(pw, ph, *manual_box_pct)
    elif detected_box:
        old_sub_box = detected_box

# ---------- BƯỚC 4: XEM TRƯỚC ----------
with st.container(key="card_preview"):
    step_header(4, "Xem trước", "Mô phỏng nhanh trên 1 khung hình của video")

    if active_video_path is None:
        st.caption("Tải video lên ở bước 1 để xem bản xem trước.")
    elif preview_frame_path is None:
        st.warning("Không có khung hình xem trước cho video này — vẫn xử lý được.")
        preview_err = st.session_state.get("preview_frame_error")
        if preview_err:
            with st.expander("🔍 Xem chi tiết lỗi kỹ thuật"):
                st.code(preview_err)
    else:
        combined_img = Image.open(preview_frame_path).convert("RGB")
        preview_box = old_sub_box
        if v_hflip:
            combined_img = ImageOps.mirror(combined_img)
            if preview_box:
                bx, by, bw, bh = preview_box
                preview_box = (combined_img.width - bx - bw, by, bw, bh)
        if remove_old_sub and preview_box is not None:
            combined_img = render_old_sub_box_preview(combined_img, preview_box, old_sub_method_key, box_color, blur_strength)
        if burn_sub:
            combined_img = render_subtitle_style_preview(
                combined_img, SAMPLE_SUBTITLE_TEXT, font_size, text_color, outline_color,
                POSITION_OPTIONS[position_label], margin_v,
            )
        if v_zoom > 0:
            combined_img = render_zoom_preview(combined_img, v_zoom)
        if v_logo:
            combined_img = render_old_sub_box_preview(
                combined_img, compute_logo_box(combined_img.width, combined_img.height, v_logo), "blur", "#000000", 25
            )
        if v_hook.strip() or v_watermark.strip():
            combined_img = render_text_overlays_preview(combined_img, v_hook, v_watermark)
        st.image(combined_img, width="stretch")

        legend = []
        if remove_old_sub:
            legend.append("🟥 Khung đỏ = vùng bị che (phụ đề gốc / logo góc)")
        if burn_sub:
            legend.append("🔤 Chữ mẫu = kiểu phụ đề mới")
        if v_hook.strip():
            legend.append("🪝 Chữ vàng = hook đầu video")
        if legend:
            st.html('<div class="legend">' + "".join(f"<span>{t}</span>" for t in legend) + "</div>")
        st.caption("Hình mô phỏng gần đúng, kết quả thật có thể chênh lệch nhẹ."
                   + ("" if (not remove_old_sub or adv_manual_old_box)
                      else " Nếu khung đỏ chưa khớp, mở '⚙️ Tuỳ chỉnh nâng cao' để tự kéo tay."))

        if enable_dub:
            st.divider()
            tts_preview_fragment(
                voice_style_label, voice,
                f"{'+' if tts_rate_pct >= 0 else ''}{tts_rate_pct}%",
                f"{'+' if tts_volume_pct >= 0 else ''}{tts_volume_pct}%",
                f"{'+' if tts_pitch_hz >= 0 else ''}{tts_pitch_hz}Hz",
                engine=tts_engine,
            )

# ---------- BƯỚC 5: XỬ LÝ ----------
with st.container(key="card_run"):
    step_header(5, "Bắt đầu xử lý", "Nhận diện → Dịch → Che phụ đề gốc → Ghi phụ đề mới → Lồng tiếng → Viral Boost")
    start_button = st.button("🚀 Bắt đầu xử lý", type="primary", width="stretch", disabled=active_video_path is None)

if start_button:
    need_finish = any([v_zoom > 0, v_color, v_logo, v_vertical, v_hook.strip(), v_watermark.strip(),
                       v_progress, v_fade, v_loud, abs(v_speed - 1.0) > 0.001])
    if not burn_sub and not enable_dub and not remove_old_sub and not v_hflip and not need_finish:
        st.warning("Bạn cần bật ít nhất một chức năng: ghi phụ đề chữ, xoá phụ đề gốc, lồng tiếng AI hoặc một hiệu ứng Viral Boost.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ext = os.path.splitext(active_video_name or active_video_path)[1] or ".mp4"
        input_path = os.path.join(tmp_dir, "input" + ext)
        shutil.copy(active_video_path, input_path)

        audio_path = os.path.join(tmp_dir, "audio.wav")
        srt_path = os.path.join(tmp_dir, "output.srt")

        status = st.status("Đang xử lý...", expanded=True)
        finish_notes = []
        thumb_bytes = None
        pack_text = None
        dub_warning_count = 0
        dub_error = None
        dub_notice = None
        untranslated_count = 0

        try:
            with status:
                st.write("🔊 Đang tách & lọc nhiễu âm thanh..." if adv_enhance_audio else "🔊 Đang tách âm thanh...")
                extract_audio(input_path, audio_path, enhance=adv_enhance_audio)
                total_duration = ffprobe_duration(input_path)

                st.write(f"🗣️ Đang nhận diện giọng nói tiếng Trung (model {model_size})...")
                transcribe_placeholder = st.empty()
                segments = transcribe_audio(
                    audio_path, model_size, beam_size=speed_cfg["beam_size"],
                    progress_cb=lambda t, text: transcribe_placeholder.write(f"  [{t:.1f}s] {text}")
                )

                if not segments:
                    status.update(label="Không nhận diện được giọng nói nào.", state="error")
                    st.error("Không nhận diện được giọng nói nào. Thử chọn model lớn hơn (medium / large-v3) "
                             "trong '⚙️ Tuỳ chỉnh nâng cao', hoặc kiểm tra video có lời nói tiếng Trung rõ ràng.")
                    st.stop()
                st.write(f"✅ Nhận diện xong {len(segments)} câu.")

                st.write("🌐 Đang dịch sang tiếng Việt...")
                translate_bar = st.progress(0, text="Đang dịch...")
                segments = translate_segments(
                    segments, progress_cb=lambda d, t: translate_bar.progress(d / t, text=f"Đã dịch {d}/{t} câu")
                )
                untranslated_count = sum(1 for s in segments if not s.get("translated_ok", True))

                srt_content = build_srt(segments, bilingual=bilingual)
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)

                original_volume_factor = original_volume_pct / 100.0
                current_video = input_path

                if remove_old_sub or burn_sub or v_hflip:
                    st.write("🎞️ Đang xử lý hình ảnh video (lật gương / xoá phụ đề gốc / ghi phụ đề mới)...")
                    resolution = get_video_resolution(current_video) or st.session_state.get("preview_video_res")
                    final_old_sub_box = None
                    if (remove_old_sub or v_hflip) and resolution is None:
                        raise FFmpegError(
                            "Không xác định được độ phân giải video để xử lý hình ảnh — "
                            "file có thể bị lỗi hoặc dùng codec không được hỗ trợ."
                        )
                    if remove_old_sub:
                        width, height = resolution
                        if adv_manual_old_box and manual_box_pct:
                            final_old_sub_box = compute_free_box(width, height, *manual_box_pct)
                        else:
                            final_old_sub_box = detect_subtitle_region(current_video, width, height, tmp_dir)
                        if final_old_sub_box:
                            bx, by, bw, bh = final_old_sub_box
                            bw = min(bw, width - bx - 2)
                            bh = min(bh, height - by - 2)
                            if bw < 4 or bh < 4:
                                final_old_sub_box = None
                            else:
                                final_old_sub_box = (bx, by, bw, bh)

                    processed_output = os.path.join(tmp_dir, "processed.mp4")
                    process_video(
                        current_video, processed_output,
                        remove_old_sub=remove_old_sub, old_sub_method=old_sub_method_key,
                        old_sub_box=final_old_sub_box, old_sub_color=hex_to_ffmpeg_color(box_color),
                        blur_strength=blur_strength,
                        burn_new_sub=burn_sub, srt_path=srt_path,
                        font_size=font_size,
                        primary_color=hex_to_ass_color(text_color),
                        outline_color=hex_to_ass_color(outline_color),
                        alignment=POSITION_OPTIONS[position_label],
                        margin_v=margin_v,
                        video_preset=speed_cfg["video_preset"],
                        hflip=v_hflip, frame_width=resolution[0] if resolution else None,
                    )
                    current_video = processed_output

                if enable_dub:
                    if tts_engine == "vieneu":
                        st.write(f"🦜 Đang tạo giọng đọc VieNeu-TTS ({voice_style_label}) — đọc lần lượt từng câu, "
                                 "lần đầu có thể phải tải model nên hơi lâu...")
                    else:
                        st.write(f"🎙️ Đang tạo giọng đọc AI ({voice_style_label}, song song nhiều luồng)...")
                    tts_bar = st.progress(0, text="Đang tạo giọng đọc...")
                    rate_str = f"{'+' if tts_rate_pct >= 0 else ''}{tts_rate_pct}%"
                    volume_str = f"{'+' if tts_volume_pct >= 0 else ''}{tts_volume_pct}%"
                    pitch_str = f"{'+' if tts_pitch_hz >= 0 else ''}{tts_pitch_hz}Hz"

                    st.session_state["_dub_notice"] = None
                    dub_track_path, failed_indices, tts_error = build_dub_track(
                        segments, voice, rate_str, volume_str, pitch_str, tmp_dir, total_duration,
                        concurrency=speed_cfg["tts_concurrency"],
                        progress_cb=lambda d, t: tts_bar.progress(d / t, text=f"Đã đọc {d}/{t} câu"),
                        engine=tts_engine,
                    )
                    dub_warning_count = len(failed_indices)
                    dub_notice = st.session_state.pop("_dub_notice", None)

                    if dub_track_path is None:
                        # Không tạo được giọng đọc nào → giữ nguyên âm thanh gốc, báo lỗi rõ ràng sau khi xong.
                        dub_error = tts_error
                        st.write("⚠️ Không tạo được giọng lồng tiếng — giữ nguyên âm thanh gốc.")
                    else:
                        st.write("🔀 Đang ghép giọng lồng tiếng vào video và tự động hạ âm lượng gốc...")
                        dub_output = os.path.join(tmp_dir, "with_dub.mp4")
                        combine_video_with_dub(current_video, dub_track_path, dub_output,
                                                keep_bg=keep_bg, bg_volume=original_volume_factor)
                        current_video = dub_output
                elif original_volume_factor != 1.0:
                    st.write("🔊 Đang điều chỉnh âm lượng video gốc...")
                    vol_output = os.path.join(tmp_dir, "with_volume.mp4")
                    adjust_video_volume(current_video, vol_output, original_volume_factor)
                    current_video = vol_output

                if need_finish:
                    st.write("🚀 Đang áp dụng Viral Boost (zoom, màu, chữ hook, tốc độ, 9:16, âm lượng...)...")
                    final_output = os.path.join(tmp_dir, "final.mp4")
                    finish_notes = finish_video(
                        current_video, final_output, tmp_dir,
                        speed=v_speed, zoom_pct=v_zoom, color_boost=v_color, logo_corner=v_logo,
                        vertical=v_vertical, hook_text=v_hook, watermark_text=v_watermark,
                        progress_bar=v_progress, fade=v_fade, loudnorm=v_loud,
                        video_preset=speed_cfg["video_preset"],
                    )
                    current_video = final_output

                if v_thumb:
                    st.write("🖼️ Đang tạo thumbnail...")
                    thumb_path = os.path.join(tmp_dir, "thumbnail.jpg")
                    thumb_title = (v_thumb_title.strip() or v_hook.strip()
                                   or _shorten(segments[0]["translated"], 40))
                    if make_thumbnail(current_video, thumb_title, thumb_path, vertical=v_vertical):
                        with open(thumb_path, "rb") as f:
                            thumb_bytes = f.read()
                    else:
                        finish_notes.append("Không tạo được thumbnail từ video này.")

                if v_pack:
                    st.write("📝 Đang soạn gói YouTube (tiêu đề, mô tả, hashtag)...")
                    pack_text = build_youtube_pack(
                        segments, total_duration, speed=v_speed, channel_name=v_channel,
                        vertical=v_vertical, custom_title=v_hook,
                    )

            status.update(label="Hoàn tất!", state="complete", expanded=False)
        except FFmpegError as e:
            status.update(label="Xử lý thất bại ❌", state="error")
            st.error(f"❌ {e}")
            st.info("💡 Lỗi thường gặp: video có codec lạ, không có audio, vùng che phụ đề tràn khung hình, hoặc ffmpeg thiếu "
                     "bộ lọc (vd drawtext). Thử video khác, tắt bớt hiệu ứng Viral Boost hoặc mở '⚙️ Tuỳ chỉnh nâng cao' "
                     "để chỉnh vùng che.")
            st.stop()
        except Exception as e:
            status.update(label="Xử lý thất bại ❌", state="error")
            st.error(f"❌ Lỗi không mong muốn: {e}")
            st.stop()

        if dub_error:
            st.error(
                "❌ Không tạo được giọng lồng tiếng AI tiếng Việt nên video đang giữ nguyên âm thanh gốc.\n\n"
                f"Nguyên nhân: {dub_error}"
            )
        elif dub_warning_count > 0:
            st.warning(
                f"⚠️ {dub_warning_count} câu AI không đọc được (có thể do mất kết nối mạng) "
                "— các câu này đã được thay bằng khoảng lặng thay vì làm hỏng toàn bộ video."
            )
        if dub_notice:
            st.warning(f"⚠️ {dub_notice}")
        if untranslated_count > 0:
            st.warning(
                f"⚠️ {untranslated_count} câu không dịch được sang tiếng Việt (có thể do mạng chập chờn) — "
                "các câu này giữ nguyên chữ Trung trong phụ đề và không được lồng tiếng."
            )
        for note in finish_notes:
            st.warning(f"⚠️ {note}")

        with open(current_video, "rb") as f:
            video_bytes = f.read()

        st.toast("Xử lý xong!", icon="✅")

        with st.container(key="card_result"):
            step_header(6, "Kết quả cuối cùng", "Xem lại, tải video, phụ đề, thumbnail và gói YouTube")

            r1, r2, r3 = st.columns(3)
            r1.metric("Số câu", len(segments))
            r2.metric("Thời lượng", f"{total_duration / v_speed:.0f}s")
            r3.metric("Câu lỗi TTS", dub_warning_count)

            tab_names = ["🎬 Video", "📜 Phụ đề"]
            if thumb_bytes:
                tab_names.append("🖼️ Thumbnail")
            if pack_text:
                tab_names.append("📦 Gói YouTube")
            tabs = st.tabs(tab_names)

            with tabs[0]:
                st.video(video_bytes)
            with tabs[1]:
                for seg in segments:
                    st.markdown(f"**[{seg['start']:.1f}s – {seg['end']:.1f}s]**")
                    if bilingual:
                        st.write(f"🇨🇳 {seg['text']}")
                    st.write(f"🇻🇳 {seg['translated']}")
                    st.divider()
            tab_idx = 2
            if thumb_bytes:
                with tabs[tab_idx]:
                    st.image(thumb_bytes, width="stretch")
                    st.download_button("⬇️ Tải thumbnail (.jpg)", data=thumb_bytes, file_name="thumbnail.jpg",
                                        mime="image/jpeg", width="stretch", on_click="ignore")
                tab_idx += 1
            if pack_text:
                with tabs[tab_idx]:
                    st.caption("Bấm biểu tượng sao chép ở góc phải khung bên dưới. Nhớ chỉnh lại cho đúng nội dung video.")
                    st.code(pack_text, language=None)
                    st.download_button("⬇️ Tải gói YouTube (.txt)", data=pack_text, file_name="goi_youtube.txt",
                                        mime="text/plain", width="stretch", on_click="ignore")

            colA, colB = st.columns(2)
            with colA:
                st.download_button("⬇️ Tải video kết quả", data=video_bytes, file_name="video_ket_qua.mp4",
                                    mime="video/mp4", type="primary", width="stretch", on_click="ignore")
            with colB:
                st.download_button("⬇️ Tải file phụ đề (.srt)", data=srt_content, file_name="phu_de.srt",
                                    mime="text/plain", width="stretch", on_click="ignore")

st.html('<div class="footer">Made with Streamlit · Whisper · edge-tts · VieNeu-TTS · ffmpeg</div>')
