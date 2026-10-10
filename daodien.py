"""
Dịch & lồng tiếng video Trung → Việt — BẢN TỰ ĐỘNG HOÁ (giao diện mới, Streamlit hiện đại).

Người dùng chỉ cần: tải video → chọn cỡ chữ → chọn màu chữ → chọn giọng đọc.
Phần còn lại (phát hiện & làm mờ phụ đề gốc, lồng tiếng AI, hạ âm lượng gốc,
màu viền tương phản, tốc độ xử lý...) được xử lý tự động.

CÀI ĐẶT:
    pip install -r requirements.txt        # cần streamlit >= 1.50
    # cần ffmpeg: sudo apt install ffmpeg / brew install ffmpeg

CHẠY:
    streamlit run app.py

CẤU TRÚC THƯ MỤC:
    app.py
    requirements.txt
    packages.txt                (Streamlit Cloud: ffmpeg)
    .streamlit/config.toml      (theme tối + nâng giới hạn upload lên 2GB)
"""

import asyncio
import hashlib
import os
import shutil
import subprocess
import tempfile

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont

st.set_page_config(
    page_title="Dịch & Lồng tiếng video Trung → Việt",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# GIAO DIỆN (CSS)
# ============================================================
APP_CSS = """
<style>
:root{
  --accent:#8b5cf6; --accent2:#22d3ee; --accent3:#f472b6;
  --card:rgba(255,255,255,.045); --card-border:rgba(255,255,255,.10);
  --muted:#9aa4c7;
}
.stApp{
  background:
    radial-gradient(900px 500px at 8% -5%, rgba(139,92,246,.28), transparent 60%),
    radial-gradient(800px 500px at 100% 0%, rgba(34,211,238,.18), transparent 55%),
    radial-gradient(700px 500px at 50% 110%, rgba(244,114,182,.16), transparent 60%),
    #0b1020;
  background-attachment: fixed;
}
header[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:860px; padding-top:2rem; padding-bottom:4rem;}

/* ---------- HERO ---------- */
.hero{
  position:relative; overflow:hidden; border-radius:26px; padding:34px 30px 28px;
  background:linear-gradient(135deg, rgba(139,92,246,.35), rgba(34,211,238,.18) 55%, rgba(244,114,182,.22));
  border:1px solid rgba(255,255,255,.14);
  box-shadow:0 20px 60px -20px rgba(139,92,246,.55);
  margin-bottom:1.4rem;
}
.hero:before{
  content:""; position:absolute; inset:-40% -10% auto auto; width:360px; height:360px; border-radius:50%;
  background:radial-gradient(circle, rgba(255,255,255,.18), transparent 65%);
}
.hero .badge{
  display:inline-block; font-size:.74rem; letter-spacing:.08em; text-transform:uppercase; font-weight:700;
  padding:5px 12px; border-radius:999px; background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.2);
  margin-bottom:12px;
}
.hero h1{
  margin:0 0 8px; font-size:2.15rem; line-height:1.15; font-weight:800; letter-spacing:-.02em;
  background:linear-gradient(90deg,#fff,#c4b5fd 45%,#67e8f9); -webkit-background-clip:text; background-clip:text; color:transparent;
}
.hero p{margin:0; color:#d5dbf5; font-size:1rem; max-width:620px;}
.hero .chips{display:flex; flex-wrap:wrap; gap:8px; margin-top:16px;}
.hero .chip{
  font-size:.82rem; padding:6px 12px; border-radius:999px;
  background:rgba(8,12,28,.45); border:1px solid rgba(255,255,255,.16);
}

/* ---------- CARD (st.container(key="card_...")) ---------- */
[class*="st-key-card_"]{
  background:var(--card); border:1px solid var(--card-border); border-radius:22px;
  padding:22px 22px 18px; margin-bottom:1.1rem;
  backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  box-shadow:0 10px 34px -18px rgba(0,0,0,.7);
  transition:border-color .25s ease, transform .25s ease;
}
[class*="st-key-card_"]:hover{border-color:rgba(139,92,246,.45);}

.step-head{display:flex; align-items:center; gap:14px; margin-bottom:6px;}
.step-num{
  flex:0 0 auto; width:38px; height:38px; border-radius:12px; display:grid; place-items:center;
  font-weight:800; color:#fff; background:linear-gradient(135deg,var(--accent),var(--accent2));
  box-shadow:0 8px 20px -6px rgba(139,92,246,.7);
}
.step-title{font-size:1.2rem; font-weight:700; margin:0; line-height:1.2;}
.step-desc{color:var(--muted); font-size:.9rem; margin:2px 0 0;}

/* ---------- WIDGETS ---------- */
[data-testid="stFileUploaderDropzone"]{
  border:2px dashed rgba(139,92,246,.55); border-radius:18px; padding:26px;
  background:linear-gradient(135deg, rgba(139,92,246,.10), rgba(34,211,238,.06));
  transition:all .25s ease;
}
[data-testid="stFileUploaderDropzone"]:hover{
  border-color:var(--accent2); background:linear-gradient(135deg, rgba(139,92,246,.18), rgba(34,211,238,.12));
}
.stButton>button, .stDownloadButton>button{
  border-radius:14px; font-weight:700; padding:.65rem 1.1rem; transition:all .2s ease;
  border:1px solid rgba(255,255,255,.16);
}
.stButton>button:hover, .stDownloadButton>button:hover{
  transform:translateY(-2px); border-color:var(--accent2); box-shadow:0 10px 24px -10px rgba(34,211,238,.6);
}
.stButton>button[kind="primary"]{
  border:none; color:#fff; font-size:1.08rem; padding:.9rem 1.2rem;
  background:linear-gradient(90deg,#8b5cf6,#6366f1 50%,#22d3ee);
  box-shadow:0 14px 34px -12px rgba(99,102,241,.9);
}
.stButton>button[kind="primary"]:hover{filter:brightness(1.1); box-shadow:0 18px 40px -12px rgba(34,211,238,.8);}
.stButton>button:disabled{opacity:.45; transform:none; box-shadow:none;}
[data-testid="stImage"] img, [data-testid="stVideo"] video{
  border-radius:16px; border:1px solid rgba(255,255,255,.12); box-shadow:0 18px 40px -22px #000;
}
[data-testid="stMetric"]{
  background:rgba(255,255,255,.05); border:1px solid var(--card-border); border-radius:16px; padding:12px 16px;
}
[data-testid="stMetricLabel"]{color:var(--muted);}
[data-testid="stExpander"]{border-radius:16px; border:1px solid var(--card-border); background:rgba(255,255,255,.03);}
[data-testid="stStatusWidget"], [data-testid="stStatus"]{border-radius:16px;}
.stTabs [data-baseweb="tab-list"]{gap:6px;}
.stTabs [data-baseweb="tab"]{border-radius:12px 12px 0 0; padding:.5rem 1rem;}
.hint{color:var(--muted); font-size:.85rem;}
.legend{
  display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 2px;
}
.legend span{
  font-size:.8rem; padding:5px 11px; border-radius:999px; background:rgba(255,255,255,.06);
  border:1px solid var(--card-border); color:#cfd6f3;
}
.footer{text-align:center; color:var(--muted); font-size:.8rem; margin-top:2rem;}

@media (max-width:640px){
  .hero{padding:24px 18px;} .hero h1{font-size:1.6rem;}
  [class*="st-key-card_"]{padding:16px 14px 12px;}
}
</style>
"""
st.html(APP_CSS)


def step_header(num: int, title: str, desc: str = ""):
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
  ghi phụ đề mới và lồng tiếng Việt.</p>
  <div class="chips">
    <span class="chip">🔎 Tự phát hiện phụ đề gốc</span>
    <span class="chip">🌐 Dịch tự động</span>
    <span class="chip">🎙️ Lồng tiếng AI</span>
    <span class="chip">🔉 Tự cân bằng âm lượng</span>
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

POSITION_OPTIONS = {"Dưới (mặc định)": 2, "Giữa màn hình": 5, "Trên": 8}

SPEED_PRESETS = {
    "⚡ Nhanh nhất": {"beam_size": 1, "video_preset": "ultrafast", "tts_concurrency": 12},
    "⚖️ Cân bằng": {"beam_size": 3, "video_preset": "veryfast", "tts_concurrency": 6},
    "🎯 Chính xác nhất": {"beam_size": 5, "video_preset": "medium", "tts_concurrency": 4},
}
DEFAULT_SPEED_LABEL = "⚡ Nhanh nhất"
DEFAULT_POSITION_LABEL = "Dưới (mặc định)"

CPU_COUNT = os.cpu_count() or 4

SAMPLE_SUBTITLE_TEXT = "Đây là câu phụ đề mẫu để xem trước"
SAMPLE_TTS_TEXT = "Xin chào, đây là giọng đọc mẫu để bạn nghe thử trước khi xử lý toàn bộ video."

AUTO_BG_VOLUME_PCT = 8
AUTO_NO_DUB_VOLUME_PCT = 100

# Giới hạn tăng tốc audio TTS khi ép khớp khung thời gian phụ đề (xem fit_audio_to_duration).
MAX_TTS_SPEEDUP = 1.15


# ============================================================
# CÁC HÀM XỬ LÝ CHUNG (logic giữ nguyên)
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


def extract_audio(video_path: str, audio_path: str):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
    _run_ffmpeg(cmd, label="tách âm thanh")


def ffprobe_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


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


def transcribe_audio(audio_path: str, model_size: str, beam_size: int = 3, progress_cb=None):
    model = load_whisper_model(model_size)
    segments, info = model.transcribe(audio_path, language="zh", vad_filter=True, beam_size=beam_size)
    results = []
    for seg in segments:
        results.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
        if progress_cb:
            progress_cb(seg.end, seg.text.strip())
    return results


def translate_segments(segments, progress_cb=None, chunk_size: int = 40):
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source="zh-CN", target="vi")

    texts = [seg["text"] for seg in segments]
    translated_all = [None] * len(segments)

    done = 0
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start:start + chunk_size]
        try:
            batch_result = translator.translate_batch(chunk)
        except Exception:
            batch_result = None

        if batch_result and len(batch_result) == len(chunk):
            for i, t in enumerate(batch_result):
                translated_all[start + i] = t if t else chunk[i]
        else:
            for i, text in enumerate(chunk):
                try:
                    translated_all[start + i] = translator.translate(text)
                except Exception:
                    translated_all[start + i] = text

        done = min(start + chunk_size, len(texts))
        if progress_cb:
            progress_cb(done, len(texts))

    for seg, translated in zip(segments, translated_all):
        seg["translated"] = translated if translated else seg["text"]
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


def process_video(video_path: str, output_path: str, *,
                   remove_old_sub: bool = False, old_sub_method: str = "blur",
                   old_sub_box=None, old_sub_color: str = "0x000000", blur_strength: int = 25,
                   burn_new_sub: bool = False, srt_path: str = None,
                   font_size: int = 20, primary_color: str = "&H00FFFFFF&",
                   outline_color: str = "&H00000000&", alignment: int = 2, margin_v: int = 25,
                   video_preset: str = "veryfast"):
    """Gộp bước xóa phụ đề gốc + ghi phụ đề mới vào MỘT lần encode video duy nhất."""
    filters = []
    stage = "[0:v]"
    counter = 0

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
    _run_ffmpeg(cmd, label="xử lý hình ảnh video (xoá phụ đề gốc / ghi phụ đề mới)")


def adjust_video_volume(video_path: str, output_path: str, volume_factor: float):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-filter:a", f"volume={volume_factor}",
        "-c:v", "copy",
        output_path,
    ]
    _run_ffmpeg(cmd, label="điều chỉnh âm lượng gốc")


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
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
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


def generate_tts_preview_audio(voice: str, rate: str, volume: str, pitch: str, out_path: str,
                                sample_text: str = SAMPLE_TTS_TEXT):
    import edge_tts
    communicate = edge_tts.Communicate(sample_text, voice=voice, rate=rate, volume=volume, pitch=pitch)
    _run_async(communicate.save(out_path))


# ============================================================
# LỒNG TIẾNG (TTS)
# ============================================================

def _run_async(coro):
    try:
        asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()


def synthesize_all_tts(items, voice: str, rate: str, volume: str, pitch: str,
                        max_retries: int = 3, concurrency: int = 6, progress_cb=None):
    import edge_tts

    failed_indices = set()
    done_count = 0
    lock = asyncio.Lock()

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
                except Exception:
                    await asyncio.sleep(0.6)
            if not success:
                failed_indices.add(idx)
        async with lock:
            done_count += 1
            if progress_cb:
                progress_cb(done_count, len(items))

    async def _synthesize_all():
        sem = asyncio.Semaphore(concurrency)
        await asyncio.gather(*[_synthesize_one(sem, idx, text, raw_path) for idx, text, raw_path in items])

    _run_async(_synthesize_all())
    return failed_indices


def fit_audio_to_duration(in_path: str, out_path: str, target_duration: float):
    """Khớp audio TTS vào khung thời gian của câu, tăng tốc tối đa MAX_TTS_SPEEDUP lần
    để giọng đọc luôn tự nhiên; phần tràn được xử lý ở build_dub_track() (không chồng tiếng)."""
    dur = ffprobe_duration(in_path)
    if dur <= target_duration or target_duration <= 0:
        subprocess.run(["ffmpeg", "-y", "-i", in_path, "-ar", "24000", "-ac", "1", out_path],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    factor = dur / target_duration
    factor = min(factor, MAX_TTS_SPEEDUP)

    if factor <= 1.02:
        subprocess.run(["ffmpeg", "-y", "-i", in_path, "-ar", "24000", "-ac", "1", out_path],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    subprocess.run(["ffmpeg", "-y", "-i", in_path, "-filter:a", f"atempo={factor:.3f}", "-ar", "24000", "-ac", "1", out_path],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def make_silence(out_path: str, duration: float):
    duration = max(duration, 0.1)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono:d={duration}", out_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def build_dub_track(segments, voice: str, rate: str, volume: str, pitch: str,
                     tmp_dir: str, total_duration: float, concurrency: int = 6, progress_cb=None):
    items = []
    for i, seg in enumerate(segments):
        text = seg["translated"].strip()
        if text:
            raw_path = os.path.join(tmp_dir, f"tts_raw_{i}.mp3")
            items.append((i, text, raw_path))

    failed_indices = synthesize_all_tts(items, voice, rate, volume, pitch,
                                         concurrency=concurrency, progress_cb=progress_cb)

    # Chống chồng tiếng: câu sau không bao giờ bắt đầu sớm hơn thời điểm câu trước kết thúc ("cursor").
    seg_files = []
    cursor = 0.0
    for idx, text, raw_path in items:
        seg = segments[idx]
        slot_duration = max(seg["end"] - seg["start"], 0.3)
        fit_path = os.path.join(tmp_dir, f"tts_fit_{idx}.wav")

        if idx in failed_indices:
            make_silence(fit_path, slot_duration)
            actual_dur = slot_duration
        else:
            fit_audio_to_duration(raw_path, fit_path, slot_duration)
            try:
                actual_dur = ffprobe_duration(fit_path)
            except Exception:
                actual_dur = slot_duration

        actual_start = max(seg["start"], cursor)
        cursor = actual_start + actual_dur

        seg_files.append((actual_start, fit_path))

    dub_track_duration = max(total_duration, cursor)

    dub_track_path = os.path.join(tmp_dir, "dub_track.wav")
    inputs = ["-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono:d={dub_track_duration}"]
    filter_parts = []
    amix_labels = ["[0:a]"]
    for idx, (start, path) in enumerate(seg_files):
        inputs += ["-i", path]
        delay_ms = int(start * 1000)
        filter_parts.append(f"[{idx + 1}:a]adelay={delay_ms}|{delay_ms}[a{idx}]")
        amix_labels.append(f"[a{idx}]")

    filter_complex = ";".join(filter_parts)
    if filter_complex:
        filter_complex += ";"
    filter_complex += "".join(amix_labels) + f"amix=inputs={len(amix_labels)}:duration=first:dropout_transition=0:normalize=0[aout]"

    cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_complex, "-map", "[aout]", dub_track_path]
    _run_ffmpeg(cmd, label="ghép các đoạn giọng đọc AI thành 1 track")
    return dub_track_path, failed_indices


def combine_video_with_dub(video_path: str, dub_track_path: str, output_path: str,
                            keep_bg: bool, bg_volume: float = 0.08):
    if keep_bg:
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-i", dub_track_path,
            "-filter_complex", f"[0:a]volume={bg_volume}[bg];[1:a]anull[dub];[bg][dub]amix=inputs=2:duration=first:normalize=0[aout]",
            "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-shortest", output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-i", dub_track_path,
            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-shortest", output_path,
        ]
    _run_ffmpeg(cmd, label="ghép giọng lồng tiếng AI vào video")


# ============================================================
# FRAGMENT: nghe thử giọng (bấm nút chỉ rerun riêng khối này, không chạy lại cả app)
# ============================================================

@st.fragment
def tts_preview_fragment(label: str, voice: str, rate: str, volume: str, pitch: str):
    st.markdown(f"**🔊 Nghe thử giọng đọc:** {label}")
    if st.button("▶️ Nghe thử giọng đọc mẫu", key="btn_tts_preview", width="stretch"):
        with st.spinner("Đang tạo giọng đọc mẫu..."):
            try:
                tts_preview_path = os.path.join(st.session_state.workdir, "tts_preview.mp3")
                generate_tts_preview_audio(voice, rate, volume, pitch, tts_preview_path)
                with open(tts_preview_path, "rb") as f:
                    st.session_state.tts_preview_bytes = f.read()
            except Exception as e:
                st.error(f"Không tạo được giọng đọc mẫu: {e}")
                st.session_state.tts_preview_bytes = None
    if st.session_state.get("tts_preview_bytes"):
        st.audio(st.session_state.tts_preview_bytes, format="audio/mp3")


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

    voice_style_label = st.selectbox("🎙️ Giọng đọc lồng tiếng AI", list(VOICE_STYLE_OPTIONS.keys()))
    voice_style = VOICE_STYLE_OPTIONS[voice_style_label]

outline_color = auto_outline_color(text_color)

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
    adv_model_size = st.selectbox("Model nhận diện giọng nói", ["tiny", "base", "small", "medium", "large-v3"], index=1)
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

# ---------- BƯỚC 3: XEM TRƯỚC ----------
with st.container(key="card_preview"):
    step_header(3, "Xem trước", "Mô phỏng nhanh trên 1 khung hình của video")

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
        if remove_old_sub and old_sub_box is not None:
            combined_img = render_old_sub_box_preview(combined_img, old_sub_box, old_sub_method_key, box_color, blur_strength)
        if burn_sub:
            combined_img = render_subtitle_style_preview(
                combined_img, SAMPLE_SUBTITLE_TEXT, font_size, text_color, outline_color,
                POSITION_OPTIONS[position_label], margin_v,
            )
        st.image(combined_img, width="stretch")

        legend = []
        if remove_old_sub:
            legend.append("🟥 Khung đỏ = vùng phụ đề gốc (tự phát hiện) sẽ bị che")
        if burn_sub:
            legend.append("🔤 Chữ mẫu = kiểu phụ đề mới")
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
            )

# ---------- BƯỚC 4: XỬ LÝ ----------
with st.container(key="card_run"):
    step_header(4, "Bắt đầu xử lý", "Nhận diện → Dịch → Che phụ đề gốc → Ghi phụ đề mới → Lồng tiếng")
    start_button = st.button("🚀 Bắt đầu xử lý", type="primary", width="stretch", disabled=active_video_path is None)

if start_button:
    if not burn_sub and not enable_dub and not remove_old_sub:
        st.warning("Bạn cần bật ít nhất một trong: ghi phụ đề chữ, xoá phụ đề gốc, hoặc lồng tiếng AI (xem Tuỳ chỉnh nâng cao).")
        st.stop()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ext = os.path.splitext(active_video_name or active_video_path)[1] or ".mp4"
        input_path = os.path.join(tmp_dir, "input" + ext)
        shutil.copy(active_video_path, input_path)

        audio_path = os.path.join(tmp_dir, "audio.wav")
        srt_path = os.path.join(tmp_dir, "output.srt")

        status = st.status("Đang xử lý...", expanded=True)

        try:
            with status:
                st.write("🔊 Đang tách âm thanh...")
                extract_audio(input_path, audio_path)
                total_duration = ffprobe_duration(input_path)

                st.write("🗣️ Đang nhận diện giọng nói tiếng Trung...")
                transcribe_placeholder = st.empty()
                segments = transcribe_audio(
                    audio_path, model_size, beam_size=speed_cfg["beam_size"],
                    progress_cb=lambda t, text: transcribe_placeholder.write(f"  [{t:.1f}s] {text}")
                )

                if not segments:
                    status.update(label="Không nhận diện được giọng nói nào.", state="error")
                    st.stop()
                st.write(f"✅ Nhận diện xong {len(segments)} câu.")

                st.write("🌐 Đang dịch sang tiếng Việt (theo batch)...")
                translate_bar = st.progress(0, text="Đang dịch...")
                segments = translate_segments(
                    segments, progress_cb=lambda d, t: translate_bar.progress(d / t, text=f"Đã dịch {d}/{t} câu")
                )

                srt_content = build_srt(segments, bilingual=bilingual)
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)

                original_volume_factor = original_volume_pct / 100.0
                current_video = input_path

                if remove_old_sub or burn_sub:
                    st.write("🎞️ Đang xử lý hình ảnh video (xoá phụ đề gốc / ghi phụ đề mới)...")
                    final_old_sub_box = None
                    if remove_old_sub:
                        resolution = get_video_resolution(current_video)
                        if resolution is None:
                            resolution = st.session_state.get("preview_video_res")
                        if resolution is None:
                            raise FFmpegError(
                                "Không xác định được độ phân giải video để xoá phụ đề gốc — "
                                "file có thể bị lỗi hoặc dùng codec không được hỗ trợ."
                            )
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
                    )
                    current_video = processed_output

                dub_warning_count = 0
                if enable_dub:
                    st.write(f"🎙️ Đang tạo giọng đọc AI ({voice_style_label}, song song nhiều luồng)...")
                    tts_bar = st.progress(0, text="Đang tạo giọng đọc...")
                    rate_str = f"{'+' if tts_rate_pct >= 0 else ''}{tts_rate_pct}%"
                    volume_str = f"{'+' if tts_volume_pct >= 0 else ''}{tts_volume_pct}%"
                    pitch_str = f"{'+' if tts_pitch_hz >= 0 else ''}{tts_pitch_hz}Hz"

                    dub_track_path, failed_indices = build_dub_track(
                        segments, voice, rate_str, volume_str, pitch_str, tmp_dir, total_duration,
                        concurrency=speed_cfg["tts_concurrency"],
                        progress_cb=lambda d, t: tts_bar.progress(d / t, text=f"Đã đọc {d}/{t} câu")
                    )
                    dub_warning_count = len(failed_indices)

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

            status.update(label="Hoàn tất!", state="complete", expanded=False)
        except FFmpegError as e:
            status.update(label="Xử lý thất bại ❌", state="error")
            st.error(f"❌ {e}")
            st.info("💡 Lỗi thường gặp: video có codec lạ, không có audio, hoặc vùng che phụ đề tràn khung hình. "
                     "Thử video khác hoặc mở '⚙️ Tuỳ chỉnh nâng cao' để tự chỉnh vùng che / tắt bớt bước xử lý.")
            st.stop()
        except Exception as e:
            status.update(label="Xử lý thất bại ❌", state="error")
            st.error(f"❌ Lỗi không mong muốn: {e}")
            st.stop()

        if dub_warning_count > 0:
            st.warning(
                f"⚠️ {dub_warning_count} câu AI không đọc được (có thể do mất kết nối mạng) "
                "— các câu này đã được thay bằng khoảng lặng thay vì làm hỏng toàn bộ video."
            )

        with open(current_video, "rb") as f:
            video_bytes = f.read()

        st.toast("Xử lý xong!", icon="✅")

        with st.container(key="card_result"):
            step_header(5, "Kết quả cuối cùng", "Xem lại, tải video và file phụ đề")

            r1, r2, r3 = st.columns(3)
            r1.metric("Số câu", len(segments))
            r2.metric("Thời lượng", f"{total_duration:.0f}s")
            r3.metric("Câu lỗi TTS", dub_warning_count)

            tab_video, tab_sub = st.tabs(["🎬 Video", "📜 Nội dung phụ đề"])
            with tab_video:
                st.video(video_bytes)
            with tab_sub:
                for seg in segments:
                    st.markdown(f"**[{seg['start']:.1f}s – {seg['end']:.1f}s]**")
                    if bilingual:
                        st.write(f"🇨🇳 {seg['text']}")
                    st.write(f"🇻🇳 {seg['translated']}")
                    st.divider()

            colA, colB = st.columns(2)
            with colA:
                st.download_button("⬇️ Tải video kết quả", data=video_bytes, file_name="video_ket_qua.mp4",
                                    mime="video/mp4", type="primary", width="stretch")
            with colB:
                st.download_button("⬇️ Tải file phụ đề (.srt)", data=srt_content, file_name="phu_de.srt",
                                    mime="text/plain", width="stretch")

st.html('<div class="footer">Made with Streamlit · Whisper · edge-tts · ffmpeg</div>')
