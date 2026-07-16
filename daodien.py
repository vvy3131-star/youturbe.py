"""
App Streamlit: Dịch phụ đề video tiếng Trung sang tiếng Việt — BẢN TỰ ĐỘNG HOÁ.

Người dùng chỉ cần:
    1) Tải video lên MÁY
    2) Chọn cỡ chữ phụ đề mới
    3) Chọn giọng đọc AI (tự động theo giới tính giọng gốc, hoặc tự chọn 1 giọng cố định)
    4) Chọn màu chữ phụ đề mới
Mọi thứ còn lại được xử lý TỰ ĐỘNG:
    - Tự động phát hiện vùng phụ đề tiếng Trung gốc (ghi cứng trên hình) và
      tự động làm mờ (blur) để che đi, không cần tự kéo khung.
    - Tự động NHẬN DIỆN GIỌNG NÓI GỐC trong video là NAM hay NỮ theo từng câu
      thoại (dựa trên cao độ giọng nói - pitch), rồi TỰ ĐỘNG chọn giọng đọc AI
      tiếng Việt tương ứng (nữ dùng giọng nữ, nam dùng giọng nam) — nếu muốn,
      vẫn có thể tự chọn hẳn 1 giọng cố định cho toàn bộ video thay vì để tự động.
    - Có nhiều giọng đọc để chọn: giọng Microsoft (Hoài My - nữ, Nam Minh - nam)
      và giọng nữ Google (qua gTTS) — có thể chọn giọng khác nhau cho phần nữ/nam.
    - Khung hình xem trước được trích xuất với nhiều lớp dự phòng để luôn cố
      lấy được ảnh, kể cả với video có định dạng/codec hơi lạ.
    - Tự động hạ âm lượng gốc xuống mức thấp nhất có thể mà vẫn giữ chút
      tiếng nền/nhạc nền phía sau giọng đọc AI (không tắt hẳn để video không
      bị "cụt" tiếng động).
    - Tự động chọn màu viền chữ tương phản với màu chữ đã chọn.
    - Tự động chọn vị trí, lề, model nhận diện, tốc độ xử lý ở mức cân bằng.

Vẫn có một khối "Tuỳ chỉnh nâng cao (không bắt buộc)" ẩn sẵn cho ai muốn tự
tay chỉnh từng thông số như bản gốc.

GIỚI HẠN DUNG LƯỢNG VIDEO:
    Mặc định Streamlit chỉ cho upload tối đa 200MB/file. Để nâng giới hạn
    này, tạo file `.streamlit/config.toml` cùng thư mục với app.py, nội dung:

        [server]
        maxUploadSize = 2048
        maxMessageSize = 2048

    (2048 = 2048MB = 2GB, có thể chỉnh số này tuỳ nhu cầu). File mẫu này
    được đính kèm sẵn — chỉ cần copy vào đúng vị trí.

CÀI ĐẶT (chạy 1 lần, local):
    pip install streamlit faster-whisper deep-translator edge-tts gTTS Pillow numpy --break-system-packages
    # cần có ffmpeg cài sẵn trên máy (sudo apt install ffmpeg / brew install ffmpeg)

CHẠY APP (local):
    streamlit run app.py

TRÊN STREAMLIT CLOUD:
    Cần có các file cùng thư mục gốc:
    - requirements.txt (streamlit, faster-whisper, deep-translator, edge-tts, gTTS, Pillow, numpy)
    - packages.txt (ffmpeg)
    - .streamlit/config.toml (xem phần "GIỚI HẠN DUNG LƯỢNG VIDEO" ở trên)
"""

import asyncio
import hashlib
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont

st.set_page_config(page_title="Dịch phụ đề & Lồng tiếng video Trung -> Việt", page_icon="🎬", layout="centered")

# CSS tối giản để gọn hơn trên điện thoại
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 780px;}
    div[data-testid="stVerticalBlock"] > div:has(> div.stButton) button {width: 100%;}
    .stTabs [data-baseweb="tab"] {padding: 0.4rem 0.8rem; font-size: 0.92rem;}
    div[data-testid="stSlider"] {padding-bottom: 0.2rem;}
</style>
""", unsafe_allow_html=True)

# ---------- DANH MỤC GIỌNG ĐỌC (nhiều nhà cung cấp) ----------
# "edge" = Microsoft edge-tts (giọng neural tự nhiên, hỗ trợ chỉnh tốc độ/cao độ trực tiếp).
# "gtts" = Google Translate TTS (qua thư viện gTTS) — thêm lựa chọn giọng khác,
#          nhưng KHÔNG hỗ trợ chỉnh cao độ, chỉ áp được tốc độ (qua hậu xử lý ffmpeg).
VOICE_CATALOG = {
    "female": [
        {"id": "edge_hoaimy", "label": "👩 Hoài My (Microsoft)", "provider": "edge", "voice": "vi-VN-HoaiMyNeural"},
        {"id": "google_vi_female", "label": "👩 Giọng nữ Google (gTTS)", "provider": "gtts", "voice": "vi"},
    ],
    "male": [
        {"id": "edge_namminh", "label": "👨 Nam Minh (Microsoft)", "provider": "edge", "voice": "vi-VN-NamMinhNeural"},
    ],
}
ALL_VOICES = VOICE_CATALOG["female"] + VOICE_CATALOG["male"]
VOICE_BY_LABEL = {v["label"]: v for v in ALL_VOICES}
GENDER_LABEL = {"female": "👩 Nữ", "male": "👨 Nam"}

STYLE_OPTIONS = {
    "🙂 Chuẩn": {"rate": 0, "pitch": 0},
    "🐢 Nhẹ nhàng, chậm": {"rate": -15, "pitch": -3},
    "⚡ Vui tươi, nhanh": {"rate": 15, "pitch": 3},
    "🎙️ Trầm ấm": {"rate": -5, "pitch": -6},
}

# ---------- VỊ TRÍ PHỤ ĐỀ MỚI (ASS Alignment - kiểu numpad) ----------
POSITION_OPTIONS = {
    "Dưới (mặc định)": 2,
    "Giữa màn hình": 5,
    "Trên": 8,
}

# ---------- CHẾ ĐỘ TỐC ĐỘ XỬ LÝ ----------
SPEED_PRESETS = {
    "⚡ Nhanh nhất": {"beam_size": 1, "video_preset": "ultrafast", "tts_concurrency": 10},
    "⚖️ Cân bằng (khuyến nghị)": {"beam_size": 3, "video_preset": "veryfast", "tts_concurrency": 6},
    "🎯 Chính xác nhất (chậm hơn)": {"beam_size": 5, "video_preset": "medium", "tts_concurrency": 4},
}

CPU_COUNT = os.cpu_count() or 4

SAMPLE_SUBTITLE_TEXT = "Đây là câu phụ đề mẫu để xem trước"
SAMPLE_TTS_TEXT = "Xin chào, đây là giọng đọc mẫu để bạn nghe thử trước khi xử lý toàn bộ video."

# Âm lượng nền gốc tự động khi có lồng tiếng AI (mức thấp nhất nhưng không
# tắt hẳn, để vẫn còn chút tiếng động/nhạc nền phía sau giọng đọc).
AUTO_BG_VOLUME_PCT = 8
# Âm lượng gốc tự động khi KHÔNG lồng tiếng (giữ nguyên).
AUTO_NO_DUB_VOLUME_PCT = 100

# Ngưỡng cao độ (Hz) để phân biệt giọng nữ/nam khi nhận diện tự động.
# Giọng nam trưởng thành thường ~85-180Hz, giọng nữ thường ~165-255Hz,
# nên 165Hz là ngưỡng phân tách phổ biến.
GENDER_PITCH_THRESHOLD_HZ = 165.0


# ============================================================
# CÁC HÀM XỬ LÝ CHUNG (không đổi logic xử lý video/audio thật)
# ============================================================

class FFmpegError(RuntimeError):
    """Lỗi ffmpeg có kèm theo vài dòng cuối của stderr để hiển thị cho người dùng,
    thay vì để Streamlit chặn và chỉ báo lỗi chung chung 'redacted'."""
    pass


def _run_ffmpeg(cmd, label: str = "ffmpeg"):
    """Chạy 1 lệnh ffmpeg/ffprobe, bắt stderr để có thể báo lỗi rõ ràng ra giao
    diện nếu thất bại (thay vì DEVNULL khiến lỗi thật bị nuốt mất)."""
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
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=s=x:p=0", path],
        capture_output=True, text=True, check=True,
    )
    w_str, h_str = result.stdout.strip().split("x")
    return int(w_str), int(h_str)


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
    """Tự động chọn màu viền chữ (đen hoặc trắng) tương phản với màu chữ đã chọn,
    để phụ đề luôn dễ đọc trên mọi nền video."""
    hex_color = text_color_hex.lstrip("#")
    if len(hex_color) != 6:
        return "#000000"
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000000" if luminance > 0.55 else "#FFFFFF"


def compute_free_box(width: int, height: int, x_pct: float, y_pct: float, w_pct: float, h_pct: float):
    """Tính vùng (x, y, w, h) theo pixel từ vị trí/kích thước TỰ DO (%)."""
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
            # ffmpeg giới hạn bán kính làm mờ kênh màu (chroma_radius) phải < 18,
            # trong khi kênh sáng (luma_radius) có thể lớn hơn — nếu dùng chung 1
            # giá trị cho cả hai (như "boxblur=25:2") sẽ bị lỗi khi > 17. Tách
            # riêng và luôn kẹp chroma trong giới hạn hợp lệ.
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

    # libx264 yêu cầu chiều rộng/cao là số chẵn — nhiều video gốc có kích thước
    # lẻ (vd 1919x1079) sẽ làm ffmpeg lỗi ngay lập tức nếu không ép về số chẵn.
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
# TỰ ĐỘNG PHÁT HIỆN VÙNG PHỤ ĐỀ GỐC (không cần OpenCV, chỉ dùng Pillow + numpy)
# ============================================================
#
# Ý tưởng: phụ đề chữ Trung ghi cứng trên video luôn nằm ở CÙNG một vùng
# (thường là dải ngang gần đáy khung hình) trong suốt video, dù nội dung chữ
# đổi liên tục. Vì vậy, nếu lấy mẫu nhiều khung hình rải đều theo thời lượng
# video, tính "năng lượng cạnh" (edge energy - đo độ tương phản/viền chữ)
# theo từng hàng/cột pixel rồi lấy trung bình qua tất cả khung hình mẫu, thì
# vùng có phụ đề sẽ luôn nổi bật hơn hẳn phần còn lại (vì có viền chữ rõ nét
# lặp lại đúng vị trí), trong khi nội dung video thường trôi/thay đổi vị trí
# edge liên tục nên không tích luỹ năng lượng rõ rệt tại một dải cố định.

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
    """Vùng mặc định dùng khi không phát hiện được vị trí rõ ràng — dải dưới
    cùng khung hình, nơi phụ đề Trung thường được đặt."""
    x = int(width * 0.05)
    y = int(height * 0.80)
    w = int(width * 0.90)
    h = int(height * 0.14)
    return x, y, w, h


def detect_subtitle_region(video_path: str, width: int, height: int, tmp_dir: str, sample_count: int = 10):
    """Trả về (x, y, w, h) theo pixel là vùng nhiều khả năng chứa phụ đề gốc
    ghi cứng trên video. Nếu không chắc chắn, trả về vùng mặc định an toàn ở
    dưới khung hình (dùng blur nên kể cả đoán chưa hoàn hảo cũng không phá
    hỏng hình ảnh)."""
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
# TỰ ĐỘNG NHẬN DIỆN GIỌNG NAM / NỮ TRONG VIDEO (theo cao độ - pitch)
# ============================================================
#
# Ý tưởng: dùng ước lượng cao độ cơ bản (F0) bằng phương pháp tự tương quan
# (autocorrelation) trên từng khung nhỏ (~40ms) trong mỗi câu thoại đã được
# faster-whisper cắt sẵn theo thời gian (start/end). Lấy trung vị (median)
# cao độ của các khung có tiếng nói rõ (bỏ khung im lặng) rồi so với ngưỡng
# GENDER_PITCH_THRESHOLD_HZ để phân loại nam/nữ. Đây là cách nhận diện đơn
# giản, không cần thư viện AI riêng biệt, hoạt động tốt với giọng nói đơn lẻ
# rõ ràng nhưng có thể kém chính xác hơn với đoạn nhiều tiếng ồn/nhạc nền.

def load_wav_mono16(path: str):
    """Đọc file wav PCM 16-bit mono đã tách sẵn (từ extract_audio) thành mảng
    numpy float32 trong khoảng [-1, 1] cùng sample rate."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


def _estimate_pitch_autocorr(frame: "np.ndarray", sr: int, fmin: float = 70.0, fmax: float = 300.0):
    """Ước lượng cao độ cơ bản (Hz) của 1 khung nhỏ bằng tự tương quan.
    Trả về None nếu khung quá yên tĩnh hoặc không rõ chu kỳ (không phải giọng nói)."""
    if frame.size == 0:
        return None
    frame = frame - np.mean(frame)
    energy = float(np.sqrt(np.mean(frame ** 2)))
    if energy < 0.01:
        return None  # gần như im lặng, bỏ qua

    windowed = frame * np.hanning(len(frame))
    corr = np.correlate(windowed, windowed, mode="full")
    corr = corr[len(corr) // 2:]
    if corr[0] <= 0:
        return None
    corr = corr / corr[0]

    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    max_lag = min(max_lag, len(corr) - 1)
    if min_lag >= max_lag:
        return None

    window_corr = corr[min_lag:max_lag]
    if window_corr.size == 0:
        return None
    peak_idx = int(np.argmax(window_corr)) + min_lag
    if corr[peak_idx] < 0.3:
        return None  # không đủ tin cậy để coi là có cao độ rõ ràng
    return sr / peak_idx


def estimate_segment_gender(samples: "np.ndarray", sr: int, start: float, end: float,
                             frame_ms: float = 40.0, hop_ms: float = 20.0):
    """Trả về (gender, median_f0) cho 1 khoảng thời gian [start, end) giây,
    hoặc None nếu không đủ dữ liệu tin cậy để nhận diện."""
    start_i = max(0, int(start * sr))
    end_i = min(len(samples), int(end * sr))
    if end_i <= start_i:
        return None

    segment = samples[start_i:end_i]
    frame_len = max(int(sr * frame_ms / 1000), 32)
    hop_len = max(int(sr * hop_ms / 1000), 16)

    pitches = []
    i = 0
    while i + frame_len <= len(segment):
        f0 = _estimate_pitch_autocorr(segment[i:i + frame_len], sr)
        if f0:
            pitches.append(f0)
        i += hop_len

    if not pitches:
        return None

    median_f0 = float(np.median(pitches))
    gender = "female" if median_f0 >= GENDER_PITCH_THRESHOLD_HZ else "male"
    return gender, median_f0


def detect_genders_for_segments(audio_path: str, segments, progress_cb=None):
    """Gán seg['gender'] ('male'/'female') và seg['pitch_hz'] cho từng câu
    thoại trong segments, dựa trên cao độ giọng nói gốc trong audio_path.
    Nếu 1 câu không đủ dữ liệu tin cậy, sẽ dùng giới tính của câu liền trước
    (giúp tránh nhảy giọng liên tục do lỗi nhận diện lẻ tẻ)."""
    samples, sr = load_wav_mono16(audio_path)
    last_gender = "female"

    for i, seg in enumerate(segments):
        result = estimate_segment_gender(samples, sr, seg["start"], seg["end"])
        if result:
            gender, f0 = result
            seg["gender"] = gender
            seg["pitch_hz"] = f0
            last_gender = gender
        else:
            seg["gender"] = last_gender
            seg["pitch_hz"] = None
        if progress_cb:
            progress_cb(i + 1, len(segments))

    return segments


# ============================================================
# CÁC HÀM XEM TRƯỚC (PREVIEW) — chỉ thao tác trên 1 khung hình mẫu, cực nhanh
# ============================================================

def ensure_workdir() -> str:
    if "workdir" not in st.session_state:
        st.session_state.workdir = tempfile.mkdtemp(prefix="vidtrans_")
    return st.session_state.workdir


def get_persistent_video_path(uploaded_file) -> str:
    """Lưu file upload vào thư mục tạm bền vững (giữ được qua các lần Streamlit tự
    rerun khi kéo thanh trượt) để tạo ảnh xem trước ngay lập tức."""
    ensure_workdir()

    file_hash = hashlib.md5(uploaded_file.getbuffer()).hexdigest()[:10]
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


def _frame_extracted_ok(out_path: str) -> bool:
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def extract_preview_frame(video_path: str, out_path: str) -> bool:
    """Trích xuất 1 khung hình xem trước, CỐ GẮNG BẰNG MỌI CÁCH lấy được ảnh
    thay vì bỏ cuộc ngay ở lần thử đầu. Thử lần lượt nhiều mốc thời gian và
    nhiều cách seek khác nhau (một số container/codec chỉ decode được đúng ở
    một vài cách nhất định), chỉ báo thất bại khi TẤT CẢ các cách đều không ra ảnh."""
    try:
        dur = ffprobe_duration(video_path)
    except Exception:
        dur = 0.0

    candidate_timestamps = []
    if dur > 0.2:
        candidate_timestamps.append(max(min(dur * 0.3, dur - 0.1), 0))
        candidate_timestamps.append(max(min(dur * 0.1, dur - 0.1), 0))
        candidate_timestamps.append(max(min(dur * 0.6, dur - 0.1), 0))
    candidate_timestamps.append(0.0)
    # loại bỏ trùng lặp nhưng giữ thứ tự ưu tiên
    seen = set()
    ordered_timestamps = []
    for t in candidate_timestamps:
        key = round(t, 2)
        if key not in seen:
            seen.add(key)
            ordered_timestamps.append(t)

    # Cách 1 (nhanh): seek TRƯỚC -i (-ss trước -i) ở nhiều mốc thời gian.
    for t in ordered_timestamps:
        try:
            cmd = ["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", video_path,
                   "-frames:v", "1", "-q:v", "2", out_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if _frame_extracted_ok(out_path):
                return True
        except Exception:
            pass

    # Cách 2 (chính xác hơn, chậm hơn): seek SAU -i, giúp với container khó decode ngẫu nhiên.
    for t in ordered_timestamps:
        try:
            cmd = ["ffmpeg", "-y", "-i", video_path, "-ss", f"{t:.2f}",
                   "-frames:v", "1", "-q:v", "2", out_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if _frame_extracted_ok(out_path):
                return True
        except Exception:
            pass

    # Cách 3 (chọn khung hình bất kỳ decode được, không cố định thời điểm).
    try:
        cmd = ["ffmpeg", "-y", "-i", video_path, "-frames:v", "1", "-q:v", "2", out_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if _frame_extracted_ok(out_path):
            return True
    except Exception:
        pass

    # Cách 4 (dự phòng cuối): ép decode lại toàn bộ luồng hình ảnh về khung đầu tiên
    # qua bộ lọc select, hữu ích với vài file có timestamp/container bất thường.
    try:
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", "select=eq(n\\,0)",
               "-frames:v", "1", "-q:v", "2", out_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if _frame_extracted_ok(out_path):
            return True
    except Exception:
        pass

    return False


def ensure_preview_frame(video_path: str):
    if st.session_state.get("preview_frame_for") == video_path and st.session_state.get("preview_frame_path"):
        return st.session_state.preview_frame_path, st.session_state.preview_video_res

    frame_path = os.path.join(st.session_state.workdir, "preview_frame.jpg")
    ok = extract_preview_frame(video_path, frame_path)
    if ok:
        try:
            resolution = get_video_resolution(video_path)
        except Exception:
            resolution = None
        st.session_state.preview_frame_for = video_path
        st.session_state.preview_frame_path = frame_path
        st.session_state.preview_video_res = resolution
        return frame_path, resolution

    st.session_state.preview_frame_for = video_path
    st.session_state.preview_frame_path = None
    st.session_state.preview_video_res = None
    return None, None


def ensure_detected_box(video_path: str, width: int, height: int):
    """Chạy phát hiện vùng phụ đề gốc 1 lần cho mỗi video, cache lại trong session_state."""
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
    if alignment == 8:  # trên
        y = scaled_margin
    elif alignment == 5:  # giữa màn hình
        y = (h - text_h) / 2
    else:  # dưới (mặc định)
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


def apply_tempo_style(in_path: str, out_path: str, rate_pct: int):
    """Áp tốc độ (nhanh/chậm) hậu xử lý bằng ffmpeg atempo — dùng cho các
    giọng (như gTTS) không hỗ trợ tham số tốc độ trực tiếp lúc tạo giọng."""
    if rate_pct == 0:
        shutil.copy(in_path, out_path)
        return
    factor = max(0.5, min(2.0, 1.0 + (rate_pct / 100.0)))
    subprocess.run(
        ["ffmpeg", "-y", "-i", in_path, "-filter:a", f"atempo={factor:.3f}", out_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _gtts_synthesize(text: str, lang: str, out_path: str):
    from gtts import gTTS
    gTTS(text=text, lang=lang, slow=False).save(out_path)


def generate_tts_preview_audio(voice_entry: dict, rate_pct: int, pitch_hz: int, out_path: str,
                                sample_text: str = SAMPLE_TTS_TEXT):
    if voice_entry["provider"] == "edge":
        import edge_tts
        rate_str = f"{'+' if rate_pct >= 0 else ''}{rate_pct}%"
        pitch_str = f"{'+' if pitch_hz >= 0 else ''}{pitch_hz}Hz"
        communicate = edge_tts.Communicate(sample_text, voice=voice_entry["voice"],
                                            rate=rate_str, volume="+0%", pitch=pitch_str)
        _run_async(communicate.save(out_path))
    else:  # gtts — không hỗ trợ pitch, chỉ áp tốc độ hậu xử lý
        raw_path = out_path + ".raw.mp3"
        _gtts_synthesize(sample_text, voice_entry["voice"], raw_path)
        apply_tempo_style(raw_path, out_path, rate_pct)


# ============================================================
# CÁC HÀM LỒNG TIẾNG (TTS)
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


def synthesize_all_tts(items, rate_pct: int, pitch_hz: int,
                        max_retries: int = 3, concurrency: int = 6, progress_cb=None):
    """items: list các tuple (idx, text, raw_path, voice_entry) — mỗi câu có thể
    dùng 1 giọng khác nhau (nữ/nam, Microsoft/Google) tuỳ lựa chọn hoặc giới
    tính đã nhận diện được."""
    import edge_tts

    rate_str = f"{'+' if rate_pct >= 0 else ''}{rate_pct}%"
    pitch_str = f"{'+' if pitch_hz >= 0 else ''}{pitch_hz}Hz"

    failed_indices = set()
    done_count = 0
    lock = asyncio.Lock()

    async def _synthesize_one(sem, idx, text, raw_path, voice_entry):
        nonlocal done_count
        async with sem:
            success = False
            loop = asyncio.get_event_loop()
            for attempt in range(max_retries):
                try:
                    if voice_entry["provider"] == "edge":
                        communicate = edge_tts.Communicate(text, voice=voice_entry["voice"],
                                                             rate=rate_str, volume="+0%", pitch=pitch_str)
                        await communicate.save(raw_path)
                    else:  # gtts
                        gtts_raw_path = raw_path + ".gtts_raw.mp3"
                        await loop.run_in_executor(None, _gtts_synthesize, text, voice_entry["voice"], gtts_raw_path)
                        await loop.run_in_executor(None, apply_tempo_style, gtts_raw_path, raw_path, rate_pct)
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
        await asyncio.gather(*[_synthesize_one(sem, idx, text, raw_path, voice_entry)
                                for idx, text, raw_path, voice_entry in items])

    _run_async(_synthesize_all())
    return failed_indices


def fit_audio_to_duration(in_path: str, out_path: str, target_duration: float):
    dur = ffprobe_duration(in_path)
    if dur <= target_duration or target_duration <= 0:
        subprocess.run(["ffmpeg", "-y", "-i", in_path, "-ar", "24000", "-ac", "1", out_path],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    factor = dur / target_duration
    filters = []
    remaining = factor
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.3f}")
    subprocess.run(["ffmpeg", "-y", "-i", in_path, "-filter:a", ",".join(filters), "-ar", "24000", "-ac", "1", out_path],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def make_silence(out_path: str, duration: float):
    duration = max(duration, 0.1)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono:d={duration}", out_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def build_dub_track(segments, rate_pct: int, pitch_hz: int, tmp_dir: str, total_duration: float, *,
                     voice_mode: str, manual_voice: dict = None,
                     female_voice: dict = None, male_voice: dict = None,
                     concurrency: int = 6, progress_cb=None):
    """Lồng tiếng cho toàn bộ video.
    - voice_mode == "manual": TẤT CẢ câu dùng chung 1 giọng do người dùng chọn (manual_voice),
      bất kể giới tính nhận diện được.
    - voice_mode == "auto": mỗi câu dùng giọng nữ/nam tương ứng với seg['gender']
      đã được nhận diện tự động (female_voice / male_voice)."""
    items = []
    for i, seg in enumerate(segments):
        text = seg["translated"].strip()
        if not text:
            continue
        raw_path = os.path.join(tmp_dir, f"tts_raw_{i}.mp3")
        if voice_mode == "manual":
            voice_entry = manual_voice
        else:
            voice_entry = female_voice if seg.get("gender", "female") == "female" else male_voice
        items.append((i, text, raw_path, voice_entry))

    failed_indices = synthesize_all_tts(items, rate_pct, pitch_hz,
                                         concurrency=concurrency, progress_cb=progress_cb)

    seg_files = []
    for idx, text, raw_path, voice_entry in items:
        seg = segments[idx]
        slot_duration = max(seg["end"] - seg["start"], 0.3)
        fit_path = os.path.join(tmp_dir, f"tts_fit_{idx}.wav")
        if idx in failed_indices:
            make_silence(fit_path, slot_duration)
        else:
            fit_audio_to_duration(raw_path, fit_path, slot_duration)
        seg_files.append((seg["start"], fit_path))

    dub_track_path = os.path.join(tmp_dir, "dub_track.wav")
    inputs = ["-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono:d={total_duration}"]
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
# GIAO DIỆN
# ============================================================

st.title("🎬 Dịch & lồng tiếng video Trung → Việt")
st.caption(
    "Chỉ cần tải video lên, chọn cỡ chữ, giọng đọc và màu chữ — mọi bước còn lại "
    "(phát hiện & xoá phụ đề gốc, nhận diện giọng nam/nữ, chỉnh âm lượng...) được xử lý tự động."
)

# ---------- 1) TẢI VIDEO ----------
st.subheader("📥 Tải video lên")
uploaded_file = st.file_uploader("📤 Chọn file video (mp4, mkv, mov, avi)", type=["mp4", "mkv", "mov", "avi"])

active_video_path = None
active_video_name = None
if uploaded_file is not None:
    active_video_path = get_persistent_video_path(uploaded_file)
    active_video_name = uploaded_file.name

preview_frame_path = None
preview_video_res = None
detected_box = None

if active_video_path is not None:
    persistent_video_path = active_video_path
    with st.spinner("Đang trích xuất khung hình xem trước..."):
        preview_frame_path, preview_video_res = ensure_preview_frame(persistent_video_path)
    if preview_frame_path is None:
        st.warning("⚠️ Không trích được khung hình xem trước dù đã thử nhiều cách (file có thể lỗi nặng) — vẫn xử lý được bình thường, chỉ là không có ảnh xem trước.")
    else:
        pw, ph = preview_video_res if preview_video_res else (None, None)
        if pw and ph:
            with st.spinner("🔎 Đang tự động phát hiện vùng phụ đề gốc..."):
                detected_box = ensure_detected_box(persistent_video_path, pw, ph)
else:
    st.info("👆 Hãy tải video lên để bắt đầu — mọi bản xem trước sẽ xuất hiện ngay bên dưới.")

pw, ph = preview_video_res if preview_video_res else (None, None)

# Gợi ý cỡ chữ mặc định theo độ phân giải video (người dùng vẫn chỉnh được)
default_font_size = 20
if ph:
    default_font_size = max(14, min(48, int(ph / 18)))

# ---------- 2) 3 THÔNG SỐ NGƯỜI DÙNG CẦN CHỌN ----------
st.divider()
st.subheader("🎛️ Chỉ 3 điều bạn cần chọn")

c1, c2 = st.columns(2)
with c1:
    font_size = st.slider("🔤 Cỡ chữ phụ đề", 10, 60, default_font_size, step=1)
with c2:
    text_color = st.color_picker("🎨 Màu chữ phụ đề", "#FFFFFF")

st.markdown("**🎙️ Giọng đọc lồng tiếng AI**")
voice_mode_label = st.radio(
    "Chế độ chọn giọng",
    ["🤖 Tự động theo giới tính giọng gốc (khuyến nghị)", "🖐️ Tự chọn 1 giọng cố định cho cả video"],
    label_visibility="collapsed",
)
voice_mode = "auto" if voice_mode_label.startswith("🤖") else "manual"

selected_female_voice = None
selected_male_voice = None
selected_manual_voice = None

if voice_mode == "auto":
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        female_label = st.selectbox("Giọng dùng khi nhận diện là NỮ", [v["label"] for v in VOICE_CATALOG["female"]])
        selected_female_voice = VOICE_BY_LABEL[female_label]
    with vcol2:
        male_label = st.selectbox("Giọng dùng khi nhận diện là NAM", [v["label"] for v in VOICE_CATALOG["male"]])
        selected_male_voice = VOICE_BY_LABEL[male_label]
    st.caption("🧠 Giọng nam/nữ tương ứng sẽ được **tự động áp dụng** theo giọng nói gốc nhận diện được trong từng câu thoại.")
else:
    manual_label = st.selectbox("Chọn 1 giọng đọc cho toàn bộ video", [v["label"] for v in ALL_VOICES])
    selected_manual_voice = VOICE_BY_LABEL[manual_label]
    st.caption("🖐️ Toàn bộ video sẽ dùng đúng 1 giọng này, không phân biệt câu thoại gốc là giọng nam hay nữ.")

style_label = st.selectbox("Phong cách đọc (tốc độ/cao độ)", list(STYLE_OPTIONS.keys()))
style_cfg = STYLE_OPTIONS[style_label]
_uses_gtts = (
    (voice_mode == "auto" and (selected_female_voice["provider"] == "gtts" or selected_male_voice["provider"] == "gtts"))
    or (voice_mode == "manual" and selected_manual_voice["provider"] == "gtts")
)
if _uses_gtts:
    st.caption("ℹ️ Giọng Google (gTTS) không hỗ trợ chỉnh cao độ — chỉ áp dụng được phần tốc độ của phong cách đã chọn.")

# Các giá trị tự động suy ra từ 3 lựa chọn trên
outline_color = auto_outline_color(text_color)

# ---------- 3) TUỲ CHỈNH NÂNG CAO (KHÔNG BẮT BUỘC) ----------
with st.expander("⚙️ Tuỳ chỉnh nâng cao (không bắt buộc — để trống thì mọi thứ đã tự động tối ưu)"):
    st.caption("Chỉ mở phần này nếu bạn muốn tự tay ghi đè lên các lựa chọn tự động.")

    adv_burn_sub = st.checkbox("Ghi phụ đề chữ lên video", value=True)
    adv_bilingual = st.checkbox("Hiện song ngữ (Trung + Việt) trong file .srt", value=False)
    adv_position_label = st.selectbox("Vị trí phụ đề mới", list(POSITION_OPTIONS.keys()), index=0)
    adv_margin_v = st.slider("Cách mép (px)", 0, 150, int(25 * ((ph or 360) / 360)), step=5)

    st.markdown("---")
    adv_manual_old_box = st.checkbox("Tự chỉnh tay vùng che phụ đề gốc (thay vì để tự động phát hiện)", value=False)
    adv_disable_remove_old = st.checkbox("KHÔNG xoá/che phụ đề gốc", value=False)
    adv_old_method_label = st.selectbox(
        "Cách che phụ đề gốc",
        ["Làm mờ (blur) - mặc định tự động", "Che bằng khung màu đặc", "Làm mượt tự nhiên (delogo)"],
        index=0,
    )
    if "delogo" in adv_old_method_label:
        adv_old_method_key = "delogo"
    elif "khung màu" in adv_old_method_label:
        adv_old_method_key = "solid"
    else:
        adv_old_method_key = "blur"

    manual_box_pct = None
    if adv_manual_old_box:
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.caption("⬆️ Trên / ⏺️ Giữa / ⬇️ Dưới")
        old_x_pct = st.slider("Vị trí ngang (%)", 0, 100, 5, step=1, key="adv_old_x_pct")
        old_y_pct = st.slider("Vị trí dọc (%)", 0, 100, 80, step=1, key="adv_old_y_pct")
        old_w_pct = st.slider("Chiều rộng vùng che (%)", 4, 100, 90, step=1, key="adv_old_w_pct")
        old_h_pct = st.slider("Chiều cao vùng che (%)", 3, 60, 14, step=1, key="adv_old_h_pct")
        manual_box_pct = (old_x_pct, old_y_pct, old_w_pct, old_h_pct)

    adv_box_color = st.color_picker("Màu khung che (chỉ dùng cho 'khung màu đặc')", "#000000",
                                     disabled=adv_old_method_key != "solid")
    adv_blur_strength = st.slider("Độ mờ (chỉ dùng cho 'làm mờ')", 5, 50, 25, disabled=adv_old_method_key != "blur")

    st.markdown("---")
    adv_disable_dub = st.checkbox("KHÔNG lồng tiếng AI (chỉ dịch phụ đề chữ)", value=False)
    adv_keep_bg = st.checkbox("Giữ tiếng nền/nhạc nền gốc phía sau giọng đọc AI", value=True)
    adv_override_bg_volume = st.checkbox("Tự chỉnh tay âm lượng nền gốc (thay vì để tự động ở mức thấp nhất)", value=False)
    adv_bg_volume_pct = st.slider("Âm lượng nền gốc (%)", 0, 100, AUTO_BG_VOLUME_PCT, step=1,
                                   disabled=not adv_override_bg_volume)
    adv_override_gender_threshold = st.checkbox("Tự chỉnh tay ngưỡng nhận diện nam/nữ (thay vì mặc định 165Hz)", value=False,
                                                 disabled=voice_mode != "auto")
    adv_gender_threshold = st.slider("Ngưỡng cao độ phân biệt nam/nữ (Hz)", 100, 220, int(GENDER_PITCH_THRESHOLD_HZ),
                                      step=5, disabled=(not adv_override_gender_threshold) or voice_mode != "auto",
                                      help="Cao độ đo được CAO HƠN ngưỡng này sẽ được coi là giọng NỮ, THẤP HƠN là giọng NAM.")

    st.markdown("---")
    adv_model_size = st.selectbox("Model nhận diện giọng nói", ["tiny", "base", "small", "medium", "large-v3"], index=1)
    adv_speed_label = st.select_slider("Tốc độ xử lý", options=list(SPEED_PRESETS.keys()),
                                        value="⚖️ Cân bằng (khuyến nghị)")

# ---------- Gộp các giá trị: tự động, trừ khi người dùng ghi đè ở phần nâng cao ----------
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
tts_rate_pct = style_cfg["rate"]
tts_pitch_hz = style_cfg["pitch"]
if voice_mode == "auto" and adv_override_gender_threshold:
    GENDER_PITCH_THRESHOLD_HZ = adv_gender_threshold  # cho phép ghi đè ngưỡng nếu người dùng tự chỉnh

if enable_dub:
    original_volume_pct = adv_bg_volume_pct if adv_override_bg_volume else AUTO_BG_VOLUME_PCT
else:
    original_volume_pct = AUTO_NO_DUB_VOLUME_PCT

# Vùng che phụ đề gốc: ưu tiên tuỳ chỉnh tay nếu bật, không thì dùng vùng tự động phát hiện
old_sub_box = None
if remove_old_sub and pw and ph:
    if adv_manual_old_box and manual_box_pct:
        old_sub_box = compute_free_box(pw, ph, *manual_box_pct)
    elif detected_box:
        old_sub_box = detected_box

# ---------- 4) KHỐI XEM TRƯỚC DUY NHẤT ----------
st.divider()
st.subheader("🖼️ Xem trước")

if active_video_path is None:
    st.info("Tải video lên ở trên để xem bản xem trước.")
elif preview_frame_path is None:
    st.warning("Không có khung hình xem trước cho video này — vẫn xử lý được, chỉ là không kiểm tra trước được vị trí/màu sắc.")
else:
    combined_img = Image.open(preview_frame_path).convert("RGB")
    if remove_old_sub and old_sub_box is not None:
        combined_img = render_old_sub_box_preview(combined_img, old_sub_box, old_sub_method_key, box_color, blur_strength)
    if burn_sub:
        combined_img = render_subtitle_style_preview(
            combined_img, SAMPLE_SUBTITLE_TEXT, font_size, text_color, outline_color,
            POSITION_OPTIONS[position_label], margin_v,
        )
    st.image(combined_img, use_container_width=True)

    legend_bits = []
    if remove_old_sub:
        legend_bits.append("🟥 khung đỏ = vùng phụ đề gốc được **tự động phát hiện** và sẽ bị làm mờ/che")
    if burn_sub:
        legend_bits.append("🔤 chữ mẫu = kiểu phụ đề mới sẽ ghi lên video")
    if legend_bits:
        st.caption(" · ".join(legend_bits) + ". Đây là hình mô phỏng gần đúng, kết quả thật có thể chênh lệch nhẹ.")
    if remove_old_sub and not adv_manual_old_box:
        st.caption("💡 Nếu khung đỏ chưa khớp chính xác vùng phụ đề gốc, mở '⚙️ Tuỳ chỉnh nâng cao' để tự kéo tay.")

    if enable_dub:
        st.markdown("**🔊 Nghe thử giọng đọc**")
        if voice_mode == "auto":
            preview_voices = [("female", selected_female_voice), ("male", selected_male_voice)]
        else:
            preview_voices = [("manual", selected_manual_voice)]

        pcols = st.columns(len(preview_voices))
        for pcol, (key, voice_entry) in zip(pcols, preview_voices):
            with pcol:
                btn_key = f"btn_tts_preview_{key}_{voice_entry['id']}"
                if st.button(f"▶️ {voice_entry['label']}", key=btn_key):
                    with st.spinner("Đang tạo giọng đọc mẫu..."):
                        try:
                            p = os.path.join(st.session_state.workdir, f"tts_preview_{key}.mp3")
                            generate_tts_preview_audio(voice_entry, tts_rate_pct, tts_pitch_hz, p)
                            with open(p, "rb") as f:
                                st.session_state[f"tts_preview_bytes_{key}"] = f.read()
                        except Exception as e:
                            st.error(f"Không tạo được giọng đọc mẫu: {e}")
                            st.session_state[f"tts_preview_bytes_{key}"] = None
                if st.session_state.get(f"tts_preview_bytes_{key}"):
                    st.audio(st.session_state[f"tts_preview_bytes_{key}"], format="audio/mp3")

# ---------- 5) BẮT ĐẦU XỬ LÝ ----------
st.divider()
start_button = st.button("🚀 Bắt đầu xử lý", type="primary", disabled=active_video_path is None)

if start_button:
    if not burn_sub and not enable_dub and not remove_old_sub:
        st.warning("Bạn cần bật ít nhất một trong: ghi phụ đề chữ, xoá phụ đề gốc, hoặc lồng tiếng AI (xem phần Tuỳ chỉnh nâng cao).")
        st.stop()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ext = os.path.splitext(active_video_name or active_video_path)[1] or ".mp4"
        input_path = os.path.join(tmp_dir, "input" + ext)
        shutil.copy(active_video_path, input_path)

        audio_path = os.path.join(tmp_dir, "audio.wav")
        srt_path = os.path.join(tmp_dir, "output.srt")

        status = st.status("Đang xử lý...", expanded=True)

        try:
            status.write("🔊 Đang tách âm thanh...")
            extract_audio(input_path, audio_path)
            total_duration = ffprobe_duration(input_path)

            status.write("🗣️ Đang nhận diện giọng nói tiếng Trung...")
            transcribe_placeholder = st.empty()
            segments = transcribe_audio(
                audio_path, model_size, beam_size=speed_cfg["beam_size"],
                progress_cb=lambda t, text: transcribe_placeholder.write(f"  [{t:.1f}s] {text}")
            )

            if not segments:
                status.update(label="Không nhận diện được giọng nói nào.", state="error")
                st.stop()
            status.write(f"✅ Nhận diện xong {len(segments)} câu.")

            if enable_dub and voice_mode == "auto":
                status.write("🧑‍🤝‍🧑 Đang tự động nhận diện giọng nam/nữ trong video...")
                gender_bar = st.progress(0)
                segments = detect_genders_for_segments(
                    audio_path, segments,
                    progress_cb=lambda d, t: gender_bar.progress(d / t),
                )
                n_female = sum(1 for s in segments if s.get("gender") == "female")
                n_male = len(segments) - n_female
                status.write(f"✅ Đã nhận diện: {n_female} câu giọng nữ, {n_male} câu giọng nam.")

            status.write("🌐 Đang dịch sang tiếng Việt (theo batch)...")
            translate_bar = st.progress(0)
            segments = translate_segments(segments, progress_cb=lambda d, t: translate_bar.progress(d / t))

            srt_content = build_srt(segments, bilingual=bilingual)
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

            original_volume_factor = original_volume_pct / 100.0
            current_video = input_path

            if remove_old_sub or burn_sub:
                status.write("🎞️ Đang xử lý hình ảnh video (xoá phụ đề gốc tự động phát hiện / ghi phụ đề mới)...")
                final_old_sub_box = None
                if remove_old_sub:
                    width, height = get_video_resolution(current_video)
                    if adv_manual_old_box and manual_box_pct:
                        final_old_sub_box = compute_free_box(width, height, *manual_box_pct)
                    else:
                        final_old_sub_box = detect_subtitle_region(current_video, width, height, tmp_dir)
                    # Đảm bảo vùng che luôn nằm gọn trong khung hình, tránh lỗi crop tràn biên.
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
                status.write(f"🎙️ Đang tạo giọng đọc AI (phong cách: {style_label})...")
                tts_bar = st.progress(0)

                dub_track_path, failed_indices = build_dub_track(
                    segments, tts_rate_pct, tts_pitch_hz, tmp_dir, total_duration,
                    voice_mode=voice_mode,
                    manual_voice=selected_manual_voice,
                    female_voice=selected_female_voice,
                    male_voice=selected_male_voice,
                    concurrency=speed_cfg["tts_concurrency"],
                    progress_cb=lambda d, t: tts_bar.progress(d / t)
                )
                dub_warning_count = len(failed_indices)

                status.write("🔀 Đang ghép giọng lồng tiếng vào video và tự động hạ âm lượng gốc...")
                dub_output = os.path.join(tmp_dir, "with_dub.mp4")
                combine_video_with_dub(current_video, dub_track_path, dub_output,
                                        keep_bg=keep_bg, bg_volume=original_volume_factor)
                current_video = dub_output
            elif original_volume_factor != 1.0:
                status.write("🔊 Đang điều chỉnh âm lượng video gốc...")
                vol_output = os.path.join(tmp_dir, "with_volume.mp4")
                adjust_video_volume(current_video, vol_output, original_volume_factor)
                current_video = vol_output

            status.update(label="Hoàn tất!", state="complete")
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

        st.success("✅ Xử lý xong!")
        st.subheader("👀 Kết quả cuối cùng")
        st.video(video_bytes)

        colA, colB = st.columns(2)
        with colA:
            st.download_button("⬇️ Tải video kết quả", data=video_bytes,
                                file_name="video_ket_qua.mp4", mime="video/mp4")
        with colB:
            st.download_button("⬇️ Tải file phụ đề (.srt)", data=srt_content,
                                file_name="phu_de.srt", mime="text/plain")

        with st.expander("📜 Xem nội dung phụ đề"):
            for seg in segments:
                gender_bit = ""
                if enable_dub and voice_mode == "auto" and seg.get("gender"):
                    gender_bit = f" · {GENDER_LABEL.get(seg['gender'], '')}"
                    if seg.get("pitch_hz"):
                        gender_bit += f" (~{seg['pitch_hz']:.0f}Hz)"
                st.write(f"**[{seg['start']:.1f}s - {seg['end']:.1f}s]{gender_bit}**")
                if bilingual:
                    st.write(f"🇨🇳 {seg['text']}")
                st.write(f"🇻🇳 {seg['translated']}")
                st.divider()
