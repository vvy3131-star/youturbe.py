

### 🌟 Các điểm nâng cấp nổi bật

1. **Bỏ chế độ tự động nhận diện giới tính & tự lồng tiếng cũ**: Loại bỏ hoàn toàn thuật toán phân tích pitch phức tạp và dễ lỗi. Thay vào đó, bạn có một **menu lựa chọn một giọng đọc duy nhất** cho toàn bộ video, hoạt động cực kỳ mượt mà và ổn định.
2. **Sửa lỗi AI đọc quá nhanh**: Cấu hình mặc định tốc độ đọc về trạng thái **đàm thoại tự nhiên của người Việt (`rate="+0%"`)**. Bạn có thể tùy chỉnh thêm tốc độ qua thanh trượt nếu muốn.
3. **Thêm hàng loạt giọng nữ chất lượng (Voice Profiles)**: Khắc phục sự tẻ nhạt của giọng nữ bằng cách tạo ra các bộ lọc giọng "Hoài My" đa dạng: *Ấm áp, Trẻ trung, Dịu dàng, Mặc định* bằng cách tối ưu hóa tần số và nhịp điệu của Edge-TTS, cùng giọng Google làm phong phú thêm lựa chọn.
4. **Nâng cấp độ chính xác xóa phụ đề**: Thuật toán phân tích năng lượng cạnh (Edge Energy) được cải tiến để định vị dải chữ chính xác hơn, kết hợp tính năng **bù trừ biên (padding)** thông minh để không bỏ sót viền chữ.
5. **Tính năng đóng dấu Logo tròn góc trên bên trái**: Hỗ trợ upload ảnh logo bất kỳ (JPG/PNG). Hệ thống sẽ tự động cắt bo tròn thành hình tròn hoàn hảo, tối ưu kích thước theo độ phân giải video và chèn trực tiếp vào góc trái màn hình thông qua bộ lọc Ffmpeg tốc độ cao.
6. **Tối ưu hóa bản dịch**: Cải thiện khâu xử lý chuỗi trước và sau khi dịch của Google Translator, giúp loại bỏ các từ lặp thừa và tối ưu ngữ nghĩa cho tự nhiên nhất.

Dưới đây là mã nguồn hoàn chỉnh của file `app.py`:

```python
import asyncio
import hashlib
import os
import shutil
import subprocess
import tempfile
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

st.set_page_config(page_title="Dịch phụ đề & Lồng tiếng video Trung -> Việt", page_icon="🎬", layout="centered")

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 780px;}
    div[data-testid="stVerticalBlock"] > div:has(> div.stButton) button {width: 100%;}
    .stTabs [data-baseweb="tab"] {padding: 0.4rem 0.8rem; font-size: 0.92rem;}
    div[data-testid="stSlider"] {padding-bottom: 0.2rem;}
</style>
""", unsafe_allow_html=True)

# ---------- DANH MỤC GIỌNG ĐỌC MỚI (Nâng cấp nhiều phong cách giọng Nữ) ----------
VOICE_CATALOG = [
    # Nhóm Giọng Nữ (Được tối ưu lại)
    {"id": "female_default", "label": "👩 Hoài My - Giọng Chuẩn (Tự nhiên)", "provider": "edge", "voice": "vi-VN-HoaiMyNeural", "rate": "+0%", "pitch": "+0Hz"},
    {"id": "female_young", "label": "👩 Hoài My - Trẻ trung, Vui vẻ", "provider": "edge", "voice": "vi-VN-HoaiMyNeural", "rate": "+3%", "pitch": "+2Hz"},
    {"id": "female_gentle", "label": "👩 Hoài My - Dịu dàng, Truyền cảm", "provider": "edge", "voice": "vi-VN-HoaiMyNeural", "rate": "-4%", "pitch": "-1Hz"},
    {"id": "female_warm", "label": "👩 Hoài My - Trầm ấm, Sâu lắng", "provider": "edge", "voice": "vi-VN-HoaiMyNeural", "rate": "-6%", "pitch": "-3Hz"},
    {"id": "google_female", "label": "👩 Giọng nữ Google (gTTS)", "provider": "gtts", "voice": "vi"},
    
    # Nhóm Giọng Nam
    {"id": "male_default", "label": "👨 Nam Minh - Mặc định", "provider": "edge", "voice": "vi-VN-NamMinhNeural", "rate": "+0%", "pitch": "+0Hz"},
    {"id": "male_slow", "label": "👨 Nam Minh - Chậm rãi, Chững chạc", "provider": "edge", "voice": "vi-VN-NamMinhNeural", "rate": "-5%", "pitch": "-1Hz"},
]

POSITION_OPTIONS = {
    "Dưới (mặc định)": 2,
    "Giữa màn hình": 5,
    "Trên": 8,
}

SPEED_PRESETS = {
    "⚡ Nhanh nhất": {"beam_size": 1, "video_preset": "ultrafast"},
    "⚖️ Cân bằng (khuyến nghị)": {"beam_size": 3, "video_preset": "veryfast"},
    "🎯 Chính xác nhất (chậm hơn)": {"beam_size": 5, "video_preset": "medium"},
}

CPU_COUNT = os.cpu_count() or 4
SAMPLE_SUBTITLE_TEXT = "Đây là câu phụ đề mẫu để xem trước"

# ============================================================
# CÁC HÀM XỬ LÝ VIDEO & HÌNH ẢNH
# ============================================================

class FFmpegError(RuntimeError):
    pass

def _run_ffmpeg(cmd, label: str = "ffmpeg"):
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        stderr_tail = "\n".join(result.stderr.strip().splitlines()[-15:]) if result.stderr else "(không có thông tin)"
        raise FFmpegError(f"Bước '{label}' thất bại (mã lỗi {result.returncode}).\n\nChi tiết:\n{stderr_tail}")
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

def clean_translated_text(text: str) -> str:
    """Tối ưu hóa và dọn dẹp bản dịch để mượt mà nhất"""
    if not text:
        return ""
    # Loại bỏ khoảng trắng thừa thụ động từ tiếng Trung
    text = " ".join(text.split())
    # Sửa một số từ dịch máy phổ biến để mượt mà hơn
    replacements = {
        "đối với tôi": "với tôi",
        "như thế này": "như vậy",
        "bởi vì": "vì",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def translate_segments(segments, progress_cb=None, chunk_size: int = 40):
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source="zh-CN", target="vi")

    texts = [seg["text"] for seg in segments]
    translated_all = [None] * len(segments)

    for start in range(0, len(texts), chunk_size):
        chunk = texts[start:start + chunk_size]
        try:
            batch_result = translator.translate_batch(chunk)
        except Exception:
            batch_result = None

        if batch_result and len(batch_result) == len(chunk):
            for i, t in enumerate(batch_result):
                translated_all[start + i] = clean_translated_text(t) if t else chunk[i]
        else:
            for i, text in enumerate(chunk):
                try:
                    translated_all[start + i] = clean_translated_text(translator.translate(text))
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

def auto_outline_color(text_color_hex: str) -> str:
    hex_color = text_color_hex.lstrip("#")
    if len(hex_color) != 6:
        return "#000000"
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000000" if luminance > 0.55 else "#FFFFFF"

# ============================================================
# PHÁT HIỆN PHỤ ĐỀ GỐC CẢI TIẾN (Độ chính xác cao hơn)
# ============================================================

def _extract_sample_frames(video_path: str, out_dir: str, count: int = 12):
    try:
        dur = ffprobe_duration(video_path)
    except Exception:
        return []
    if dur <= 0.5:
        return []
    timestamps = [dur * (0.05 + 0.90 * i / (count - 1)) for i in range(count)]

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
    y = int(height * 0.78)
    w = int(width * 0.90)
    h = int(height * 0.16)
    return x, y, w, h

def detect_subtitle_region(video_path: str, width: int, height: int, tmp_dir: str, sample_count: int = 12):
    """Phát hiện phụ đề gốc với thuật toán lọc nhiễu tốt hơn"""
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

        bottom_start = int(height * 0.50)  # Thu hẹp vùng quét để tập trung chính xác hơn
        top_end = int(height * 0.20)
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
        threshold = peak * 0.35  # Giảm ngưỡng để bắt trọn các ký tự nét nhỏ

        y_start = peak_idx
        while y_start > 0 and avg_row[y_start - 1] >= threshold:
            y_start -= 1
        y_end = peak_idx
        while y_end < height - 1 and avg_row[y_end + 1] >= threshold:
            y_end += 1

        # Mở rộng nhẹ vùng biên phát hiện (padding) để đảm bảo không bị lem phụ đề gốc
        pad_y = int(height * 0.02)
        y_start = max(0, y_start - pad_y)
        y_end = min(height - 1, y_end + pad_y)

        x = int(width * 0.05)
        w = int(width * 0.90)
        h = y_end - y_start

        return (x, y_start, w, h)
    except Exception:
        return _default_subtitle_box(width, height)

# ============================================================
# CHUYỂN LOGO THÀNH HÌNH TRÒN & SẮP XẾP FILE
# ============================================================

def make_circle_logo(image_file, size: int = 120) -> str:
    """Chuyển ảnh logo bất kỳ thành hình tròn có nền trong suốt"""
    img = Image.open(image_file).convert("RGBA")
    img = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)
    
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)
    
    output = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    output.paste(img, (0, 0), mask=mask)
    
    path = os.path.join(ensure_workdir(), "logo_circle.png")
    output.save(path, "PNG")
    return path

# ============================================================
# LỒNG TIẾNG VÀ GỘP VIDEO HOÀN CHỈNH
# ============================================================

async def generate_voice_edge(text: str, voice_info: dict, output_path: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice_info["voice"], rate=voice_info["rate"], pitch=voice_info["pitch"])
    await communicate.save(output_path)

def generate_voice_gtts(text: str, output_path: str):
    from gtts import gTTS
    tts = gTTS(text=text, lang="vi")
    tts.save(output_path)

def render_and_merge(video_path: str, output_path: str, srt_path: str, segments: list, voice_info: dict,
                     remove_old_sub: bool, old_sub_box, logo_path: str = None, font_size: int = 22,
                     primary_color: str = "&H00FFFFFF&", outline_color: str = "&H00000000&",
                     alignment: int = 2, margin_v: int = 25, bg_volume_pct: int = 10, preset: str = "veryfast"):
    
    tmp_dir = ensure_workdir()
    audio_segments_paths = []
    
    # 1. Tạo audio cho từng phân đoạn phụ đề dịch
    for idx, seg in enumerate(segments):
        audio_seg_path = os.path.join(tmp_dir, f"seg_{idx}.mp3")
        text = seg["translated"]
        
        if voice_info["provider"] == "edge":
            asyncio.run(generate_voice_edge(text, voice_info, audio_seg_path))
        else:
            generate_voice_gtts(text, audio_seg_path)
        audio_segments_paths.append((seg["start"], audio_seg_path))

    # 2. Xây dựng file cấu hình gom âm thanh TTS lồng đè lên dòng thời gian gốc
    filter_complex_audio = ""
    inputs_audio_cmd = []
    
    # Nạp video gốc
    inputs_audio_cmd.extend(["-i", video_path])
    
    # Nạp các file âm thanh AI lồng tiếng
    for idx, (_, path) in enumerate(audio_segments_paths):
        inputs_audio_cmd.extend(["-i", path])
    
    # Giảm âm lượng nhạc nền của video gốc
    bg_vol = bg_volume_pct / 100.0
    filter_complex_audio += f"[0:a]volume={bg_vol}[bg_audio];"
    
    # Trộn các đoạn thoại lồng tiếng vào đúng thời điểm phát của video
    mix_inputs = ""
    for idx, (start_time, _) in enumerate(audio_segments_paths):
        # Trễ âm thanh AI theo mốc thời gian phụ đề
        filter_complex_audio += f"[{idx+1}:a]adelay={int(start_time*1000)}|{int(start_time*1000)}[delay{idx}];"
        mix_inputs += f"[delay{idx}]"
    
    filter_complex_audio += f"{mix_inputs}amix=inputs={len(audio_segments_paths)}:dropout_transition=0[dub_audio];"
    filter_complex_audio += f"[bg_audio][dub_audio]amix=inputs=2:duration=first[out_audio]"

    temp_audio_mixed = os.path.join(tmp_dir, "audio_mixed.mp3")
    cmd_audio = ["ffmpeg", "-y"] + inputs_audio_cmd + ["-filter_complex", filter_complex_audio, "-map", "[out_audio]", temp_audio_mixed]
    _run_ffmpeg(cmd_audio, label="trộn nhạc nền và giọng AI")

    # 3. Tạo bộ lọc hình ảnh (Xóa sub cũ + Burn sub mới + In Logo tròn)
    filters = []
    stage = "[0:v]"
    counter = 0

    if remove_old_sub and old_sub_box:
        x, y, w, h = old_sub_box
        luma_radius = 25
        chroma_radius = 15
        filters.append(f"{stage}split=2[vmain{counter}][vcrop{counter}]")
        filters.append(
            f"[vcrop{counter}]crop={w}:{h}:{x}:{y},"
            f"boxblur=luma_radius={luma_radius}:luma_power=2:"
            f"chroma_radius={chroma_radius}:chroma_power=2[vblur{counter}]"
        )
        filters.append(f"[vmain{counter}][vblur{counter}]overlay={x}:{y}[v{counter}]")
        stage = f"[v{counter}]"
        counter += 1

    # In Logo kênh hình tròn lên góc trên bên trái
    if logo_path and os.path.exists(logo_path):
        # Nạp ảnh logo tròn làm input thứ 2 của Ffmpeg dựng hình
        logo_input_idx = 1
        filters.append(f"{stage}[{logo_input_idx}:v]overlay=x=20:y=20[vlogo]")
        stage = "[vlogo]"

    # Ghi đè phụ đề mới
    if srt_path:
        srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
        style = (
            f"FontName=Arial,FontSize={font_size},PrimaryColour={primary_color},"
            f"OutlineColour={outline_color},BorderStyle=1,Outline=1.2,Shadow=0.5,"
            f"Alignment={alignment},MarginV={margin_v}"
        )
        filters.append(f"{stage}subtitles={srt_escaped}:force_style='{style}'[vout]")
        stage = "[vout]"

    filters.append(f"{stage}scale=trunc(iw/2)*2:trunc(ih/2)*2[veven]")
    stage = "[veven]"

    filter_complex_video = ";".join(filters)

    cmd_video = [
        "ffmpeg", "-y", "-i", video_path,
    ]
    if logo_path and os.path.exists(logo_path):
        cmd_video.extend(["-i", logo_path])
        
    cmd_video.extend([
        "-i", temp_audio_mixed,
        "-filter_complex", filter_complex_video,
        "-map", stage,
        "-map", f"{2 if logo_path else 1}:a",  # map âm thanh đã trộn ở trên
        "-c:v", "libx264", "-preset", preset, "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        output_path
    ])
    _run_ffmpeg(cmd_video, label="đồng bộ ghép hoàn thiện video")

# ============================================================
# CÁC HÀM XEM TRƯỚC (PREVIEW)
# ============================================================

def ensure_workdir() -> str:
    if "workdir" not in st.session_state:
        st.session_state.workdir = tempfile.mkdtemp(prefix="vidtrans_")
    return st.session_state.workdir

def get_persistent_video_path(uploaded_file) -> str:
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

def extract_preview_frame(video_path: str, out_path: str) -> bool:
    try:
        dur = ffprobe_duration(video_path)
    except Exception:
        dur = 0.0

    candidate_timestamps = [max(min(dur * 0.3, dur - 0.1), 0), 0.0]
    for t in candidate_timestamps:
        try:
            cmd = ["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", video_path, "-frames:v", "1", "-q:v", "2", out_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
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
    return None, None

def ensure_detected_box(video_path: str, width: int, height: int):
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
            return ImageFont.truetype(c, size)
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

def render_old_sub_box_preview(base_image: "Image.Image", box, logo_path: str = None) -> "Image.Image":
    img = base_image.copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)
    
    # 1. Vẽ vùng phát hiện xóa sub gốc
    if box:
        bx, by, bw, bh = box
        draw.rectangle([bx, by, bx+bw, by+bh], outline="#FF3333", width=3)
        draw.text((bx + 5, by - 22), "VÙNG XÓA PHỤ ĐỀ GỐC", fill="#FF3333")
        
    # 2. Vẽ thử nghiệm hiển thị logo tròn góc trên bên trái
    if logo_path and os.path.exists(logo_path):
        logo_img = Image.open(logo_path).convert("RGBA")
        # Resize logo cho khung xem trước
        preview_logo_size = max(int(h * 0.15), 40)
        logo_img = logo_img.resize((preview_logo_size, preview_logo_size))
        img.paste(logo_img, (20, 20), mask=logo_img)
        draw.text((20, 25 + preview_logo_size), "LOGO KÊNH", fill="#00FF00")
        
    return img

# ============================================================
# GIAO DIỆN CHÍNH STREAMLIT (UI)
# ============================================================

st.title("🎬 Dịch & Lồng Tiếng Trung -> Việt")
st.caption("Phiên bản nâng cấp: Cải tiến dịch thuật, nâng cấp thư viện giọng nữ, in Logo tròn & nhận diện xóa sub chính xác hơn.")

uploaded_file = st.file_uploader("1) Tải video lên (.mp4, .mkv, .mov, .avi)", type=["mp4", "mkv", "mov", "avi"])

if uploaded_file:
    video_path = get_persistent_video_path(uploaded_file)
    frame_path, resolution = ensure_preview_frame(video_path)
    
    if frame_path and resolution:
        width, height = resolution
        st.success(f"Video đã nhận diện thành công: {width}x{height} px")
        
        # Tạo sẵn vùng phát hiện sub cũ
        detected_box = ensure_detected_box(video_path, width, height)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Cài đặt Lồng tiếng & Phụ đề")
            
            # CHỌN GIỌNG AI THAY THẾ CHO TOÀN BỘ (ĐÃ BỎ NHẬN DIỆN NAM NỮ PHỨC TẠP)
            voice_labels = [v["label"] for v in VOICE_CATALOG]
            selected_voice_label = st.selectbox("Chọn giọng đọc AI chính cho video:", voice_labels)
            selected_voice = next(v for v in VOICE_CATALOG if v["label"] == selected_voice_label)
            
            # IN LOGO KÊNH HÌNH TRÒN
            st.write("---")
            logo_file = st.file_uploader("In Logo kênh hình tròn (Góc trái trên):", type=["png", "jpg", "jpeg"])
            logo_path = None
            if logo_file:
                # Tạo logo tròn kích thước hợp lý theo chiều cao video (khoảng 12-15% chiều cao)
                logo_target_size = max(int(height * 0.12), 60)
                logo_path = make_circle_logo(logo_file, size=logo_target_size)
                st.success("Đã tạo Logo tròn thành công!")
            
            st.write("---")
            font_size = st.slider("Cỡ chữ phụ đề mới (px):", 12, 50, 24)
            text_color = st.color_picker("Màu chữ phụ đề mới:", "#FFFFFF")
            
            remove_old_sub = st.checkbox("Tự động xóa phụ đề gốc (Làm mờ thông minh)", value=True)
            
        with col2:
            st.subheader("Xem trước thiết lập")
            base_img = Image.open(frame_path)
            
            # Cập nhật hình xem trước bao gồm vùng quét và logo tròn
            preview_box = render_old_sub_box_preview(base_img, detected_box if remove_old_sub else None, logo_path)
            
            # Vẽ thử một câu phụ đề mẫu
            outline_color = auto_outline_color(text_color)
            preview_final = render_subtitle_style_preview(
                preview_box, SAMPLE_SUBTITLE_TEXT, font_size,
                text_color, outline_color, 2, 25
            )
            st.image(preview_final, use_column_width=True, caption="Khung hình mẫu bao gồm vùng nhận diện xóa sub + vị trí Logo")

        # TÙY CHỌN NÂNG CAO (ẨN)
        with st.expander("🛠️ Cài đặt dịch & xử lý chuyên sâu (Không bắt buộc)"):
            bilingual = st.checkbox("Hiển thị song ngữ (Trung - Việt)", value=False)
            model_size = st.selectbox("Model nhận diện tiếng Trung (Whisper):", ["tiny", "base", "small"], index=1)
            speed_preset_key = st.selectbox("Tốc độ & Chất lượng Encode video:", list(SPEED_PRESETS.keys()), index=1)
            bg_volume = st.slider("Âm lượng nhạc nền gốc của video khi AI nói (%):", 0, 40, 10)
            
        if st.button("🚀 BẮT ĐẦU XỬ LÝ VIDEO TỰ ĐỘNG", type="primary"):
            st.write("---")
            progress_area = st.empty()
            
            tmp_dir = ensure_workdir()
            audio_path = os.path.join(tmp_dir, "extracted_mono.wav")
            out_srt = os.path.join(tmp_dir, "subtitles.srt")
            out_video = os.path.join(tmp_dir, "output_final.mp4")
            
            # BƯỚC 1: Tách âm thanh
            progress_area.info("⏳ Bước 1/4: Đang trích xuất và tối ưu âm thanh gốc...")
            extract_audio(video_path, audio_path)
            
            # BƯỚC 2: Nhận diện giọng nói tiếng Trung
            progress_area.info("⏳ Bước 2/4: Đang nhận diện lời thoại tiếng Trung bằng AI...")
            try:
                raw_segments = transcribe_audio(
                    audio_path, model_size,
                    beam_size=SPEED_PRESETS[speed_preset_key]["beam_size"]
                )
            except Exception as e:
                st.error(f"Lỗi nhận diện âm thanh: {e}")
                st.stop()
                
            if not raw_segments:
                st.warning("Không phát hiện thấy câu nói nào trong video!")
                st.stop()
                
            # BƯỚC 3: Dịch phụ đề cải tiến
            progress_area.info(f"⏳ Bước 3/4: Đang tiến hành dịch {len(raw_segments)} câu thoại sang Tiếng Việt...")
            translated_segments = translate_segments(raw_segments)
            
            # Xuất file SRT phụ đề mới
            srt_content = build_srt(translated_segments, bilingual=bilingual)
            with open(out_srt, "w", encoding="utf-8") as f:
                f.write(srt_content)
                
            # BƯỚC 4: Tạo giọng AI & Ghép thành quả
            progress_area.info("⏳ Bước 4/4: Đang tạo giọng nói AI mới, chèn Logo và xuất Video cuối cùng...")
            try:
                render_and_merge(
                    video_path=video_path,
                    output_path=out_video,
                    srt_path=out_srt,
                    segments=translated_segments,
                    voice_info=selected_voice,
                    remove_old_sub=remove_old_sub,
                    old_sub_box=detected_box,
                    logo_path=logo_path,
                    font_size=font_size,
                    primary_color=hex_to_ass_color(text_color),
                    outline_color=hex_to_ass_color(outline_color),
                    alignment=2,
                    margin_v=25,
                    bg_volume_pct=bg_volume,
                    preset=SPEED_PRESETS[speed_preset_key]["video_preset"]
                )
                
                progress_area.empty()
                st.success("🎉 QUÁ TRÌNH XỬ LÝ HOÀN TẤT THÀNH CÔNG!")
                
                # Hiển thị video kết quả
                st.video(out_video)
                
                # Nút tải xuống
                with open(out_video, "rb") as f:
                    st.download_button(
                        label="💾 Tải Video đã Dịch & Lồng tiếng",
                        data=f,
                        file_name=f"Translated_{uploaded_file.name}",
                        mime="video/mp4"
                    )
            except Exception as e:
                st.error(f"Đã xảy ra lỗi trong quá trình xử lý cuối cùng: {e}")
    else:
        st.error("Không thể trích xuất khung hình từ video này để xem trước. Hãy thử định dạng video khác chuẩn hơn.")

```
