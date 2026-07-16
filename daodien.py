import asyncio
import hashlib
import os
import shutil
import subprocess
import tempfile
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

# Khởi chạy giao diện dạng Wide siêu rộng để dễ kiểm soát
st.set_page_config(page_title="Dịch & Lồng Tiếng Siêu Tốc", page_icon="⚡", layout="wide")

# CSS Thiết kế giao diện sống động, trực quan và nhiều màu sắc
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
    }
    .main-title {
        color: #1e3a8a;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2px;
    }
    .section-auto {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #0284c7;
        margin-bottom: 18px;
        color: #0369a1;
    }
    .section-manual {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #d97706;
        margin-bottom: 18px;
        color: #78350f;
    }
    .section-audio {
        background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #9333ea;
        margin-bottom: 18px;
        color: #6b21a8;
    }
    div[data-testid="stVerticalBlock"] > div:has(> div.stButton) button {
        background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 1.2rem !important;
        border-radius: 10px !important;
        height: 55px !important;
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# ---------- DANH MỤC GIỌNG ĐỌC ----------
VOICE_CATALOG = [
    {"id": "female_default", "label": "👩 Hoài My - Giọng Chuẩn (Tự nhiên)", "provider": "edge", "voice": "vi-VN-HoaiMyNeural", "rate": "+0%", "pitch": "+0Hz"},
    {"id": "female_young", "label": "👩 Hoài My - Trẻ trung, Vui vẻ", "provider": "edge", "voice": "vi-VN-HoaiMyNeural", "rate": "+4%", "pitch": "+2Hz"},
    {"id": "female_gentle", "label": "👩 Hoài My - Dịu dàng", "provider": "edge", "voice": "vi-VN-HoaiMyNeural", "rate": "-4%", "pitch": "-1Hz"},
    {"id": "google_female", "label": "👩 Giọng nữ Google (gTTS)", "provider": "gtts", "voice": "vi"},
    {"id": "male_default", "label": "👨 Nam Minh - Mặc định", "provider": "edge", "voice": "vi-VN-NamMinhNeural", "rate": "+0%", "pitch": "+0Hz"},
]

CPU_COUNT = os.cpu_count() or 4

# ============================================================
# CÁC HÀM CÔNG NGHỆ CHẠY ẨN TỐC ĐỘ CAO
# ============================================================

class FFmpegError(RuntimeError):
    pass

def _run_ffmpeg(cmd, label: str = "ffmpeg"):
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        stderr_tail = "\n".join(result.stderr.strip().splitlines()[-15:]) if result.stderr else "(không có thông tin)"
        raise FFmpegError(f"Bước '{label}' thất bại.\n\nChi tiết:\n{stderr_tail}")
    return result

def extract_audio(video_path: str, audio_path: str):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-threads", str(CPU_COUNT), audio_path]
    _run_ffmpeg(cmd, label="trích xuất âm thanh gốc")

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

def transcribe_audio(audio_path: str, model_size: str, beam_size: int = 1, progress_cb=None):
    model = load_whisper_model(model_size)
    segments, info = model.transcribe(audio_path, language="zh", vad_filter=True, beam_size=beam_size)
    results = []
    for seg in segments:
        results.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
        if progress_cb:
            progress_cb(seg.end, seg.text.strip())
    return results

def translate_segments_fast(segments, progress_cb=None, chunk_size: int = 80):
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
        seg["translated"] = " ".join(translated.split()) if translated else seg["text"]
    return segments

def build_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
        lines.append(seg["translated"])
        lines.append("")
    return "\n".join(lines)

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

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
# NHẬN DIỆN VÙNG PHỤ ĐỀ TỰ ĐỘNG
# ============================================================

def detect_subtitle_region_fast(video_path: str, width: int, height: int, tmp_dir: str):
    try:
        dur = ffprobe_duration(video_path)
        timestamps = [dur * 0.25, dur * 0.5, dur * 0.75]
        frame_paths = []
        for i, t in enumerate(timestamps):
            p = os.path.join(tmp_dir, f"fast_det_{i}.jpg")
            cmd = ["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", video_path, "-frames:v", "1", "-q:v", "5", p]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(p):
                frame_paths.append(p)

        if len(frame_paths) < 2:
            return int(width * 0.05), int(height * 0.78), int(width * 0.90), int(height * 0.16)

        row_profiles = []
        for p in frame_paths:
            img = Image.open(p).convert("L").resize((width, height))
            arr = np.asarray(img, dtype=np.float32)
            edge = np.abs(np.diff(arr, axis=1))
            edge = np.pad(edge, ((0, 0), (0, 1)))
            row_energy = edge.sum(axis=1)
            m = row_energy.max()
            row_profiles.append(row_energy / m if m > 0 else row_energy)

        avg_row = np.mean(row_profiles, axis=0)
        bottom_start = int(height * 0.55)
        bottom_zone = avg_row[bottom_start:]
        peak_idx = int(np.argmax(bottom_zone)) + bottom_start
        
        y_start = max(0, peak_idx - int(height * 0.08))
        y_end = min(height - 1, peak_idx + int(height * 0.08))
        
        return int(width * 0.05), int(y_start), int(width * 0.90), int(y_end - y_start)
    except Exception:
        return int(width * 0.05), int(height * 0.78), int(width * 0.90), int(height * 0.16)

# ============================================================
# LOGO THƯƠNG HIỆU
# ============================================================

def make_circle_logo(image_file, size: int = 120) -> str:
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
# CƠ CHẾ SỬA LỖI AI ĐỌC PHỤ ĐỀ VÀ PHỐI ÂM KHÔNG LỖI
# ============================================================

async def generate_voice_edge(text: str, voice_info: dict, output_path: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice_info["voice"], rate=voice_info["rate"], pitch=voice_info["pitch"])
    await communicate.save(output_path)

def generate_voice_gtts(text: str, output_path: str):
    from gtts import gTTS
    tts = gTTS(text=text, lang="vi")
    tts.save(output_path)

def render_and_merge_fast(video_path: str, output_path: str, srt_path: str, segments: list, voice_info: dict,
                          remove_old_sub: bool, old_sub_box, logo_path: str = None, font_size: int = 22,
                          primary_color: str = "&H00FFFFFF&", outline_color: str = "&H00000000&",
                          bg_volume_pct: int = 15, voice_volume_pct: int = 100,
                          alignment: int = 2, margin_v: int = 25):
    
    tmp_dir = ensure_workdir()
    audio_segments_paths = []
    
    # Sửa lỗi mất tiếng: Đảm bảo dọn dẹp và tạo đường dẫn MP3 tạm không trùng lặp
    for idx, seg in enumerate(segments):
        audio_seg_path = os.path.join(tmp_dir, f"voice_segment_{idx}_{hash(seg['translated'])}.mp3")
        text = seg["translated"]
        
        try:
            if voice_info["provider"] == "edge":
                asyncio.run(generate_voice_edge(text, voice_info, audio_seg_path))
            else:
                generate_voice_gtts(text, audio_seg_path)
            
            if os.path.exists(audio_seg_path) and os.path.getsize(audio_seg_path) > 0:
                audio_segments_paths.append((seg["start"], audio_seg_path))
        except Exception as tts_err:
            st.warning(f"Bỏ qua phân đoạn thoại {idx} do lỗi đọc: {tts_err}")

    if not audio_segments_paths:
        raise RuntimeError("Không có tệp lồng tiếng AI nào được tạo thành công! Vui lòng kiểm tra lại kết nối mạng.")

    # 2. Phối âm lượng thông minh tự động kết hợp kéo tay thủ công
    filter_complex_audio = ""
    inputs_audio_cmd = ["-i", video_path]
    
    for idx, (_, path) in enumerate(audio_segments_paths):
        inputs_audio_cmd.extend(["-i", path])
    
    bg_vol = bg_volume_pct / 100.0
    v_vol = voice_volume_pct / 100.0
    
    filter_complex_audio += f"[0:a]volume={bg_vol}[bg_audio];"
    
    mix_inputs = ""
    for idx, (start_time, _) in enumerate(audio_segments_paths):
        filter_complex_audio += f"[{idx+1}:a]adelay={int(start_time*1000)}|{int(start_time*1000)}[delay{idx}];"
        mix_inputs += f"[delay{idx}]"
    
    filter_complex_audio += f"{mix_inputs}amix=inputs={len(audio_segments_paths)}:dropout_transition=0,volume={v_vol}[dub_audio];"
    filter_complex_audio += f"[bg_audio][dub_audio]amix=inputs=2:duration=first[out_audio]"

    temp_audio_mixed = os.path.join(tmp_dir, "audio_mixed.mp3")
    cmd_audio = ["ffmpeg", "-y"] + inputs_audio_cmd + ["-filter_complex", filter_complex_audio, "-map", "[out_audio]", "-threads", str(CPU_COUNT), temp_audio_mixed]
    _run_ffmpeg(cmd_audio, label="hòa trộn nhạc nền và giọng AI lồng tiếng")

    # 3. Tạo bộ lọc hình ảnh (Xóa phụ đề cũ + Vị trí phụ đề mới)
    filters = []
    stage = "[0:v]"
    counter = 0

    if remove_old_sub and old_sub_box:
        x, y, w, h = old_sub_box
        filters.append(f"{stage}split=2[vmain{counter}][vcrop{counter}]")
        filters.append(
            f"[vcrop{counter}]crop={w}:{h}:{x}:{y},"
            f"boxblur=luma_radius=20:luma_power=1:chroma_radius=10:chroma_power=1[vblur{counter}]"
        )
        filters.append(f"[vmain{counter}][vblur{counter}]overlay={x}:{y}[v{counter}]")
        stage = f"[v{counter}]"
        counter += 1

    if logo_path and os.path.exists(logo_path):
        logo_input_idx = 1
        filters.append(f"{stage}[{logo_input_idx}:v]overlay=x=20:y=20[vlogo]")
        stage = "[vlogo]"

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

    cmd_video = ["ffmpeg", "-y", "-i", video_path]
    if logo_path and os.path.exists(logo_path):
        cmd_video.extend(["-i", logo_path])
        
    cmd_video.extend([
        "-i", temp_audio_mixed,
        "-filter_complex", filter_complex_video,
        "-map", stage,
        "-map", f"{2 if logo_path else 1}:a",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
        "-threads", "0",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ])
    _run_ffmpeg(cmd_video, label="ghép xuất video thành phẩm tốc độ cao")

# ============================================================
# CÁC HÀM XEM TRƯỚC (PREVIEW)
# ============================================================

def ensure_workdir() -> str:
    if "workdir" not in st.session_state:
        st.session_state.workdir = tempfile.mkdtemp(prefix="vidtrans_fast_")
    return st.session_state.workdir

def get_persistent_video_path(uploaded_file) -> str:
    ensure_workdir()
    file_hash = hashlib.md5(uploaded_file.getbuffer()).hexdigest()[:10]
    ext = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    path = os.path.join(st.session_state.workdir, f"src_{file_hash}{ext}")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return path

def extract_preview_frame(video_path: str, frame_path: str) -> bool:
    try:
        cmd = ["ffmpeg", "-y", "-ss", "00:00:03", "-i", video_path, "-vframes", "1", "-q:v", "2", frame_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return os.path.exists(frame_path)
    except Exception:
        try:
            cmd = ["ffmpeg", "-y", "-ss", "00:00:00", "-i", video_path, "-vframes", "1", "-q:v", "2", frame_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return os.path.exists(frame_path)
        except Exception:
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

def render_preview_with_box(frame_path: str, box, logo_path: str = None, font_size: int = 24, text_color: str = "#FFFFFF", alignment: int = 2, margin_v: int = 25) -> "Image.Image":
    img = Image.open(frame_path)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    if box:
        x, y, bw, bh = box
        draw.rectangle([x, y, x + bw, y + bh], outline="#FF3333", width=4)
        draw.text((x + 10, max(0, y - 25)), "VÙNG LÀM MỜ (XÓA SUB GỐC)", fill="#FF3333")
        
    if logo_path and os.path.exists(logo_path):
        logo_img = Image.open(logo_path).convert("RGBA")
        preview_logo_size = max(int(h * 0.12), 45)
        logo_img = logo_img.resize((preview_logo_size, preview_logo_size))
        img.paste(logo_img, (20, 20), mask=logo_img)
        draw.text((20, 25 + preview_logo_size), "LOGO", fill="#00FF00")
        
    try:
        sample_text = "Phụ đề mẫu tiếng Việt"
        y_pos = h - margin_v - font_size
        
        if alignment == 1:
            x_pos = int(w * 0.05)
        elif alignment == 3:
            x_pos = int(w * 0.95) - 200
        else:
            x_pos = (w - 200) // 2
            
        draw.rectangle([x_pos - 5, y_pos - 2, x_pos + 205, y_pos + font_size + 2], fill="#00000099")
        draw.text((x_pos, y_pos), sample_text, fill=text_color)
    except Exception:
        pass
        
    return img

# ============================================================
# GIAO DIỆN CHÍNH TRỰC QUAN MÀU SẮC (UI)
# ============================================================

st.markdown("<h1 class='main-title'>⚡ Dịch & Lồng Tiếng Thông Minh</h1>", unsafe_allow_html=True)
st.caption("<p style='text-align: center; color: #64748b; font-size: 1.05rem;'>Giao diện tối giản mới dễ dùng • Sửa triệt để lỗi mất giọng đọc lồng tiếng • Tự động & Thủ công song hành</p>", unsafe_allow_html=True)
st.write("---")

uploaded_file = st.file_uploader("📂 Hãy kéo thả hoặc chọn video của bạn tại đây:", type=["mp4", "mkv", "mov", "avi"])

if uploaded_file:
    video_path = get_persistent_video_path(uploaded_file)
    frame_path, resolution = ensure_preview_frame(video_path)
    
    if frame_path and resolution:
        width, height = resolution
        
        if "box_x" not in st.session_state:
            def_x, def_y, def_w, def_h = detect_subtitle_region_fast(video_path, width, height, st.session_state.workdir)
            st.session_state.box_x = def_x
            st.session_state.box_y = def_y
            st.session_state.box_w = def_w
            st.session_state.box_h = def_h

        # Chia màn hình làm 2 khu vực ngang rất dễ quan sát
        col_ctrl, col_preview = st.columns([1, 1])
        
        with col_ctrl:
            st.subheader("🛠️ Cài đặt Video & Âm thanh")
            
            # --- KHỐI ĐIỀU CHỈNH ÂM LƯỢNG (TỰ ĐỘNG & THỦ CÔNG) ---
            st.markdown("<div class='section-audio'>🔊 <b>ĐIỀU CHỈNH ÂM LƯỢNG</b></div>", unsafe_allow_html=True)
            auto_volume = st.checkbox("⚙️ Chế độ tự động tối ưu hóa âm lượng", value=True)
            
            if not auto_volume:
                bg_vol = st.slider("📁 Âm lượng video gốc (Nhạc nền):", 0, 100, 15, format="%d%%")
                voice_vol = st.slider("🎙️ Âm lượng lồng tiếng AI:", 0, 200, 100, format="%d%%")
            else:
                bg_vol = 15
                voice_vol = 100
                st.info("🎯 *Đang tự động: Hạ nhạc nền gốc xuống 15%, Đẩy giọng AI lồng tiếng lên 100%*")
                
            # --- KHỐI VỊ TRÍ PHỤ ĐỀ MỚI ---
            st.markdown("<div class='section-auto'>📝 <b>VỊ TRÍ PHỤ ĐỀ MỚI</b></div>", unsafe_allow_html=True)
            auto_sub_pos = st.checkbox("⚙️ Chế độ tự động căn giữa chân màn hình", value=True)
            
            if not auto_sub_pos:
                align_option = st.selectbox("Lề phụ đề mới:", ["Căn giữa (Center)", "Căn trái (Left)", "Căn phải (Right)"])
                alignment_map = {"Căn trái (Left)": 1, "Căn giữa (Center)": 2, "Căn phải (Right)": 3}
                sub_alignment = alignment_map[align_option]
                sub_margin_v = st.slider("Chỉnh khoảng cách lề đáy (px):", 5, int(height * 0.4), 25)
            else:
                sub_alignment = 2
                sub_margin_v = 25
                st.info("🎯 *Đang tự động: Phụ đề mới xuất hiện chính giữa cách đáy 25px*")

            # --- KHỐI KHUNG XÓA SUB CŨ ---
            st.markdown("<div class='section-manual'>🖼️ <b>KÍCH THƯỚC KHUNG XÓA SUB</b></div>", unsafe_allow_html=True)
            remove_old_sub = st.checkbox("Bật bộ lọc xóa phụ đề tiếng Trung gốc", value=True)
            
            if remove_old_sub:
                auto_box = st.checkbox("⚙️ Tự động quét vùng chữ tiếng Trung", value=True)
                if not auto_box:
                    bx = st.slider("Tọa độ ngang X:", 0, width, st.session_state.box_x)
                    by = st.slider("Tọa độ dọc Y:", 0, height, st.session_state.box_y)
                    bw = st.slider("Chiều rộng khung xóa (Width):", 10, width - bx, st.session_state.box_w)
                    bh = st.slider("Chiều cao khung xóa (Height):", 10, height - by, st.session_state.box_h)
                    
                    st.session_state.box_x = bx
                    st.session_state.box_y = by
                    st.session_state.box_w = bw
                    st.session_state.box_h = bh
                    current_box = (bx, by, bw, bh)
                else:
                    current_box = (st.session_state.box_x, st.session_state.box_y, st.session_state.box_w, st.session_state.box_h)
                    st.info("🎯 *Đang sử dụng tọa độ tự động quét bởi thuật toán AI*")
            else:
                current_box = None
                
            st.write("")
            logo_file = st.file_uploader("In logo góc trái trên màn hình (nếu có):", type=["png", "jpg", "jpeg"])
            logo_path = None
            if logo_file:
                logo_target_size = max(int(height * 0.12), 60)
                logo_path = make_circle_logo(logo_file, size=logo_target_size)
                st.success("Logo đã được nén tròn thành công!")
                
            st.write("")
            voice_labels = [v["label"] for v in VOICE_CATALOG]
            selected_voice_label = st.selectbox("Chọn giọng đọc lồng tiếng Việt:", voice_labels)
            selected_voice = next(v for v in VOICE_CATALOG if v["label"] == selected_voice_label)

        with col_preview:
            st.subheader("📺 Khung Nhìn Thử Nghiệm (Preview)")
            
            font_size = st.slider("Điều chỉnh kích cỡ chữ phụ đề (px):", 12, 60, 24)
            text_color = st.color_picker("Tự chọn màu chữ mới:", "#FFFFFF")
            outline_color = auto_outline_color(text_color)
            
            preview_img = render_preview_with_box(
                frame_path, 
                current_box if remove_old_sub else None, 
                logo_path,
                font_size=font_size,
                text_color=text_color,
                alignment=sub_alignment,
                margin_v=sub_margin_v
            )
            st.image(preview_img, use_container_width=True, caption="Ảnh mô phỏng tọa độ xóa phụ đề và vị trí chữ")

        if st.button("🚀 BẮT ĐẦU XỬ LÝ VIDEO NGAY", type="primary"):
            st.write("---")
            progress_area = st.empty()
            
            tmp_dir = ensure_workdir()
            audio_path = os.path.join(tmp_dir, "extracted_mono.wav")
            out_srt = os.path.join(tmp_dir, "subtitles.srt")
            out_video = os.path.join(tmp_dir, "output_final.mp4")
            
            # Tiến trình 1
            progress_area.info("⏳ Đang bóc tách nhạc nền gốc...")
            extract_audio(video_path, audio_path)
            
            # Tiến trình 2
            progress_area.info("⏳ Đang quét nhận diện chữ tiếng Trung bằng Whisper...")
            try:
                raw_segments = transcribe_audio(audio_path, "base", beam_size=1)
            except Exception as e:
                st.error(f"Lỗi nhận dạng: {e}")
                st.stop()
                
            if not raw_segments:
                st.warning("Không phát hiện câu thoại nào trong video.")
                st.stop()
                
            # Tiến trình 3
            progress_area.info("⏳ Đang tiến hành chuyển đổi ngôn ngữ...")
            translated_segments = translate_segments_fast(raw_segments)
            
            srt_content = build_srt(translated_segments)
            with open(out_srt, "w", encoding="utf-8") as f:
                f.write(srt_content)
                
            # Tiến trình 4
            progress_area.info("⏳ Đang thực hiện lồng tiếng AI & Ghép phụ đề tốc độ cao...")
            try:
                render_and_merge_fast(
                    video_path=video_path,
                    output_path=out_video,
                    srt_path=out_srt,
                    segments=translated_segments,
                    voice_info=selected_voice,
                    remove_old_sub=remove_old_sub,
                    old_sub_box=current_box,
                    logo_path=logo_path,
                    font_size=font_size,
                    primary_color=hex_to_ass_color(text_color),
                    outline_color=hex_to_ass_color(outline_color),
                    bg_volume_pct=bg_vol,
                    voice_volume_pct=voice_vol,
                    alignment=sub_alignment,
                    margin_v=sub_margin_v
                )
                
                progress_area.empty()
                st.success("🎉 QUÁ TRÌNH HOÀN TẤT THÀNH CÔNG!")
                st.video(out_video)
                
                with open(out_video, "rb") as f:
                    st.download_button(
                        label="💾 Tải Video Hoàn Chỉnh Về Máy",
                        data=f,
                        file_name=f"Perfect_Dub_{uploaded_file.name}",
                        mime="video/mp4"
                    )
            except Exception as e:
                st.error(f"Gặp trục trặc trong khâu render cuối: {e}")
