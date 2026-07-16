"""
App Streamlit: Dịch phụ đề video tiếng Trung sang tiếng Việt,
ghi phụ đề vào video, và lồng tiếng AI bằng giọng đọc tiếng Việt.

CÀI ĐẶT (chạy 1 lần, local):
    pip install streamlit faster-whisper deep-translator edge-tts --break-system-packages
    # cần có ffmpeg cài sẵn trên máy (sudo apt install ffmpeg / brew install ffmpeg)

CHẠY APP (local):
    streamlit run app.py

TRÊN STREAMLIT CLOUD:
    Cần có 2 file cùng thư mục gốc:
    - requirements.txt (streamlit, faster-whisper, deep-translator, edge-tts)
    - packages.txt (ffmpeg)
"""

import asyncio
import os
import subprocess
import tempfile

import streamlit as st

st.set_page_config(page_title="Dịch phụ đề & Lồng tiếng video Trung -> Việt", page_icon="🎬", layout="centered")

# ---------- DANH SÁCH GIỌNG ĐỌC (edge-tts, tiếng Việt) ----------
VOICE_PROFILES = {
    "Nữ - Hoài My (hoạt ngôn, nhanh nhẹn)": {"voice": "vi-VN-HoaiMyNeural", "rate": "+20%", "pitch": "+10Hz"},
    "Nữ - Hoài My (dịu dàng, tự nhiên)": {"voice": "vi-VN-HoaiMyNeural", "rate": "+0%", "pitch": "+0Hz"},
    "Nam - Nam Minh (năng động)": {"voice": "vi-VN-NamMinhNeural", "rate": "+15%", "pitch": "+5Hz"},
    "Nam - Nam Minh (trầm ấm, chững chạc)": {"voice": "vi-VN-NamMinhNeural", "rate": "+0%", "pitch": "-3Hz"},
}


# ---------- CÁC HÀM XỬ LÝ CHUNG ----------

def extract_audio(video_path: str, audio_path: str):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ffprobe_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_audio(audio_path: str, model_size: str, progress_cb=None):
    model = load_whisper_model(model_size)
    segments, info = model.transcribe(audio_path, language="zh", vad_filter=True)
    results = []
    for seg in segments:
        results.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
        if progress_cb:
            progress_cb(seg.end, seg.text.strip())
    return results


def translate_segments(segments, progress_cb=None):
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source="zh-CN", target="vi")
    for i, seg in enumerate(segments):
        try:
            seg["translated"] = translator.translate(seg["text"])
        except Exception:
            seg["translated"] = seg["text"]
        if progress_cb:
            progress_cb(i + 1, len(segments))
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


def burn_subtitles(video_path: str, srt_path: str, output_path: str):
    srt_escaped = srt_path.replace(":", "\\:")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"subtitles={srt_escaped}:force_style='FontName=Arial,FontSize=20,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=1'",
        "-c:a", "copy",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------- CÁC HÀM LỒNG TIẾNG (TTS) ----------

def synthesize_tts(text: str, voice: str, rate: str, pitch: str, out_path: str):
    """Gọi thẳng thư viện edge_tts (Python) thay vì gọi lệnh CLI 'edge-tts',
    vì trên một số môi trường (như Streamlit Cloud) lệnh CLI có thể không nằm trong PATH."""
    import edge_tts

    async def _run():
        communicate = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
        await communicate.save(out_path)

    try:
        asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run())
        loop.close()


def fit_audio_to_duration(in_path: str, out_path: str, target_duration: float):
    """Nếu giọng đọc dài hơn khoảng thời gian của câu phụ đề, tăng tốc để vừa khớp."""
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


def build_dub_track(segments, voice_profile, tmp_dir, total_duration, progress_cb=None):
    seg_files = []
    for i, seg in enumerate(segments):
        text = seg["translated"].strip()
        if progress_cb:
            progress_cb(i + 1, len(segments))
        if not text:
            continue
        raw_path = os.path.join(tmp_dir, f"tts_raw_{i}.mp3")
        fit_path = os.path.join(tmp_dir, f"tts_fit_{i}.wav")
        synthesize_tts(text, voice_profile["voice"], voice_profile["rate"], voice_profile["pitch"], raw_path)
        slot_duration = max(seg["end"] - seg["start"], 0.3)
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
    filter_complex += "".join(amix_labels) + f"amix=inputs={len(amix_labels)}:duration=first:dropout_transition=0[aout]"

    cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_complex, "-map", "[aout]", dub_track_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dub_track_path


def combine_video_with_dub(video_path: str, dub_track_path: str, output_path: str, keep_bg: bool, bg_volume: float = 0.15):
    if keep_bg:
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-i", dub_track_path,
            "-filter_complex", f"[0:a]volume={bg_volume}[bg];[1:a]anull[dub];[bg][dub]amix=inputs=2:duration=first[aout]",
            "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-shortest", output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-i", dub_track_path,
            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-shortest", output_path,
        ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------- GIAO DIỆN ----------

st.title("🎬 Dịch phụ đề & Lồng tiếng video Trung → Việt")
st.write("Tải video tiếng Trung lên. Ứng dụng sẽ nhận diện giọng nói, dịch sang tiếng Việt, ghi phụ đề và có thể lồng tiếng AI.")

with st.expander("⚙️ Tùy chọn nhận diện & phụ đề", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        model_size = st.selectbox("Độ chính xác (model)", ["tiny", "base", "small", "medium", "large-v3"], index=2,
                                   help="Model càng lớn càng chính xác nhưng càng chậm.")
    with col2:
        bilingual = st.checkbox("Hiện song ngữ (Trung + Việt) trong phụ đề chữ", value=False)
    burn_sub = st.checkbox("📝 Ghi phụ đề chữ lên video", value=True)

with st.expander("🎙️ Tùy chọn lồng tiếng AI", expanded=True):
    enable_dub = st.checkbox("Bật lồng tiếng AI bằng giọng đọc tiếng Việt", value=False)
    voice_label = st.selectbox("Chọn giọng đọc", list(VOICE_PROFILES.keys()), disabled=not enable_dub)
    keep_bg = st.checkbox("Giữ lại âm thanh/nhạc nền gốc (âm lượng nhỏ)", value=True, disabled=not enable_dub)

uploaded_file = st.file_uploader("Chọn file video", type=["mp4", "mkv", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("🚀 Bắt đầu xử lý", type="primary"):
        if not burn_sub and not enable_dub:
            st.warning("Bạn cần chọn ít nhất một trong hai: ghi phụ đề chữ hoặc lồng tiếng AI.")
            st.stop()

        with tempfile.TemporaryDirectory() as tmp_dir:
            ext = os.path.splitext(uploaded_file.name)[1] or ".mp4"
            input_path = os.path.join(tmp_dir, "input" + ext)
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            audio_path = os.path.join(tmp_dir, "audio.wav")
            srt_path = os.path.join(tmp_dir, "output.srt")

            status = st.status("Đang xử lý...", expanded=True)

            status.write("🔊 Đang tách âm thanh...")
            extract_audio(input_path, audio_path)
            total_duration = ffprobe_duration(input_path)

            status.write("🗣️ Đang nhận diện giọng nói tiếng Trung...")
            transcribe_placeholder = st.empty()
            segments = transcribe_audio(
                audio_path, model_size,
                progress_cb=lambda t, text: transcribe_placeholder.write(f"  [{t:.1f}s] {text}")
            )

            if not segments:
                status.update(label="Không nhận diện được giọng nói nào.", state="error")
                st.stop()
            status.write(f"✅ Nhận diện xong {len(segments)} câu.")

            status.write("🌐 Đang dịch sang tiếng Việt...")
            translate_bar = st.progress(0)
            segments = translate_segments(segments, progress_cb=lambda d, t: translate_bar.progress(d / t))

            srt_content = build_srt(segments, bilingual=bilingual)
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

            # Bước 1: ghi phụ đề chữ (nếu chọn)
            current_video = input_path
            if burn_sub:
                status.write("🎞️ Đang ghi phụ đề chữ vào video...")
                sub_output = os.path.join(tmp_dir, "with_sub.mp4")
                burn_subtitles(current_video, srt_path, sub_output)
                current_video = sub_output

            # Bước 2: lồng tiếng AI (nếu chọn)
            if enable_dub:
                status.write("🎙️ Đang tạo giọng đọc AI cho từng câu...")
                tts_bar = st.progress(0)
                voice_profile = VOICE_PROFILES[voice_label]
                dub_track_path = build_dub_track(
                    segments, voice_profile, tmp_dir, total_duration,
                    progress_cb=lambda d, t: tts_bar.progress(d / t)
                )
                status.write("🔀 Đang ghép giọng lồng tiếng vào video...")
                dub_output = os.path.join(tmp_dir, "with_dub.mp4")
                combine_video_with_dub(current_video, dub_track_path, dub_output, keep_bg=keep_bg)
                current_video = dub_output

            status.update(label="Hoàn tất!", state="complete")

            with open(current_video, "rb") as f:
                video_bytes = f.read()

            st.success("✅ Xử lý xong! Xem trước video bên dưới trước khi tải về.")
            st.subheader("👀 Xem trước")
            st.video(video_bytes)

            st.subheader("⬇️ Tải xuống")
            colA, colB = st.columns(2)
            with colA:
                st.download_button("⬇️ Tải video kết quả", data=video_bytes,
                                    file_name="video_ket_qua.mp4", mime="video/mp4")
            with colB:
                st.download_button("⬇️ Tải file phụ đề (.srt)", data=srt_content,
                                    file_name="phu_de.srt", mime="text/plain")

            with st.expander("📜 Xem nội dung phụ đề"):
                for seg in segments:
                    st.write(f"**[{seg['start']:.1f}s - {seg['end']:.1f}s]**")
                    if bilingual:
                        st.write(f"🇨🇳 {seg['text']}")
                    st.write(f"🇻🇳 {seg['translated']}")
                    st.divider()
else:
    st.info("👆 Hãy tải lên một video tiếng Trung để bắt đầu.")
