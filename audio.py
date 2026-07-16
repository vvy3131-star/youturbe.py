"""
App Streamlit: Dịch phụ đề video tiếng Trung sang tiếng Việt và ghi vào video.

CÀI ĐẶT (chạy 1 lần):
    pip install streamlit faster-whisper deep-translator --break-system-packages
    # cần có ffmpeg cài sẵn trên máy (sudo apt install ffmpeg / brew install ffmpeg)

CHẠY APP:
    streamlit run app.py

Sau đó trình duyệt sẽ tự mở tại http://localhost:8501
"""

import os
import subprocess
import tempfile
import time

import streamlit as st

st.set_page_config(page_title="Dịch phụ đề video Trung -> Việt", page_icon="🎬", layout="centered")


# ---------- CÁC HÀM XỬ LÝ ----------

def extract_audio(video_path: str, audio_path: str):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
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


# ---------- GIAO DIỆN ----------

st.title("🎬 Dịch phụ đề video Trung → Việt")
st.write("Tải video tiếng Trung lên, ứng dụng sẽ tự nhận diện giọng nói, dịch sang tiếng Việt và ghi phụ đề vào video.")

col1, col2 = st.columns(2)
with col1:
    model_size = st.selectbox(
        "Độ chính xác (model)",
        ["tiny", "base", "small", "medium", "large-v3"],
        index=2,
        help="Model càng lớn càng chính xác nhưng càng chậm.",
    )
with col2:
    bilingual = st.checkbox("Hiện song ngữ (Trung + Việt)", value=False)

uploaded_file = st.file_uploader("Chọn file video", type=["mp4", "mkv", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("🚀 Bắt đầu xử lý", type="primary"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "input" + os.path.splitext(uploaded_file.name)[1])
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            audio_path = os.path.join(tmp_dir, "audio.wav")
            srt_path = os.path.join(tmp_dir, "output.srt")
            output_path = os.path.join(tmp_dir, "output.mp4")

            status = st.status("Đang xử lý...", expanded=True)

            status.write("🔊 Đang tách âm thanh...")
            extract_audio(input_path, audio_path)

            status.write("🗣️ Đang nhận diện giọng nói tiếng Trung...")
            transcribe_placeholder = st.empty()

            def on_transcribe_progress(t, text):
                transcribe_placeholder.write(f"  [{t:.1f}s] {text}")

            segments = transcribe_audio(audio_path, model_size, progress_cb=on_transcribe_progress)

            if not segments:
                status.update(label="Không nhận diện được giọng nói nào.", state="error")
                st.stop()

            status.write(f"✅ Nhận diện xong {len(segments)} câu.")

            status.write("🌐 Đang dịch sang tiếng Việt...")
            translate_bar = st.progress(0)

            def on_translate_progress(done, total):
                translate_bar.progress(done / total)

            segments = translate_segments(segments, progress_cb=on_translate_progress)

            srt_content = build_srt(segments, bilingual=bilingual)
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

            status.write("🎞️ Đang ghi phụ đề vào video...")
            burn_subtitles(input_path, srt_path, output_path)

            status.update(label="Hoàn tất!", state="complete")

            st.success("Xử lý xong! Xem kết quả bên dưới.")

            with open(output_path, "rb") as f:
                video_bytes = f.read()

            st.video(video_bytes)

            colA, colB = st.columns(2)
            with colA:
                st.download_button(
                    "⬇️ Tải video có phụ đề",
                    data=video_bytes,
                    file_name="video_co_phu_de.mp4",
                    mime="video/mp4",
                )
            with colB:
                st.download_button(
                    "⬇️ Tải file phụ đề (.srt)",
                    data=srt_content,
                    file_name="phu_de.srt",
                    mime="text/plain",
                )

            with st.expander("📜 Xem nội dung phụ đề"):
                for seg in segments:
                    st.write(f"**[{seg['start']:.1f}s - {seg['end']:.1f}s]**")
                    if bilingual:
                        st.write(f"🇨🇳 {seg['text']}")
                    st.write(f"🇻🇳 {seg['translated']}")
                    st.divider()
else:
    st.info("👆 Hãy tải lên một video tiếng Trung để bắt đầu.")
