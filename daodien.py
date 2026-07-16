import streamlit as st
import os
import tempfile

# ==========================
# Cấu hình trang
# ==========================
st.set_page_config(
    page_title="AI Dịch Video Trung -> Việt",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI Dịch Phụ Đề Video Trung ➜ Việt")
st.write("Upload video, AI sẽ nhận dạng tiếng Trung, dịch sang tiếng Việt và ghi phụ đề vào video.")

# ==========================
# Sidebar
# ==========================
st.sidebar.header("⚙️ Cài đặt")

font_size = st.sidebar.slider(
    "Cỡ chữ",
    min_value=18,
    max_value=60,
    value=32
)

font_color = st.sidebar.color_picker(
    "Màu chữ",
    "#FFFFFF"
)

outline = st.sidebar.checkbox(
    "Viền đen",
    value=True
)

generate_srt = st.sidebar.checkbox(
    "Xuất file SRT",
    value=True
)

burn_subtitle = st.sidebar.checkbox(
    "Ghi phụ đề vào video",
    value=True
)

# ==========================
# Upload Video
# ==========================

video = st.file_uploader(
    "📂 Chọn video",
    type=["mp4", "mov", "avi", "mkv"]
)

if video:

    st.success("Đã tải video thành công!")

    st.video(video)

    # Lưu tạm
    temp_dir = tempfile.mkdtemp()

    video_path = os.path.join(
        temp_dir,
        video.name
    )

    with open(video_path, "wb") as f:
        f.write(video.read())

    st.info(f"Video được lưu tại:\n{video_path}")

    # ==========================
    # Nút xử lý
    # ==========================

    if st.button("🚀 BẮT ĐẦU DỊCH"):

        progress = st.progress(0)

        status = st.empty()

        status.info("🎵 Đang tách âm thanh...")
        progress.progress(20)

        # audio.extract_audio(video_path)

        status.info("🎤 Đang nhận dạng tiếng Trung...")
        progress.progress(40)

        # whisper.transcribe()

        status.info("🌐 Đang dịch sang tiếng Việt...")
        progress.progress(60)

        # translator.translate()

        status.info("📝 Đang tạo phụ đề...")
        progress.progress(80)

        # subtitle.create_srt()

        if burn_subtitle:
            status.info("🎬 Đang ghi phụ đề vào video...")
            progress.progress(95)

            # render.render_video()

        progress.progress(100)

        status.success("✅ Hoàn thành!")

        st.success("Các module xử lý sẽ được kết nối ở những phần tiếp theo.")

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "⬇ Download SRT",
                data="",
                file_name="subtitle.srt"
            )

        with col2:
            st.download_button(
                "⬇ Download Video",
                data="",
                file_name="video_sub.mp4"
            )

else:

    st.warning("Hãy chọn một video để bắt đầu.")
