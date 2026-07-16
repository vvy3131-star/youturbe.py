import os
import ffmpeg
import config


def extract_audio(video_path):
    """
    Tách audio từ video
    """

    output_audio = os.path.join(
        config.TEMP_FOLDER,
        "audio.wav"
    )

    if os.path.exists(output_audio):
        os.remove(output_audio)

    (
        ffmpeg
        .input(video_path)
        .output(
            output_audio,
            ac=1,
            ar=config.SAMPLE_RATE,
            format="wav"
        )
        .overwrite_output()
        .run(quiet=True)
    )

    return output_audio