# Download and extract room_1D.mp4
(uv run utils/download_gdrive.py https://drive.google.com/file/d/1GAMrAmxUiQb--3gmDs5tJbrmg8q6B0fk/view ./data/room_1D.mp4 && uv run utils/extract_video.py ./data/room_1D.mp4) &

# Download and extract room_2D.mp4
(uv run utils/download_gdrive.py https://drive.google.com/file/d/1GGkV6i6W01o5WY_poBz1j11XsB8agTVY/view ./data/room_2D.mp4 && uv run utils/extract_video.py ./data/room_2D.mp4)
