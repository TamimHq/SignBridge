import requests
import numpy as np

import requests

# Use a real BdSL video — you know what sign it is
video_path = r"E:\SL\SignBD-Word\SignBD-Word_RGB\DATASET\maa\p1_c_maa.mp4"

with open(video_path, 'rb') as f:
    files = {'video': ('clip.mp4', f, 'video/mp4')}
    data = {'language': 'bdsl_bn'}
    r = requests.post("http://localhost:8000/api/recognize_video", files=files, data=data)

print("Status:", r.status_code)
print("Response:", r.json())