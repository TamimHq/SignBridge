import requests
import numpy as np

# Fake 30-frame keypoint sequence (30 frames x 144 features)
fake_keypoints = np.random.randn(30, 144).tolist()

response = requests.post(
    "http://localhost:8000/api/recognize",
    json={"keypoints": fake_keypoints, "language": "bdsl_bn"}
)

print("Status:", response.status_code)
print("Response:", response.json())