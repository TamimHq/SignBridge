import requests

tests = [
    (r"D:\project_sL\SignBD-Word\SignBD-Word_RGB\DATASET\maa\p1_c_maa.mp4", "maa"),
    (r"D:\project_SL\SignBD-Word\SignBD-Word_RGB\DATASET\baba\p1_c_baba.mp4", "baba"),
    (r"D:\project_sL\SignBD-Word\SignBD-Word_RGB\DATASET\valo\p1_c_valo.mp4", "valo"),
    (r"D:\project_sL\SignBD-Word\SignBD-Word_RGB\DATASET\pakhi\p5_c_pakhi.mp4", "pakhi"),
    (r"D:\project_sL\SignBD-Word\SignBD-Word_RGB\DATASET\maa\p7_c_maa.mp4", "maa"),
]

correct = 0
for path, expected in tests:
    try:
        with open(path, 'rb') as f:
            r = requests.post(
                "http://localhost:8000/api/recognize_video",
                files={'video': ('clip.mp4', f, 'video/mp4')},
                data={'language': 'bdsl_bn'},
                timeout=60,
            )
    except FileNotFoundError:
        print(f"?  {expected:10} — file not found: {path}")
        continue
    except Exception as e:
        print(f"✗  {expected:10} — request failed: {e}")
        continue

    if r.status_code != 200:
        print(f"✗  {expected:10} — HTTP {r.status_code}: {r.text[:150]}")
        continue

    got = r.json()
    ok = got['word'] == expected
    correct += ok
    mark = "✓" if ok else "✗"
    print(f"{mark}  expected {expected:10} got {got['word']:10} conf {got['confidence']}")

print(f"\n{correct}/{len(tests)} correct")