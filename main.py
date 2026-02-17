import os
import cv2
import mss 
import numpy as np 
from ultralytics.models import YOLO
import time
from playsound3 import playsound

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

model_path = "runs/detect/train16/weights/best.pt"
found_folder = "found"
audio_file = "hino.mp3"

if not os.path.exists(found_folder):
    os.makedirs(found_folder)
else:
    print(f"Pasta '{found_folder}' já existe. Limpando arquivos antigos...") 

    for file in os.listdir(found_folder):
        os.remove(os.path.join(found_folder, file))


detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=model_path,
    confidence_threshold=0.8,
    device="cuda:0"
)

has_to_play_sound = False
is_playing_sound = False
sound = None

with mss.mss() as sct:
    bounding_box = sct.monitors[1]

    while True:
        sct_img = sct.grab(bounding_box)
        frame = np.array(sct_img)
        frame = np.ascontiguousarray(frame[:, :, :3])

        slice_size = 320
        result = get_sliced_prediction(
            frame,
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=0.5,
            overlap_width_ratio=0.5
        )

        object_prediction_list = result.object_prediction_list
        found = len(result.object_prediction_list) > 0

        annotated_frame = frame.copy()

        for prediction in object_prediction_list:
            if prediction.category.name != "cruz-de-malta":
                found = False
                continue

            bbox = prediction.bbox
            x, y, maxX, maxY = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)

            cv2.rectangle(annotated_frame, (x, y), (maxX, maxY), (0, 255, 0), 2)
            text = f"{prediction.category.name} {prediction.score.value:.2f}"
            cv2.putText(annotated_frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imwrite('frame.jpg', annotated_frame)

        if found:
            print("✅ Achou")

            timestamp = time.strftime("%d_%m-%H_%M_%S")
            filename = f"{found_folder}/{timestamp}.jpg"

            cv2.imwrite(filename, annotated_frame)

        if found and has_to_play_sound and (sound is None or not sound.is_alive()):
            sound = playsound(audio_file, block=False)

        if not found and not has_to_play_sound and sound is not None and sound.is_alive():
            sound.stop()

        has_to_play_sound = found

                