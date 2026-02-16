import cv2
import mss 
import numpy as np 
from ultralytics.models import YOLO
import time

model_path = "runs/detect/train5/weights/best.pt"

model = YOLO(model_path)

with mss.mss() as sct:
    bounding_box = sct.monitors[1]


    while True:
        sct_img = sct.grab(bounding_box)
        frame = np.array(sct_img)
        frame = np.ascontiguousarray(frame[:, :, :3])

        results = model(frame, conf=0.1, verbose=False) 

        found = len(results[0].boxes) > 0 

        annotated_frame = results[0].plot()

        cv2.imwrite('frame.jpg', annotated_frame)

        print("Achado: ", found)

        if found:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"found/{timestamp}.jpg"

            cv2.imwrite(filename, annotated_frame)

                