import os
from roboflow import Roboflow
from ultralytics.models import YOLO
from dotenv import load_dotenv

load_dotenv()

rf = Roboflow(api_key=os.getenv("API_KEY"))
project = rf.workspace("p0sseid0n").project("cruz-de-malta")
version = project.version(11)
dataset = version.download("yolo26")
                
model = YOLO("yolo26s.pt") 

model.train(
    data=f"{dataset.location}/data.yaml", 
    epochs=100,       # Vai passar 50 vezes pelas 32 fotos
    plots=True,
    device=0,
    batch=16,
    workers=0
)