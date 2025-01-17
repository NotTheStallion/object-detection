import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision import models
import os
from tqdm import tqdm
import wandb

import time
import psutil
from PIL import Image
import torch
from torchvision import transforms

from codecarbon import EmissionsTracker
import cv2



def energy_profiler(func):
    def wrapper(*args, **kwargs):
        wandb.init(project="object_classification", name=func.__name__)
        tracker = EmissionsTracker(log_level="error")
        
        tracker.start()

        result = func(*args, **kwargs)
        
        emissions: float = tracker.stop()
        
        total_cpu_energy = 0
        total_ram_energy = 0
        total_gpu_energy = 0
        total_energy = 0
        emissions_gCO2e = 0
        
        try :
            total_cpu_energy = tracker._total_cpu_energy.kWh * 1000
            total_ram_energy = tracker._total_ram_energy.kWh * 1000
            total_gpu_energy = tracker._total_gpu_energy.kWh * 1000
            total_energy = tracker._total_energy.kWh * 1000
            emissions_gCO2e = emissions * 1000
        except:
            pass
        
        os.environ["WANDB_SILENT"] = "True"
        wandb.log({
            "CPU (Wh)": total_cpu_energy,
            "RAM (Wh)": total_ram_energy,
            "GPU (Wh)": total_gpu_energy,
            "Energy (Wh)": total_energy,
            "Emissions (gCO2e)": emissions_gCO2e
        })
        wandb.finish()
        
        return result
    return wrapper



def load_video_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames


def save_image(image, save_path):
    cv2.imwrite(save_path, image)



def get_object_crops(frames, saliency_dir):
    object_crops = []

    for i, frame in enumerate(frames):
        saliency_path = os.path.join(saliency_dir, f"Saliency_{i:05d}.png")
        if not os.path.exists(saliency_path):
            print(f"Saliency map not found: {saliency_path}")
            continue

        saliency_map = cv2.imread(saliency_path, cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(saliency_map, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            crop = frame[y:y+h, x:x+w]
            object_crops.append(crop)

    return object_crops