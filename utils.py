
import os

import wandb



from codecarbon import EmissionsTracker
import cv2
import re

import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers


def parse_bboxes_file(bboxes_file):
    bboxes = []
    try:
        with open(bboxes_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                frame_index, class_id = int(parts[0]), int(parts[1])
                if len(parts) > 2:
                    x_min, y_min, width, height = map(int, parts[2:])
                    x_max = x_min + width
                    y_max = y_min + height
                    bboxes.append({
                        'frame_index': frame_index,
                        'class_id': class_id,
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max
                    })
    except Exception as e:
        print(f"[ERREUR] Impossible de lire {bboxes_file} : {e}")
    return bboxes

def extract_crops_from_video(video_path, bboxes, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    for bbox in bboxes:
        frame_index = bbox['frame_index']
        class_id = bbox['class_id']
        x_min, y_min, x_max, y_max = bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            continue

        crop = frame[y_min:y_max, x_min:x_max]
        class_dir = output_dir
        os.makedirs(class_dir, exist_ok=True)

        crop_name = f"frame_{frame_index:04d}_crop_{video_path.split("/")[3]}.jpg"
        crop_path = os.path.join(class_dir, crop_name)
        cv2.imwrite(crop_path, crop)


    cap.release()

def find_matching_txt(video_file, bboxes_dir):
    # On extrait le nom sans extension
    base_name = os.path.splitext(video_file)[0]
    pattern = rf"^{re.escape(base_name)}.*_bboxes\.txt$"

    for file in os.listdir(bboxes_dir):
        if re.match(pattern, file):
            return os.path.join(bboxes_dir, file)
    return None

def creata_bbox_data(videos_root, bboxes_root, output_root):
    for dirpath, _, files in os.walk(videos_root):
        for i, file in enumerate(files):
            if file.endswith('.mp4'):
                video_path = os.path.join(dirpath, file)
                relative_path = os.path.relpath(dirpath, videos_root)
                bboxes_dir = os.path.join(bboxes_root, relative_path)


                if os.path.exists(bboxes_dir):
                    bbox_file = find_matching_txt(file, bboxes_dir)
                    if bbox_file:
                        # print(output_root)
                        # print("/".join(relative_path.split("/")[:2]))
                        # print(file)
                        output_dir = os.path.join(output_root, "/".join(relative_path.split("/")[:2]))
                        # print(output_dir)
                        bboxes = parse_bboxes_file(bbox_file)
                        if bboxes:
                            extract_crops_from_video(video_path, bboxes, output_dir)



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
        _, thresh = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            crop = frame[y:y+h, x:x+w]
            object_crops.append(crop)

    return object_crops




def get_all_directories(root, objects):
    directories = {}
    for obj in objects:
        obj_path = os.path.join(root, obj)
        if os.path.isdir(obj_path):
            directories[obj] = [os.path.join(obj_path, d) for d in os.listdir(obj_path) if os.path.isdir(os.path.join(obj_path, d))]
    return directories


def get_mp4_paths(directories):
    mp4_paths = {}
    for obj, dirs in directories.items():
        mp4_paths[obj] = []
        for dir in dirs:
            for file in os.listdir(dir):
                if file.endswith(".mp4"):
                    mp4_paths[obj].append(os.path.join(dir, file))
    return mp4_paths


@energy_profiler
def create_saliency_data(set="train"):
    root = os.path.join("GITW_light", set)
    directories = get_all_directories(root, OBJECTS)
    mp4_paths = get_mp4_paths(directories)

    for obj, paths in mp4_paths.items():
        for path in paths:
            # print(path)
            frames = load_video_frames(path)
            # print(f"Number of frames: {len(frames)}")
            saliency_dir = os.path.join(os.path.dirname(path), "SaliencyMaps")
            object_crops = get_object_crops(frames, saliency_dir)
            # print(f"Number of object crops: {len(object_crops)}")
            
            
            save_dir = os.path.join(os.path.join("data", set), obj)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for i, crop in enumerate(object_crops):
                save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(path))[0]}_crop_{i}.jpg")
                save_image(crop, save_path)



OBJECTS = ["Bowl", "CanOfCocaCola", "Jam", "MilkBottle", "Mug", "OilBottle", "Rice", "Sugar", "VinegarBottle"]



class CustomVGG19Model:
    def __init__(self, num_classes, input_shape, apply_augmentation=False):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.apply_augmentation = apply_augmentation
        self.model = self.build_model()

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        if self.apply_augmentation:
            layer = layers.RandomFlip('horizontal')(inputs)
            layer = layers.RandomRotation(0.1)(layer)
            layer = layers.RandomZoom(0.2)(layer)
        else:
            layer = inputs
        layer = layers.Rescaling(1/255)(layer)

        base_model = VGG19(weights='imagenet', include_top=False, input_tensor=layer)
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def summary(self):
        self.model.summary()
    
    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

if __name__ == "__main__":
    
    videos_root = "GITW_light"
    bboxes_root = "GITW_light_bboxes"
    output_root = "bboxes"

    creata_bbox_data(videos_root, bboxes_root, output_root)
    create_saliency_data("train")
    create_saliency_data("test")