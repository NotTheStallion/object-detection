from utils import *
import os

OBJECTS = ["Bowl", "CanOfCocaCola", "Jam", "MilkBottle", "Mug", "OilBottle", "Rice", "Sugar", "VinegarBottle"]

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
def create_data(set="train"):
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


if __name__ == "__main__":
    create_data("train")
    create_data("test")
