import os
import cv2
import re

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
        print(f"[INFO] Bounding boxes trouvées dans {bboxes_file} : {len(bboxes)}.")
    except Exception as e:
        print(f"[ERREUR] Impossible de lire {bboxes_file} : {e}")
    return bboxes

def extract_crops_from_video(video_path, bboxes, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la vidéo : {video_path}")
        return

    for bbox in bboxes:
        frame_index = bbox['frame_index']
        class_id = bbox['class_id']
        x_min, y_min, x_max, y_max = bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(f"[ERREUR] Frame {frame_index} introuvable dans {video_path}.")
            continue

        crop = frame[y_min:y_max, x_min:x_max]
        class_dir = os.path.join(output_dir, f"class_{class_id}")
        os.makedirs(class_dir, exist_ok=True)

        crop_name = f"frame_{frame_index:04d}_crop.jpg"
        crop_path = os.path.join(class_dir, crop_name)
        cv2.imwrite(crop_path, crop)

    cap.release()
    print(f"[INFO] Extraction terminée pour : {video_path}")

def find_matching_txt(video_file, bboxes_dir):
    # On extrait le nom sans extension
    base_name = os.path.splitext(video_file)[0]
    pattern = rf"^{re.escape(base_name)}.*_bboxes\.txt$"

    for file in os.listdir(bboxes_dir):
        if re.match(pattern, file):
            return os.path.join(bboxes_dir, file)
    return None

def process_videos_and_bboxes(videos_root, bboxes_root, output_root):
    for dirpath, _, files in os.walk(videos_root):
        for i, file in enumerate(files):
            if file.endswith('.mp4'):
                video_path = os.path.join(dirpath, file)
                relative_path = os.path.relpath(dirpath, videos_root)
                bboxes_dir = os.path.join(bboxes_root, relative_path)

                print(f"[DEBUG] Chemin vidéo : {video_path}")
                print(f"[DEBUG] Dossier bounding box : {bboxes_dir}")

                if os.path.exists(bboxes_dir):
                    bbox_file = find_matching_txt(file, bboxes_dir)
                    if bbox_file:
                        print(output_root)
                        print("/".join(relative_path.split("/")[:2]))
                        print(file)
                        output_dir = os.path.join(output_root, "/".join(relative_path.split("/")[:2]))
                        print(output_dir)
                        print(f"[INFO] Traitement de la vidéo : {video_path}")
                        bboxes = parse_bboxes_file(bbox_file)
                        if bboxes:
                            extract_crops_from_video(video_path, bboxes, output_dir)
                        else:
                            print(f"[ERREUR] Pas de bounding boxes valides dans {bbox_file}.")
                    else:
                        print(f"[INFO] Aucun fichier de bounding boxes associé pour : {video_path}. Vidéo ignorée.")
                else:
                    print(f"[INFO] Aucun dossier de bounding boxes trouvé pour : {video_path}. Vidéo ignorée.")

# Exemple d'utilisation
videos_root = "GITW_light"
bboxes_root = "GITW_light_bboxes"
output_root = "bboxes"

process_videos_and_bboxes(videos_root, bboxes_root, output_root)
