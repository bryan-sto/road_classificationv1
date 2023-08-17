import os
import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import json
import multiprocessing
from PIL import Image

class_names = ['Baik', 'Buruk', 'Sedang']
output_folder = 'final_cnn/output_folder'
validation_dir = 'final_cnn/predict'

def load_model_and_predict(model_path, img_path):
    model = models.resnet152(weights='IMAGENET1K_V2')
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 3)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(img_path)
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)

    _, predicted = torch.max(outputs, 1)
    predicted_class = predicted.item() + 1

    return predicted_class

def process_image(img_file):
    img_path = os.path.join(validation_dir, img_file)
    predicted_class = load_model_and_predict('final_cnn/model/0.8384_model_ft_time17m_epoch32.pt', img_path)
    predicted_class_label = class_names[predicted_class - 1]
    return img_path, predicted_class_label

def generate_video(video_filename, predicted_outputs):
    image_folder = 'final_cnn/predict'
    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith("png")]

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'h264') 
    video = cv2.VideoWriter(video_filename, fourcc, 3, (width, height))

    for img_path, predicted_label in predicted_outputs:
        frame = cv2.imread(img_path)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

def main(street_name, image_folder):
    os.makedirs(output_folder, exist_ok=True)

    class_distribution_filename = os.path.join(output_folder, f'class_distribution_{street_name}.json')
    video_filename = os.path.join(output_folder, f'generated_video_{street_name}.mp4')

    validation_data = os.listdir(image_folder)

    num_images = len(validation_data)
    class_distribution = {'baik': 0, 'buruk': 0, 'sedang': 0}

    num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_workers)
    predicted_outputs = pool.map(process_image, validation_data)
    pool.close()
    pool.join()

    for img_path, predicted_label in predicted_outputs:
        class_distribution[predicted_label] += 1

    total_images = num_images
    class_distribution_percent = {
        cls: f'{round(count / total_images * 100, 2)}%' for cls, count in class_distribution.items()
    }
    
    class_distribution_json = {
        "nama_jalan": street_name,
        **class_distribution_percent
    }

    print("Class Distribution (Percentage):")
    print(json.dumps(class_distribution_json, indent=4))

    with open(class_distribution_filename, 'w') as outfile:
        json.dump(class_distribution_json, outfile, indent=4)

    print(f"Class distribution analysis completed. Results saved as '{class_distribution_filename}'.")

    print(f"Generating video for jalan '{street_name}'...")
    generate_video(video_filename, predicted_outputs)
    print(f"Video generated for jalan '{street_name}' and saved as '{video_filename}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate class distribution and video for a street.")
    parser.add_argument("street_name", type=str, help="Name of the street")
    parser.add_argument("image_folder", type=str, help="Path to the folder containing images")
    args = parser.parse_args()
    main(args.street_name, args.image_folder)
    

