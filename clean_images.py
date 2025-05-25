import os
from PIL import Image

def clean_and_convert_images(folder_path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    count = 1

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                new_name = f"img_{count:04d}.jpg"
                new_path = os.path.join(folder_path, new_name)
                img.save(new_path, "JPEG")
                if img_path != new_path:
                    os.remove(img_path)
                count += 1
        except Exception as e:
            print(f"Deleted corrupted image: {img_name}")
            os.remove(img_path)

folder = r'D:\human vs Ai\dataset\train\human'

clean_and_convert_images(folder)
