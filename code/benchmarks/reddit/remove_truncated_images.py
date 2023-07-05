from PIL import Image
import os

def is_file_truncated(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return False
    except (IOError, SyntaxError):
        return True

def remove_truncated_files(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if is_file_truncated(file_path):
            os.remove(file_path)
            print(f"Removed truncated file: {file_name}")

folder_path = '/datasets/images'
remove_truncated_files(folder_path)
