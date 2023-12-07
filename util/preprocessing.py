from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from PIL import Image
from imutils import paths
from imutils import *
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import functools

'''
    Please unzip your raw data in /root/plant_dataset/
    The directory structure should be as follows:

    plant_dataset
    ├── test       
    │   ├── test_label.csv 
    │   └── images         
    ├── train     
    │   ├── images        
    │   └── train_label.csv
    └── val       
        ├── val_label.csv  
        └── images

    This structure is important for the scripts to correctly locate and process the data.
'''


def resize_image(image_path, output_size, path):
    image = Image.open(image_path)
    resized_image = image.resize(output_size)
    output_path = path / Path(image_path).name
    resized_image.save(output_path)
    
def preprocessing(config):
    output_size = (config.INPUT_HEIGHT, config.INPUT_WIDTH)
    max_workers = 50
    def update_progress_bar(_):
        progress_bar.update(1)
    
    # ----------------train data----------------
    print("Resize the train data...")
    image_paths = list(paths.list_images(config.TRAIN_INIT_DIR))
    progress_bar = tqdm(total=len(image_paths), desc='Resizing train images')
    # 使用50个线程进行resize
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(resize_image, image_path, output_size, config.TRAIN_DIR) for image_path in image_paths]
        for future in concurrent.futures.as_completed(futures):
            future.add_done_callback(update_progress_bar)
    progress_bar.close()

    # ----------------test data----------------
    print("Resize the test data...")
    image_paths = list(paths.list_images(config.TEST_INIT_DIR))
    progress_bar = tqdm(total=len(image_paths), desc='Resizing test images')
    # 使用50个线程进行resize
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(resize_image, image_path, output_size, config.TEST_DIR) for image_path in image_paths]
        for future in concurrent.futures.as_completed(futures):
            future.add_done_callback(update_progress_bar)
    progress_bar.close()

    # ----------------valid data----------------
    print("Resize the valid data...")
    image_paths = list(paths.list_images(config.VAL_INIT_DIR))
    progress_bar = tqdm(total=len(image_paths), desc='Resizing valid images')
    # 使用50个线程进行resize
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(resize_image, image_path, output_size, config.VAL_DIR) for image_path in image_paths]
        for future in concurrent.futures.as_completed(futures):
            future.add_done_callback(update_progress_bar)
    progress_bar.close()

if __name__ == '__main__':
    preprocessing(config)