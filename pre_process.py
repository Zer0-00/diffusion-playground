import torch
from torchvision.transforms import transforms as T
from PIL import Image
import numpy as np
import os
import csv
from utils import create_folders
from tqdm import tqdm

class image_processor():
    def __init__(
        self,
        image_size = [256,256],
        transforms = None
    ):
        if transforms is None:
            self.transforms = T.Compose([
                T.Resize(image_size),
                T.Grayscale(),
                T.ToTensor(),
            ])
        else:
            self.transforms = transforms
    def process(self, data_dir, output_dir):
        image = Image.open(data_dir)
        processed_image = self.transforms(image)
        with open(output_dir, 'wb') as f:
            np.save(f, processed_image)
            
def process_chexpert(data_path, label_dir, output_path, image_size = 256, transforms=None):
    processor = image_processor(image_size=image_size, transforms=transforms)
    
    for class_name in ["healthy", "pleural effusions"]:
        create_folders(os.path.join(output_path, class_name))
    
    #count total number of lines
    total_num = 0
    with open(label_dir, 'r') as lf:
        reader = csv.DictReader(lf)
        for row in reader:
            total_num += 1
        
    def extract_output_name(img_dir):
        dirs = img_dir.split('/')
        name = "{}_{}.npy".format(dirs[2], dirs[3])
        return name
        
    def adapt_path(origin_path):
        dirs = origin_path.split('/')
        output_dir = os.path.join(dirs)
        return output_dir

    
    with open(label_dir, 'r') as lf:
        reader = csv.DictReader(lf)
        reader = tqdm(reader, total=total_num, unit="pics")
        
        for row in reader:
            if row["Frontal/Lateral"] == "Lateral":
                continue
            if float(row["No Finding"]) == 1:
                img_dir = os.path.join(data_path, adapt_path(row["Path"]))
                output_dir = os.path.join(output_path, "healthy", extract_output_name(row["Path"]))
                processor.process(img_dir, output_dir)                
            elif float(row["Pleural Effusion"]) == 1:
                img_dir = os.path.join(data_path, adapt_path(row["Path"]))
                output_dir = os.path.join(output_path, "pleural effusions", extract_output_name(row["Path"]))
                processor.process(img_dir, output_dir)
    
if __name__ == "__main__":
    data_path = '..'
    label_dir = os.path.join(data_path, "train.csv")


                