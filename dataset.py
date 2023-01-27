from torch.utils.data import Dataset
import os
from torchvision import transforms as T
import torch
import skimage.io as io
import numpy as np

class MVtec_Leather(Dataset):
    def __init__(self, data_dir,
                 anomalous = False, 
                 img_size = (256,256),
                 rgb = True,
                 prepare = ("random_crop"),
                 include_good = True,
    ):
        self.data_dir = data_dir
        self.anomalous = anomalous
        self.rgb = rgb
        self.img_size = img_size
        self.prepare = prepare
        self.include_good = include_good
        self.classes = ["color", "cut", "fold", "glue", "poke"]
        
        if not self.anomalous:
            self.data_dir = os.path.join(self.data_dir, "train", "good")
        
        #set transformation
        transform_list = [T.ToPILImage()]
        
        if not self.rgb:
            self.channel = 1
            transform_list.append(T.Grayscale(num_output_channels=self.channel))
        else:
            self.channel = 3
        
        transform_list.append(T.ToTensor())
        

        
        # normalize_factor = ((0.5,)*self.channel, (0.5,)*self.channel)
        # transform_list.append(T.Normalize(*normalize_factor))
        
        self.transform = T.Compose(transform_list)
        
        
        prepare_list = []
        if "random_crop" in self.prepare:
            prepare_list.append(T.RandomCrop(self.img_size))
        else:
            prepare_list.append(T.Resize(self.img_size))
        
        if "random_rotation" in self.prepare:
            prepare_list.append(T.RandomRotation(180))
            
        self.preparation = T.Compose(prepare_list)
            
        #process filenames
        if self.include_good:
            self.classes.append("good")
        
        if self.anomalous:
            mask_transform_list = [
            T.ToPILImage(),
            T.Grayscale(num_output_channels=1),
            T.ToTensor()
            ]
            self.mask_transform = T.Compose(mask_transform_list)
            self.filenames = []
            for cl in self.classes:
                class_dir = os.path.join(self.data_dir, "test", cl) 
                self.filenames += [os.path.join(class_dir,file_name) for file_name in os.listdir(class_dir) if file_name.endswith(".png")]
        else:
            self.filenames = [os.path.join(self.data_dir,file_name) for file_name in os.listdir(self.data_dir) if file_name.endswith(".png")]
        
        self.filenames = sorted(self.filenames, key = lambda x: int(x[-7:-4]))    
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        output = {"filename":self.filenames[idx]}
        
        img = io.imread(self.filenames[idx])
        
        #get mask if needed
        if self.anomalous:
            name_split = self.filenames[idx].split(os.sep)
            if name_split[-2] == "good":
                mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
            else:
                mask = io.imread(os.path.join(self.data_dir, "ground_truth", name_split[-2], name_split[-1][:-4]+ "_mask.png"))

            mask = self.mask_transform(mask)
            
        #prepare image and mask        
        img = self.transform(img)
        
        
        
        #pre-prepare image and mask
        if self.anomalous:
            cat_img = torch.cat((img, mask), 0)
            prepared_img = self.preparation(cat_img)
            
            output["input"] = prepared_img[:self.channel]
            output["mask"] = prepared_img[self.channel:]
        else:
            prepared_img = self.preparation(img)
            output["input"] = prepared_img
        
        return output

class CheXpert(Dataset):
    def __init__(
        self,
        data_dir,
        anomalous = False
    ):  
        self.data_dir = data_dir
        self.anomalous = anomalous
        
        self.health_path = os.path.join(data_dir,"healthy")
        self.image_dirs = [os.path.join(self.health_path, image_dir) for image_dir in os.listdir(self.health_path)]
        
        if self.anomalous:
            self.y = [0]*len(self.health_path)
            self.anomaly_path = os.path.join(data_dir,"pleural effusions")
            self.image_dirs += [os.path.join(self.anomaly_path, image_dir) for image_dir in os.listdir(self.anomaly_path)]
            self.y += [1]*len(self.anomaly_path)
            
            
        self.transforms = T.Compose([])
        
    def __len__(self):
        return len(self.image_dirs)
    
    def __getitem__(self, idx):
        image_dir = self.image_dirs[idx]
        
        #the dataset has been changed to (C,H,W)
        image = torch.Tensor(np.load(image_dir))
        
        image = self.transforms(image)
        
        output = {
            "input": image,
            "name" : image_dir
        }
        
        if self.anomalous:
            output += {"y": self.y[idx]}
        
        return output
