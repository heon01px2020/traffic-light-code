import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from PIL import ImageFile, Image
from torchvision import transforms
import random

class TrafficLightDataset(Dataset):
    
    def __init__(self, csv_file, root_dir, transformation = True):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transformation = transformation

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_name = os.path.join(self.root_dir, self.labels.iloc[index, 0])
        image = Image.open(img_name)
        
        
        
        light_mode = self.labels.iloc[index, 1] #mode of the traffic light
        
        block = self.labels.iloc[index,6] #the blocked label
        
        points = self.labels.iloc[index, 2:6] #the coordinates for direction vector start and endpoints

        points = [points[0]/4032, points[1]/3024, points[2]/4032, points[3]/3024] #normalize vector
        
        if self.transformation:
            num = random.random()
            if num >= 0.5:
                image = transforms.functional.hflip(image)
                points[0] = 1-points[0]
                points[2] = 1- points[2]
        
        image = np.transpose(image, (2, 0, 1))
        points = torch.tensor(points)
        #combine all into a dictionary
        final_label = {'image': image, 'mode':light_mode, 'points': points, 'block': block}
        
        #if we decide to incorporate transformations in the future
        

        return final_label