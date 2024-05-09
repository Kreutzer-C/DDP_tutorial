from torch.utils.data import Dataset
from PIL import Image
import os

class ODIR_DataSet(Dataset):
    def __init__(self, img_dir, transform):
        self.img_list = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = os.path.join('./data', self.img_list[item])
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)

        return img