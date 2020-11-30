from torch.utils.data import Dataset
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
from PIL import Image
import os


class EnumPairedDataset(Dataset):
    '''
        Dataset subclass for enumed paired images datasets (i.e images have
        same name on input and target and name is an integer).
    '''
    def __init__(self, x_root, y_root, transform=None, target_transform=None,
                 images_extension='png'):
        super(Dataset, self).__init__()
        self.x_root = x_root
        self.y_root = y_root
        self.transform = transform

        if self.transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), transform])

        self.target_transform = target_transform
        self.length = len(os.listdir(self.x_root))
        self.images_extension = images_extension
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = idx.tolist()

        x_path = os.path.join(self.x_root, f"{index}.{self.images_extension}")
        y_path = os.path.join(self.y_root, f"{index}.{self.images_extension}")
        x = Image.open(x_path)
        y = Image.open(y_path)
        if self.transform:
            x = self.transform(x)
            y = self.to_tensor(y)

        return x, y, f"{index}.{self.images_extension}"

    def __len__(self):
        return self.length

class SingleFolderDataset(Dataset):

    def __init__(self, root, transform=None, images_extension='png'):
        super(Dataset, self).__init__()
        self.root = root
        self.transform = transform
        self.images_extension = images_extension
        self.image_names = []
        for img_name in os.listdir(root):
            img_name = img_name.split('.')[0]
            if 'gt' not in img_name:
                if os.path.exists(os.path.join(root,
                                               f"{img_name}_gt.{self.images_extension}")):
                    self.image_names.append(img_name)

        if self.transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), transform])

    def __getitem__(self, index):
        name = self.image_names[index]
        if torch.is_tensor(index):
            index = idx.tolist()

        x_path = os.path.join(self.root, f"{name}.{self.images_extension}")
        y_path = os.path.join(self.root, f"{name}_gt.{self.images_extension}")
        x = Image.open(x_path)
        y = Image.open(y_path)
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y, os.path.join(self.root, name)

    def __len__(self):
        return len(self.image_names)


def enum_paired_dirs(*dirs, scale=None):
    '''
        Rename all files contained in directory paths
        in dirs, to numbers (e.g 1,2,3.jpg) so files can
        be loaded more eficiently (lazy loading). First path
        in dirs is used as reference. If scale is not None
        it is expected that for all directories but the first
        in dirs, files contained have x1 as part of its name
        and scale as part of name for files in first directory
        in dirs.
    '''
    assert len(dirs) > 1, "At least two directory paths should be provided"
    ref = dirs[0]
    for i, file_name in enumerate(os.listdir(ref)):
        file_name, file_extension = os.path.splitext(file_name)
        ref_path = os.path.join(ref, f"{file_name}{file_extension}")
        new_ref_path = os.path.join(ref, f"{i}{file_extension}")
        print(f"{ref_path} => {new_ref_path}")
        os.rename(ref_path, new_ref_path)
        if scale is not None:
            file_name = file_name.replace(scale, 'x1')
        for _dir in dirs[1:]:
            _path = os.path.join(_dir, f"{file_name}{file_extension}")
            _new_path = os.path.join(_dir, f"{i}{file_extension}")
            os.rename(_path, _new_path)
            print(f"{_path} => {_new_path}")

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

if __name__ == "__main__":
    ref_dir = '/home/rojas/datasets/real-world-super-resolution/Train_x2/train_HR'
    dir_1 = '/home/rojas/datasets/real-world-super-resolution/Train_x2/train_LR_samples'
    dir_2 = '/home/rojas/datasets/real-world-super-resolution/Train_x2/train_LR_bicubic_from_HR_samples'
    scale = 'x2'
    enum_paired_dirs(dir_1, dir_2)
