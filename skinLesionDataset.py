import torch
from torch.utils.data import Dataset

class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.image_extensions = [".jpeg", ".jpg", ".png"]

    def __len__(self):
        return len(self.df)

    def find_image_path(self, image_id):
        for ext in self.image_extensions:
            img_path = os.path.join(self.img_dir, f"{image_id}{ext}")
            if os.path.exists(img_path):
                return img_path
        raise FileNotFoundError(f"Image file not found for ID: {image_id}")

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]["image_id"]
        img_name = self.find_image_path(image_id)
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = {
            "super_class": torch.tensor(self.df.iloc[idx]["super_class"], dtype=torch.long),
            "malignancy": torch.tensor(self.df.iloc[idx]["malignancy"], dtype=torch.long),
            "main_class_1": torch.tensor(self.df.iloc[idx]["main_class_1"], dtype=torch.long),
            "main_class_2": torch.tensor(self.df.iloc[idx]["main_class_2"], dtype=torch.long),
            "sub_class": torch.tensor(self.df.iloc[idx]["sub_class"], dtype=torch.long),
        }

        return image, labels