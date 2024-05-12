from torchvision.io import read_image
from torchvision import tv_tensors
import torchvision.transforms.functional as TF

class AerialDroneDataset:
    def __init__(self, image_names, transform) -> None:
        self.transform = transform
        self.image_names = image_names

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self,idx):
        img = read_image("Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset/original_images/" + self.image_names[idx] + ".jpg")
        mask = read_image("Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset/label_images_semantic/" + self.image_names[idx] + ".png")
                # Wrap sample and targets into torchvision tv_tensors:
        desired_image_size = (512, 512)  # Desired size for the image
        img = TF.resize(img, desired_image_size)
        mask = TF.resize(mask, desired_image_size)
        if self.transform is not None:
            img, mask = self.transform(img.float(), mask.float())
        return img, mask.squeeze().long()