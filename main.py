import torch.nn as nn
import os
import torch
from torchvision.io import read_image
import glob
from torch.utils.data import DataLoader
import logging
import argparse
import torchvision.transforms as transforms
from utils import get_transform
from tqdm import trange, tqdm
from torch.nn.parallel import DataParallel
import matplotlib.pyplot as plt

from model import Unet
from sklearn.model_selection import train_test_split
from dataset import AerialDroneDataset

logging.basicConfig(filename="training_log.txt", level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger(__name__)
# wandb_logger = WandbLogger(project="ImageClassificationPytochLightning", name="training", log_model="all")
def train(model, train_loader, val_loader,  optimizer, scheduler, criterion, epoch, device):
    # model = DataParallel(model, device_ids=[0, 1])
    model = model.to(device)     
    best_val_loss = float('inf')
    for i in trange(epoch, desc="Epoch"):
        training_loss = 0
        model.train()
        for img, mask in tqdm(train_loader):
            img, mask = img.to(device), mask.to(device)
            pred_mask = model(img)
            loss = criterion(pred_mask, mask)
            loss.backward()
            training_loss +=loss
            optimizer.zero_grad()
            optimizer.step()
            scheduler.step()
        log.info(f"training loss of epoch {i+1} is {training_loss/len(train_loader)}")
        if i%5==0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for val_img, val_mask in val_loader:
                    val_img, val_mask = val_img.to(device), val_mask.to(device)
                    val_pred_mask = model(val_img)
                    val_loss += criterion(val_pred_mask, val_mask)
                log.info(f"val loss is {val_loss/len(val_loader)}")
                if val_loss/len(val_loader) < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f"output/bestmodel_at_epoch{i}.ckpt")

def eval():
    pass
def visualize_prediction(test_image, pred):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(test_image.permute(1, 2, 0))  # Assuming test_image is a torch tensor
    axes[0].set_title('Test Image')

    axes[1].imshow(pred.squeeze(), cmap='gray')  # Assuming pred is a single-channel image
    axes[1].set_title('Prediction')

    for ax in axes:
        ax.axis('off')  # Turn off axis for both subplots

    plt.show()

def test(best_model, test_image):
    best_model.eval()
    test_transform = transforms.Resize((512, 512))
    transformed_test_image = test_transform(test_image).unsqueeze(dim=0)
    with torch.no_grad():
        _, pred = torch.max(best_model(transformed_test_image), dim=1)
    visualize_prediction(transformed_test_image, pred)

    return pred
        
    


def main():
    parser = argparse.ArgumentParser(description='Semantic classfication')
    parser.add_argument('-l', '--lr', default=0.0005)
    parser.add_argument('-b', '--batch_size', default=4)
    parser.add_argument('-e', '--num_epoch', default=50)
    parser.add_argument('-tr','--train', action='store_true')
    parser.add_argument('-t','--test', action='store_true')
    args = parser.parse_args()

    image_paths = glob.glob("Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset/original_images/*.jpg")
    image_names = [os.path.basename(image_path)[:-4] for image_path in image_paths]

    remaining_images, test_images = train_test_split(image_names, test_size=0.1, random_state=1)
    train_images, val_images = train_test_split(remaining_images, test_size=0.15, random_state=1)

    train_set = AerialDroneDataset(train_images, transform=get_transform(train=True))
    val_set = AerialDroneDataset(val_images, transform=get_transform(train=False))
    test_set = AerialDroneDataset(test_images, transform=get_transform(train=False))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    model = Unet() 

    # Optimize loss as the sum of IoU, Dice and CE, specifically this function:  IoU+Dice+0.8∗CE .
    # criterion = {
    #     "dice": DiceLoss(),
    #     "iou": IoULoss(),
    #     "bce": nn.CrossEntropyLoss()
    # }
    # import catalyst
    # from catalyst.contrib.nn import RAdam, Lookahead
    # from catalyst import utils
    # from torch import optim
    # learning_rate = 0.001

    # logdir = "./logs/segmentation"
    # model_params = utils.process_model_params(model)
    # base_optimizer = RAdam(model_params, lr=learning_rate, weight_decay=0.0003)
    # optimizer = Lookahead(base_optimizer)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)

    max_lr = 1e-3
    epoch = 15
    weight_decay = 1e-4

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                                steps_per_epoch=len(train_loader))
    # train(model, train_loader, val_loader,  optimizer, scheduler, criterion, epoch, device)

    test_image, _ = test_set[1]

    model.load_state_dict(torch.load("output/bestmodel_at_epoch10.ckpt"))
    pred = test(best_model=model, test_image=test_image)
    # Convert to the color mask using the RGB value for each from the csv file

    # Display the pred
if __name__ == "__main__":
    main()