import os
from glob import glob
import random
from time import time
import torch
from Radiology_Dataset import RadiologyDataset
from model import VoxCNN
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchinfo import summary
from os.path import join


def test_function(model, test_loader, device):
    model.to(device=device)
    model.eval()
    with torch.no_grad():
        for imgs, _, subject_name in test_loader:
            imgs = imgs.to(device=device)
            outputs = model(imgs)
            prediction = torch.argmax(F.softmax(outputs), dim=1)
            # print("Predictions: ", predicted)
    return prediction.detach().cpu().numpy()[0], subject_name


def test_main(data_root: str, model_path: str, class_dict: dict):
    test_image_path = [data_root]
    test_set = RadiologyDataset(image_paths=test_image_path, labels=[0])
    test_loader = DataLoader(test_set, batch_size=1)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Testing on device {device}.")
    print("Initializing VoxCNN Model")
    model = VoxCNN(num_classes=len(class_dict.keys()))
    model.load_state_dict(torch.load(join(model_path, "EPI")+".pt"))
    print(f"Identifying")
    prediction, subject_name = test_function(model, test_loader, device)
    return prediction, subject_name


if __name__ == "__main__":
    DATA_PATH = "Data/Inference/ADNI_129_S_4396_MR_EPI_current_corrected_image_Br_20120421161123227_S140896_I299592_1.nii"
    MODEL_PATH = "Trained_Models"
    class_mapping = {0: "CN", 1: "EMCI", 2: "LMCI", 3: "AD"}
    pred, sub_name = test_main(DATA_PATH, MODEL_PATH, class_mapping)
    print("Subject Name:", sub_name)
    print("Prediction Value:", pred)
    print("Disease class:", class_mapping.get(pred))
