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
    total = 0
    correct = 0
    model.to(device=device)
    model.eval()
    ground_truth = list()
    predictions = list()
    subjects = list()
    with torch.no_grad():
        for imgs, labels, subject_names in test_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            # print("Ground Truth:", labels)
            ground_truth.append(labels)
            _, predicted = torch.max(outputs, dim=1)
            # print("Predictions: ", predicted)
            predictions.append(F.softmax(outputs))
            total += labels.shape[0]
            correct += int((predicted == labels).sum())
            subjects.extend(subject_names)
    test_accuracy = correct / total
    print("Test Accuracy: ", test_accuracy)
    return ground_truth, predictions, subjects


def test_main(data_root: str, model_path: str, batch_size: int, class_dict: dict):
    img_types = ["EPI", "FA", "MD"]
    test_dict = dict()
    for img_type in img_types:
        test_image_paths = []
        test_image_labels = []
        for class_type in class_dict.keys():
            folder_root = join(data_root, "ADNI2_" + img_type)
            print(f"Number of images in {img_type}/{class_type}: {len(os.listdir(join(folder_root, class_type)))}")
            test_dict[img_type] = dict()
            test_img_paths = glob(join(folder_root, class_type) + os.sep + "*.nii")
            test_dict[img_type][class_type] = {"test": test_img_paths}
            test_image_paths += test_img_paths
            for i in range(len(test_img_paths)):
                test_image_labels.append(class_dict.get(class_type))
        assert len(test_image_paths) == len(test_image_labels), "Mismatch in Number of train images and train labels"
        combined_train = list(zip(test_image_paths, test_image_labels))

        random.shuffle(combined_train)
        test_image_paths, test_image_labels = zip(*combined_train)
        test_image_paths, test_image_labels = list(test_image_paths), list(test_image_labels)

        test_set = RadiologyDataset(image_paths=test_image_paths, labels=test_image_labels)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

        device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        print(f"Testing on device {device}.")
        # writer = SummaryWriter("runs/Fashion_mnist")
        print("Initializing VoxCNN Model")
        model = VoxCNN(num_classes=len(class_dict.keys()))
        model.load_state_dict(torch.load(join(model_path, img_type)+".pt"))
        model.to(device=device)
        print(summary(model, (2, 1, 128, 128, 128)))
        print(f"Starting testing")
        ground_truths, predictions, subject_names = test_function(model, test_loader, device)
        return ground_truths, predictions, subject_names
