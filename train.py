import os
import numpy as np
from glob import glob
from pathlib import Path
import random
from time import time
import torch
from Radiology_Dataset import RadiologyDataset
from torch import nn as nn
from model import VoxCNN
from torch.utils.data import DataLoader
from torch import optim
from torchinfo import summary
from os.path import join


def train_function(epochs, optimizer, model, model_name, loss_func, train_loader, val_loader, device):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    val_loss_min = np.Inf
    model.to(device=device)
    print("Starting training")
    start_time = time()
    for e in range(epochs):
        train_loss = 0
        val_loss = 0
        train_total = 0
        train_correct = 0
        val_total = 0
        val_correct = 0
        model.train()
        for images, labels, subject_names in train_loader:
            images = images.to(device=device)
            labels = labels.to(device=device)
            # print("Input Image Shape: ", images.shape)
            # print("Labels shape: ", labels.shape)
            # print("Labels: ",labels)
            optimizer.zero_grad()
            pred = model(images)
#             print("Prediction Shape: ", pred.shape)
            _, predicted = torch.max(pred, dim=1)
            #             print("Predictions: ", predicted)
            #             predicted = torch.unsqueeze(predicted, dim=1)   # To make dims of predicted and labels equal
            loss = loss_func(pred, labels)
            loss.backward()
            optimizer.step()
            # .item returns the float value from loss tensor
            # we multiply with images.size(0) [which is the batch size] as loss contains the average loss for batch of images
            # and we want the total loss for batch of images
            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            # Calculate Accuracy
            train_total += batch_size
            # Count the number of matching values in predicted and labels tensors
            train_correct += int((predicted == labels).sum())
        model.eval()
        with torch.no_grad():
            for images, labels, subject_names in val_loader:
                images = images.to(device=device)
                labels = labels.to(device=device)
                outputs = model(images)
                _, predicted = torch.max(outputs, dim=1)
                # predicted = torch.unsqueeze(predicted, dim=1)
                batch_size = images.size(0)
                loss = loss_func(outputs, labels)
                val_loss += loss.item() * batch_size
                val_total += batch_size
                val_correct += int((predicted == labels).sum())

        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = val_loss / len(val_loader.dataset)
        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        print('Epoch {}, Training loss: {:.6f}, Validation Loss {:.6f}'.format(e + 1, train_loss, valid_loss))
        print('Training Accuracy: {:.3f}, Validation Accuracy: {:.3f}'.format(train_accuracy, val_accuracy))
        if valid_loss <= val_loss_min:
            print("Validation loss decreased({:.6f} --> {:.6f})".format(val_loss_min, valid_loss))
            val_loss_min = valid_loss
        # if val_accuracy >= 0.5:
            save_model_path = f"Trained_Models/{model_name}.pt"
            print("Saving trained model at ", save_model_path)
            torch.save(model.state_dict(), save_model_path)
            # save model
            # torch.save({
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            # }, 'checkpoint_epoch_' + str(e+1) + '.pt')
    end_time = time()
    total_time = (end_time - start_time) / 60.0
    print("Total time taken: ", total_time)
    return train_losses, val_losses, train_accs, val_accs


def get_train_val_files(class_root, ratio=0.2):
    all_files = glob(class_root + os.sep + "*.nii")
    random.shuffle(all_files)
    val_files = random.sample(all_files, int(round(ratio*len(all_files))))
    train_files = list(set(all_files).symmetric_difference(set(val_files)))
    assert len(train_files) + len(val_files) == len(all_files), f"Train files {len(train_files)} and Validation files {len(val_files)} do not add upto all files {len(all_files)}"
    return train_files, val_files


def train_main(data_root, epochs, batch_size, val_split, learning_rate, class_dict):
    # img_types = ["EPI", "FA", "MD"]
    img_types = ["FA"]
    train_dict = dict()
    print(f"Creating validation split with {val_split} * train data")
    for img_type in img_types:
        train_image_paths = []
        train_image_labels = []
        val_image_paths = []
        val_image_labels = []
        for class_type in class_dict.keys():
            folder_root = join(data_root, "ADNI2_"+img_type)
            print(f"Number of images in {img_type}/{class_type}: {len(os.listdir(join(folder_root, class_type)))}")
            train_dict[img_type] = dict()
            train_img_paths, val_img_paths = get_train_val_files(join(folder_root, class_type), val_split)
            print(f"Number of training images: {len(train_img_paths)}")
            print(f"Number of validation images: {len(val_img_paths)}")
            train_dict[img_type][class_type] = {"train": train_img_paths, "val": val_img_paths}
            train_image_paths += train_img_paths
            for i in range(len(train_img_paths)):
                train_image_labels.append(class_dict.get(class_type))
            val_image_paths += val_img_paths
            for i in range(len(val_img_paths)):
                val_image_labels.append(class_dict.get(class_type))
        assert len(train_image_paths) == len(train_image_labels), "Mismatch in Number of train images and train labels"
        assert len(val_image_paths) == len(val_image_labels), "Mismatch in Number of train images and train labels"
        print(f"Total Training images for {img_type}: {len(train_image_paths)}")
        print(f"Total Validation images for {img_type}: {len(val_image_paths)}")
        combined_train = list(zip(train_image_paths, train_image_labels))
        combined_val = list(zip(val_image_paths, val_image_labels))

        random.shuffle(combined_train)
        train_image_paths, train_image_labels = zip(*combined_train)
        train_image_paths, train_image_labels = list(train_image_paths), list(train_image_labels)

        random.shuffle(combined_val)
        val_image_paths, val_image_labels = zip(*combined_val)
        val_image_paths, val_image_labels = list(val_image_paths), list(val_image_labels)

        train_set = RadiologyDataset(image_paths=train_image_paths, labels=train_image_labels)
        val_set = RadiologyDataset(image_paths=val_image_paths, labels=val_image_labels)

        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=torch.cuda.is_available(),
                                  drop_last=True)

        val_loader = DataLoader(val_set,
                                batch_size=batch_size,
                                num_workers=4,
                                drop_last=True)

        device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        print(f"Training on device {device}.")
        # writer = SummaryWriter("runs/Fashion_mnist")
        print("Initializing VoxCNN Model")
        model = VoxCNN(num_classes=len(class_dict.keys()))
        model.to(device=device)
        print(summary(model, (2, 1, 128, 128, 128)))
        # loss_func = nn.NLLLoss()
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        print("Batch Size:", batch_size)
        print("Learning Rate", learning_rate)
        print(f"Training model to classify among {len(class_dict.keys())} classes")
        print(f"Starting training with a total of {epochs} epochs")
        train_losses, val_losses, train_accuracy, val_accuracy = train_function(epochs, optimizer, model, img_type,
                                                                       loss_func, train_loader, val_loader, device)
        print("Average Train loss: ", np.asarray(train_losses, dtype=np.float32).mean())
        print("Average Validation loss: ", np.asarray(val_losses, dtype=np.float32).mean())
        print("Average Train Accuracy: ", np.asarray(train_accuracy, dtype=np.float32).mean())
        print("Average Validation Accuracy: ", np.asarray(val_accuracy, dtype=np.float32).mean())


if __name__ == "__main__":
    data_dict = {"CN": 0, "EMCI": 1, "LMCI": 2, "AD": 3}
    train_main("Data/Train", epochs=50, batch_size=32, val_split=0.2, learning_rate=0.0000027, class_dict=data_dict)