"""
Train DL model

Written by Jatin Mathur and Ed Oughton.

Winter 2020

"""
import configparser
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, 'raw')
DATA_PROCESSED = os.path.join(BASE_PATH, 'processed')


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    """
    Function to train model.

    """
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = (running_loss /
                        len(dataloaders[phase].dataset))
            epoch_acc = (running_corrects.double() /
                        len(dataloaders[phase].dataset))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,
                epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    """
    Set parameter

    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract,
    use_pretrained=True):
    # Initialize these variables which will be set in this if
    # statement. Each of these variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes,
            kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

if __name__ == '__main__':

    print('Setting global parameters')
    DATA_IMAGES = os.path.join(BASE_PATH, '..', 'images', 'ims_mw')
    MODEL_NAME = "vgg"  #[resnet,alexnet,vgg,squeezenet,densenet,inception]
    NUM_CLASSES = 3 #set number of classes
    BATCH_SIZE = 2 #set batch size for
    NUM_EPOCHS = 2 #Number of epochs to train for
    # Flag for feature extracting. When False, we finetune the whole model,
    # when True we only update the reshaped layer params
    FEATURE_EXTRACT = True

    print('Initializing the model for this run')
    model_ft, input_size = initialize_model(MODEL_NAME, NUM_CLASSES,
        FEATURE_EXTRACT, use_pretrained=True)

    # Print the model we just instantiated
    print('This is the model:')
    print(model_ft)

    print('Data augmentation and normalization for training')
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    print('Creating training and validation datasets')
    image_datasets = {x: datasets.ImageFolder(
            os.path.join(DATA_IMAGES, x), \
            data_transforms[x]) for x in ['train', 'valid']}

    print('Creating training and validation dataloaders')
    dataloaders_dict = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=BATCH_SIZE, \
        shuffle=True, num_workers=4) for x in ['train', 'valid']}

    print('Detecting if we have a GPU available')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Sending model to GPU')
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Parameterss to learn:")
    if FEATURE_EXTRACT:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    print('Observe that all parameters are being optimized')
    optimizer_ft = optim.SGD(params_to_update, lr=1e-4, momentum=0.9)

    print('Setting up the loss fxn')
    criterion = nn.CrossEntropyLoss()

    print('Training and evaluating')
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion,
        optimizer_ft, num_epochs=NUM_EPOCHS)

    print('Saving model')
    torch.save(model_ft, os.path.join(DATA_PROCESSED, 'trained_model.pt'))
