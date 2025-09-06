import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from utilities import load_data, train_network, test_accuracy, model_setup
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Train a new network on a dataset of images and saves the model to a checkpoint')

parser.add_argument('--data_dir', action = 'store', dest = 'data_directory', default = '../aipnd-project/flowers',
                    help = 'Set path to training data.')

parser.add_argument('--arch', action='store', default = 'vgg16',
                    help= 'Choose pretrained model: vgg16 or alexnet (default = vgg16)')

parser.add_argument('--save_dir', action = 'store',
                    dest = 'save_directory', default = 'checkpoint.pth',
                    help = 'Set checkpoint directory (default = checkpoint.pth')

parser.add_argument('--learning_rate', action = 'store',
                    dest = 'lr', type=int, default = 0.001,
                    help = 'Set learning rate (default = 0.001)')

parser.add_argument('--dropout', action = 'store',
                    dest='drpt', type=int, default = 0.05,
                    help = 'Set dropout rate (default = 0.05)')

parser.add_argument('--hidden_units', action = 'store',
                    dest = 'hidden_units', type=int, default = 512,
                    help = 'Set number of hidden units (default = 512)')

parser.add_argument('--epochs', action = 'store',
                    dest = 'num_epochs', type = int, default = 1,
                    help = 'Set number of epochs (default = 10)')

parser.add_argument('--gpu', action = "store_true", default = True,
                    help = 'Set GPU usage (default = True)')

results = parser.parse_args()
data_dir = results.data_directory
save_dir = results.save_directory
learning_rate = results.lr
dropout = results.drpt
hidden_units = results.hidden_units
epochs = results.num_epochs
gpu_mode = results.gpu
architecture = results.arch

# Load and process images
train_loader, valid_loader, test_loader, train_data, test_data, valid_data = load_data(data_dir)

#model architecture:
model, criterion, optimizer = model_setup(architecture, dropout, hidden_units, learning_rate, gpu_mode)

#train model
train_network(train_loader, valid_loader, model, criterion, optimizer, epochs, gpu_mode)

#test accuracy:
test_accuracy(model, test_loader)

#save checkpoint
model.cpu()
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'arch': architecture,
              'class_to_idx': train_data.class_to_idx
             }

torch.save (checkpoint, save_dir)

# arch = results.arch

# if arch == 'vgg16':
#     model = models.vgg16(pretrained=True)
#     classifier = nn.Sequential(OrderedDict([
#         ('fc1', nn.Linear(25088, hidden_units)),
#         ('relu', nn.ReLU()),
#         ('dropout1', nn.Dropout(dropout)),
#         ('fc2', nn.Linear(hidden_units, 102)),
#         ('output', nn.LogSoftmax(dim=1))
#     ]))

# elif arch == 'alexnet':
#     model = models.alexnet(pretrained=True)
#     classifier = nn.Sequential(OrderedDict([
#         ('fc1', nn.Linear(9216, 4096)),
#         ('relu1', nn.ReLU()),
#         ('dropout1', nn.Dropout(dropout)),
#         ('fc2', nn.Linear(4096, hidden_units)),
#         ('relu2', nn.ReLU()),
#         ('dropout2', nn.Dropout(dropout)),
#         ('fc3', nn.Linear(hidden_units, 102)),
#         ('output', nn.LogSoftmax(dim=1))
#     ]))

# # Freeze weights
# for param in model.parameters():
#     param.requires_grad = False

# # start training network
# model.classifier = classifier

# # optimizer
# optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

# # loss criterion
# criterion = nn.NLLLoss()

# # train the network
# model, optimizer = train_network(train_loader, validation_loader, model, criterion, optimizer, epochs=6, print_every=50)

# # testing accuracy
# test_model(model, test_loader)

# # Saving the model: The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary
# model.cpu()
# checkpoint = {'classifier': model.classifier,
#               'state_dict': model.state_dict (),
#               'arch': arch,
#               'class_to_idx': train_data.class_to_idx
#              }

# torch.save (checkpoint, save_dir)