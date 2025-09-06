import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

def load_data(data_dir="./flowers" ):
#     train_dir = data_dir + '/train'
#     valid_dir = data_dir + '/valid'
#     test_dir = data_dir + '/test'

       
    train_transforms = transforms.Compose([transforms.RandomRotation(60),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_test_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])
                                             ])
        

       
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=valid_test_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_test_transforms)

       
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    return train_loader , valid_loader, test_loader, train_data, test_data, valid_data

def model_setup(architecure='vgg16', dropout=0.5, hidden_units=600, learning_rate=0.001, gpu_mode = True):

    
    architecures = { "vgg16":25088,
                        "alexnet":9216 }
        
    if architecure == 'vgg16':
            model = models.vgg16(pretrained=True)
    elif architecure == 'alexnet':
            model = models.alexnet(pretrained = True)
    else:
            print("Not a valid architecure. Choose vgg16 or alexnet")
            
    if gpu_mode == True:
        model.to('cuda')
    else:
        pass

        # the weights of the pretrained model are frozen to avoid backpropping through them
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout1', nn.Dropout(0.05)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
            ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    if torch.cuda.is_available() and gpu_mode == True:
        model.cuda()

    return model, criterion, optimizer

def test_accuracy(model, test_loader):
    
    test_accuracy = 0

    for images,labels in test_loader:
        model.eval()
        images,labels = images.to('cuda'),labels.to('cuda')
        log_ps = model.forward(images)
        ps = torch.exp(log_ps)
        top_ps,top_class = ps.topk(1,dim=1)
        matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
        accuracy = matches.mean()
        test_accuracy += accuracy

    print(f'Test dataset accuracy: {test_accuracy/len(test_loader)*100:.2f}%')


def train_network(train_loader, valid_loader, model, criterion, optimizer, epochs=1, gpu_mode = True):
    
    if gpu_mode == True:
        model.to('cuda')
    else:
        pass
    epochs = 6
    learning_rate = 1e-3
    optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)
    criterion = nn.NLLLoss() # recommended when using Softmax

    print_every = 50 # model trains on 50 batches of images at a time

    running_loss = 0
    running_accuracy = 0
    validation_loss = []
    training_loss = []

    # Switch to GPU (cuda)
    model.to('cuda')

    # defines the training process
    for e in range(epochs):
        
        batches = 0 


        model.train()

        for images,labels in train_loader:
            batches += 1

            # images and labels to the GPU (cuda)
            images,labels = images.to('cuda'),labels.to('cuda')

            # batch through network
            log_ps = model.forward(images)
            loss = criterion(log_ps,labels)
            loss.backward()
            optimizer.step()

            #  metrics
            ps = torch.exp(log_ps)
            top_ps, top_class = ps.topk(1,dim=1)
            matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
            accuracy = matches.mean()

            # reset optimiser gradient and tracks metrics
            optimizer.zero_grad()
            running_loss += loss.item()
            running_accuracy += accuracy.item()

            # run the model on the validation set
            if batches%print_every == 0:

                # metrics
                validation_loss = 0
                validation_accuracy = 0

                # turns on evaluation mode, turns off calculation of gradients
                model.eval()
                with torch.no_grad():
                    for images,labels in valid_loader:
                        images,labels = images.to('cuda'),labels.to('cuda')
                        log_ps = model.forward(images)
                        loss = criterion(log_ps,labels)
                        ps = torch.exp(log_ps)
                        top_ps, top_class = ps.topk(1,dim=1)
                        matches = (top_class == \
                                    labels.view(*top_class.shape)).type(torch.FloatTensor)
                        accuracy = matches.mean()

                        # validation metrics
                        validation_loss += loss.item()
                        validation_accuracy += accuracy.item()



                print(f'Epoch {e+1}/{epochs} | Batch {batches}')
                print(f'Running Training Loss: {running_loss/print_every:.3f}')
                print(f'Running Training Accuracy: {running_accuracy/print_every*100:.2f}%')
                print(f'Validation Loss: {validation_loss/len(valid_loader):.3f}')
                print(f'Validation Accuracy: {validation_accuracy/len(valid_loader)*100:.2f}%')

                # resets the metrics and turns on training mode
                running_loss = running_accuracy = 0
                model.train()

                
        return model, optimizer



