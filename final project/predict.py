import argparse
import json
import numpy as np
import torch
from torchvision import datasets, transforms, models
from PIL import Image
# from utilities import load_data, predict

parser = argparse.ArgumentParser(description='Path to an image and a checkpoint, returns top K most probably classes for the image')

parser.add_argument('--image_dir', action='store',
                    default = '../aipnd-project/flowers/test/2/image_05100',
                    help='Set path to image (default = ../aipnd-project/flowers/test/100/image_07896)')

parser.add_argument('--save_dir', action='store', default = 'checkpoint.pth',
                    help='Checkpoint location (default = checkpoint.pth)')

parser.add_argument('--topk', action='store',
                    dest='topk', type=int, default = 5,
                    help='Set K flower class (default = 5)')

parser.add_argument('--cat_to_name', action='store',
                    dest='cat_name_dir', default = 'cat_to_name.json',
                    help='Mapping from category label to category name (default = cat_to_name.json')

parser.add_argument('--gpu', action="store_true", default=True,
                    help='GPU usage, CUDA (default = True)')

results = parser.parse_args()

image_dir = results.image_dir
topk = results.topk
cat_names = results.cat_name_dir
gpu_mode = results.gpu
save_dir = results.save_dir

#load checkpoint

def load_checkpoint(path):
    checkpoint = torch.load('checkpoint.pth')
    model = models.vgg16(pretrained=True)
    model.state_dict (checkpoint['state_dict'])
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

model = load_checkpoint('checkpoint.pth')  

####
    
# checkpoint = torch.load(save_dir)
# if checkpoint['arch'] == 'vgg16':
#     model = models.vgg16(pretrained=True)

# elif checkpoint['arch'] == 'alexnet':
#     model = models.alexnet(pretrained=True)

# model.state_dict (checkpoint['state_dict'])
# model.classifier = checkpoint['classifier']
# model.class_to_idx = checkpoint['class_to_idx']
    
# for param in model.parameters():
#     param.requires_grad = False


# with open('cat_to_name.json', 'r') as f:
#     cat_to_name = json.load(f)

# image processing
def process_image(image_dir):
    pil_image = Image.open(f'{image_dir}' + '.jpg')

    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

    pil_transform = transform(pil_image)

    np_image = np.array(pil_transform)

    img_tensor = torch.from_numpy(np_image).type(torch.FloatTensor)

    processed_image = img_tensor.unsqueeze(0)

    return processed_image

# predict class

###########
def predict(image_path, model, topk=5, gpu_mode='gpu'):
    if torch.cuda.is_available() and gpu_mode == 'gpu':
            model.to('cuda:0')

    img_torch = Util.process_image(image_path).unsqueeze_(0).float()
        
    if gpu_mode == 'gpu':
            with torch.no_grad():
                output = model.forward(img_torch.cuda())
    else:
            with torch.no_grad():
                output=model.forward(img_torch)
            
    probability = F.softmax(output.data,dim=1)
        
    return probability.topk(topk)
#############

# def predict(image_dir, model, topk, gpu_mode):
    
#     image = process_image(image_dir)

#     # Convert processed image to CUDA tensor
#     if gpu_mode == True:
#         model.to('cuda')
#     else:
#         model.cpu()
        
#     if gpu_mode == True:
#         image = image.to('cuda')
#     else:
#         pass

#     # predict the class from an image file
    
#     loaded_model = load_checkpoint(model).cpu()
    
#     img = process_image(image_dir)
    
#     img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    
#     # batch size argument
#     model_input = img_tensor.unsqueeze(0)
    
#     # Calculate probability
#     loaded_model.eval()
#     with torch.no_grad():
#         model_output = loaded_model.forward(model_input)
    
#     probs = torch.exp(model_output)
#     probs_topk = probs.topk(k)[0]
#     idx_topk = probs.topk(k)[1]
    
#     # Convert stored top probabilities to Numpy arrays
#     probs_topk_array = np.array(probs_topk)[0]
#     idx_topk_array = np.array(idx_topk)[0]
    
#     class_to_idx = loaded_model.class_to_idx
    
#     # Invert the dictionary to obtain mapping from index to class
#     idx_to_class = {val: key for key, val in class_to_idx.items()}
#     class_topk_array = []
#     for idx in idx_topk_array:
#         class_topk_array +=[idx_to_class[idx]]
        
#     return probs_topk_array, class_topk_array

    
# # Path to image and model checkpoint
# # image_path = data_dir + '/test/100/image_07896'
# # model_path = 'checkpoint.pth' 

# # Predict and print the probabilities and classes
# probs, classes = predict(image_dir, model,  topk, gpu_mode)
# print(probs)
# print(classes)