#imports here

import torch
import numpy as np
from PIL import Image
import json
import argparse
from torch import nn
from torchvision import models

def json_file(file):
        with open(file, 'r') as f:
            catname = json.load(f)
        return catname

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath,arch,power):
    #gpu
    if torch.cuda.is_available() and power=='cuda':
        checkpoint= torch.load(filepath)
    else:
    #cpu
        checkpoint= torch.load(filepath, map_location=lambda storage, loc: storage)
        
    #load our model used
    arch=checkpoint['arch']
    units=checkpoint['hidden_layers']
    if arch=='vgg19':
        model=models.vgg19(pretrained=True)
        print('model downloaded')
    elif arch=='vgg13':
        model=models.vgg13(pretrained=True)
        print('model downloaded')
        
    for para in model.parameters():
            para.requires_grad=False
    #change the classifier
    classifier=nn.Sequential(nn.Linear(model.classifier[0].in_features,units),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(units,102),
                        nn.LogSoftmax(dim=1))
    
    model.classifier=classifier
    #load the parameters saved from our trained model
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer= checkpoint['optimizer_state']
    model.epochs=checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img = Image.open(image)
    width, height = img.size
    ratio=width/height

    if width>height:
        size=ratio*256,256
        img.thumbnail(size)
    else:
        size=256,ratio*256
        img.thumbnail(size)

#crop
    left = (img.width-224)/2
    right = left+224
    bottom = (img.height-224)/2
    top = bottom+224

    img1 = img.crop((left, bottom, right, top))

#normalize
    np_image = np.array(img1)/255
    means=np.array([0.485, 0.456, 0.406])
    std =np.array([0.229, 0.224, 0.225])
    net=(np_image-means)/std
    fin_img=net.transpose(2,0,1)
    return fin_img

def predict(image_path, model, topkn):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    #Implement the code to predict the class from an image file
    model.eval()
        
    ima=process_image(image_path)
    #change numpy array to tensor
    img = torch.from_numpy(ima).type(torch.FloatTensor)
    #add batch size to our image load with unsqueeze_(0)
    ps=torch.exp(model.forward(img.unsqueeze_(0)))
        
    #probability of prediction. Assign to a list for results manipulation
    pp, pp_class=ps.topk(topkn)
    flower_typ=pp_class.tolist()[0]
    flower_pred=pp.tolist()[0]
               
    return flower_pred,flower_typ

def show_pred(image_path,model,topknumber,cat_to_name):
    #Name of the flower from json file
    flower_id = image_path.split('/')[2]
    title= cat_to_name[flower_id]
    
    #running previous functions for prediction and showing flower image
    pred, idx= predict(image_path, model,topknumber)
    
    #flowers name of the n best predictions - first we flip the dict from class to idx to be ease to retrive the flower category
    #from cat_to_name
    invert_idx={}
    for i,k in model.class_to_idx.items():
        invert_idx[k]=i
    
    flo_name=[]
    for i in idx:
        f=cat_to_name[invert_idx[i]]
        flo_name.append(f)
 

    #showing results        
    print('\nWhich flower is it? ', title)
    print('\nModel prediction:\n')
    for i in range(topknumber):
        print('{} -- {:.3f}%' .format(flo_name[i],pred[i]))
    

def main():
    #Get command line instruction 

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", help="Store the model to use for the deep learning",
                    action="store", default='vgg19')
    parser.add_argument("image_path", help="Store the model to use for the deep learning",
                    action="store", default='vgg19')
    parser.add_argument('modelsaved', default="./checkpoint.pth", help="Path for the file where the trained model is stored")
    parser.add_argument('--top_k', action="store",type=int, default=5)
    parser.add_argument('--gpu', action="store", default='cuda')
    parser.add_argument('--category', action="store", default="cat_to_name.json")
    args = parser.parse_args()
    if args.image_path:
        image_path=args.image_path
    if args.modelsaved:
        save=args.modelsaved
    category=args.category
    power=args.gpu
    topk=args.top_k
    arch=args.arch

    #execution of the application 
    while True:
        print('Deep learning - prediction')
        start = input('\nReady to use the trained model? Enter yes or no.\n')
        #Loop to get the correct answer
        while True:
            if start.lower() not in ['yes','no']:
                print('Oups you did not write the correct answer. Please try again\n')
                start = input('\nReady to use the trained model? Enter yes or no.\n')
            else:
              break
        if start.lower() != 'yes':
            break
        #Load the functions
        cat_to_name = json_file(category)
        model_pred=load_checkpoint(save,arch,power)
        show_pred(image_path, model_pred,topk,cat_to_name)
        print('\nEnd of session!')
        break

if __name__ == "__main__":
	main()
# image_path = 'flowers/test/1/image_06743.jpg'