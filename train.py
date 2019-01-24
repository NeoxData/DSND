#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: christophe
"""

#imports
import argparse
import time
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

#functions to be used for the model preparation, validation and saving 
def prep(arch,epochs,lea_rate,save_place,hdunits,data_dir,power):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

#transformation of images from dataset
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    valid_transforms=transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_transforms=transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

#Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)

    image_val_datasets= datasets.ImageFolder(valid_dir, transform=valid_transforms)

    image_test_datasets= datasets.ImageFolder(test_dir, transform=test_transforms)

#Using the image datasets and the trainforms, define the dataloaders
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    validloaders = torch.utils.data.DataLoader(image_val_datasets, batch_size=64)
    testloaders = torch.utils.data.DataLoader(image_test_datasets, batch_size=64)
#model choice
    if arch=='vgg19':
        model=models.vgg19(pretrained=True)
        print('model downloaded')
    elif arch=='vgg13':
        model=models.vgg13(pretrained=True)
        print('model downloaded')

    for para in model.parameters():
        para.requires_grad=False
 #modification of the model classifier   
    classifier=nn.Sequential(nn.Linear(model.classifier[0].in_features,hdunits),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hdunits,102),
                        nn.LogSoftmax(dim=1))

    model.classifier=classifier
    criterion=nn.NLLLoss()
    optimizer=optim.SGD(model.classifier.parameters(),lr=lea_rate, momentum=0.9)
    
    print('model classifer modified')
    print('Starting to learn! Please wait..')
    device=torch.device('cuda' if torch.cuda.is_available() and power=='cuda' else 'cpu')
#training the model
    model.to(device)
    running_loss=0
    for e in range(epochs):
        for images, labels in dataloaders:
            images=images.to(device)
            labels=labels.to(device)
        
            optimizer.zero_grad()
        
            start=time.time()
        
            output=model.forward(images)
            loss=criterion(output,labels)            
            loss.backward()
            optimizer.step()
        
            running_loss+=loss.item()
    
        else:
            print("epoch:{}/{}". format(e+1, epochs),
              f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds",
              "Training loss: {:.3f}".format(running_loss/len(dataloaders)))
            accuracy=0
            model.eval()
            with torch.no_grad():
                for input_val, label_val in validloaders:
                    input_val=input_val.to(device)
                    label_val=label_val.to(device)
                
                    ps=torch.exp(model.forward(input_val))
                
                    pp, pp_class=ps.topk(1, dim=1)
                 
                    equal=pp_class==label_val.view(*pp_class.shape)
                    accuracy+=torch.mean(equal.type(torch.FloatTensor))
                
                print("Validation Accuracy: {:.3f}".format(accuracy/len(validloaders)))
                running_loss=0
                model.train()
    return model, testloaders,image_datasets,optimizer            
    print('Model Trained!')
#Test the model with test dataset
def validation(testloaders,power,model):
    device=torch.device('cuda' if torch.cuda.is_available() and power=='cuda' else 'cpu')
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for input_test, label_test in testloaders:
            input_test=input_test.to(device)
            label_test=label_test.to(device)
                
            ps=torch.exp(model.forward(input_test))
                
            pp, pp_class=ps.topk(1, dim=1)
                
            equal=pp_class==label_test.view(*pp_class.shape)
            accuracy+=torch.mean(equal.type(torch.FloatTensor))
                
    print("Test Accuracy: {:.3f}".format(accuracy/len(testloaders)))

# Save the checkpoint 
def save(model,image_datasets, save_place, epochs,arch,optimizer, hdunits):
    
    model.class_to_idx = image_datasets.class_to_idx

    checkpoint={'input_size':model.classifier[0].in_features,
                'hidden_layers': hdunits,
           'output_size':102,
           'state_dict':model.state_dict(),
            'class_to_idx': model.class_to_idx,
           'epochs': epochs,
            'arch': arch,
           'optimizer_state':optimizer.state_dict()}

    torch.save(checkpoint, save_place)

def main():
    #Get command line instruction 

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", help="Store the model to use for the deep learning",
                    action="store", default='vgg19')
    parser.add_argument('directory', type=str, default="./flowers", help="Path for the directory file where the dataset     is stored")
    parser.add_argument('--learning_rate', action="store",type=int, default=0.001)
    parser.add_argument('--hidden_units', action="store", type=int, default=2000)
    parser.add_argument('--epochs',action="store", type=int, default=5)
    parser.add_argument('--gpu', action="store", default='cuda')
    parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
    args = parser.parse_args()
    if args.directory:
        data_dir=args.directory
    power=args.gpu
    save_place=args.save_dir
    hdunits=args.hidden_units
    lea_rate= args.learning_rate
    epochs=args.epochs
    arch=args.arch
    #execution of the application 
    while True:
        print('Deep learning application for prediction')
        start = input('\nReady? Enter yes or no.\n')
        #Loop to get the correct answer
        while True:
            if start.lower() not in ['yes','no']:
                print('Oups you did not write the correct answer. Please try again\n')
                start = input('\nReady to use the trained model? Enter yes or no.\n')
            else:
              break
        
        
        if start.lower() != 'yes':
            break
        #load the functions
        model,testloaders,image_datasets,optimizer=prep(arch,epochs,lea_rate,save_place,hdunits,data_dir,power)
        validation(testloaders,power,model)
        
        accu = input('\nAccording to the validation accuracy, do you want to save the model? Enter yes or no.\n')
#Loop to get the correct answer
        while True:
            if accu.lower() not in ['yes','no']:
                print('Oups you did not write the correct answer. Please try again\n')
                accu = input('\nAccording to the validation accuracy, do you want to save the model? Enter yes or no.\n')
            else:
              break    
    
    
        if accu.lower() == 'yes':
            save(model,image_datasets, save_place, epochs,arch,optimizer,hdunits)
            print('Model training saved. End of session!')
            break
        else:
            print('End of session!')
            break

if __name__ == "__main__":
	main()