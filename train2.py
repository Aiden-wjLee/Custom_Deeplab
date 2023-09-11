from SegmentationDataset import *
from Deeplab_resnet import DeepLabv3_plus

import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms,datasets
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler

import numpy as np
import gc
import cv2
from tqdm import tqdm
import datetime
import sys

from Discordsend import *
gc.collect()
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def data_loder(dataset_path, batch_size, crop_size, transform='default'):
    train_dataset = SegmentationDataset(f'{dataset_path}/train/',crop_size,'augmentation')#, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validation_dataset = SegmentationDataset(f'{dataset_path}/validation/',crop_size, 'augmentation')#, transform_nearest)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    
    return train_loader, validation_loader#, test_loader

def train(model, train_loader, criterion, optimizer, device, epoch,crop_size):
    model.train()
    train_loss = 0; train_correct = 0
    total_batches = len(train_loader)
    one_fifth_batches = total_batches // 5
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training", ncols=100)): #docker
        data, target = data.to(device), target.long().to(device) #target.float().to(device)
        optimizer.zero_grad()
        output = model(data)
        #Calculate loss and backprop
        loss = criterion(output, target.squeeze(1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        pred = output.argmax(dim=1, keepdim=True) #각 dim=1에 대해 최댓값을 구하고, 출력의 형태는 output의 형태와 같도록 설정.
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        if (batch_idx + 1) % one_fifth_batches == 0:
            print(f'Progress: {((batch_idx + 1) / total_batches) * 100:.1f}%')
        
        sys.stdout.flush()
    train_loss /= len(train_loader)
    train_accuracy = 100. * train_correct / (crop_size[0]*crop_size[1]*len(train_loader.dataset))
    
    return train_loss, train_accuracy#100. * train_correct / (256*256*len(train_loader.dataset)) 

def validation(model, val_loader,criterion, device, best_loss,crop_size):
    val_loss=0 ; val_correct=0
    model.eval()
    with torch.no_grad():
        for data, target in tqdm(val_loader, file=sys.stdout,desc="Validating", ncols=70): #docker
            data, target = data.to(device), target.long().to(device)
            
            #to device
            output = model(data)
            val_loss += criterion(output, target.squeeze(1))#.item()
            
            pred = output.argmax(dim=1, keepdim=True) #각 dim=1에 대해 최댓값을 구하고, 출력의 형태는 output의 형태와 같도록 설정.
            val_correct += pred.eq(target.view_as(pred)).sum().item()
        val_loss /= len(val_loader)
        val_accuracy = 100. * val_correct / (crop_size[0]*crop_size[1]*len(val_loader.dataset))
    return val_loss, val_accuracy#100. * val_correct / (len(val_loader.dataset)*256*256)

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss=0 ; test_correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target)

            pred = output.argmax(dim=1, keepdim=True) #각 dim=1에 대해 최댓값을 구하고, 출력의 형태는 output의 형태와 같도록 설정.
            test_correct += pred.eq(target.view_as(pred)).sum().item() #pred값과 target이 같은 값의 개수 

    return test_loss / len(test_loader), test_correct / len(test_loader.dataset)

def check_pth(model, folder_name):
    base_path = f"./pth_/BEST_{folder_name}_{0}"
    ext = ".pth"
    idx = 0
    # 파일이 존재하는지 검사
    while os.path.isfile(base_path + ext):
        idx += 1
        base_path = f"./pth_/BEST_{folder_name}_{idx}"
    return idx

def write_log_file(log_file, lr, num_epoch, batch_size, crop_size, data_path, folder_name, pth_i, class_num):
    log_file.write("========================================\n")
    log_file.write(f'pth_file: ./pth_/BEST_{folder_name}_{pth_i}.pth\n')
    log_file.write(f'Day: {datetime.datetime.now()}\n')
    log_file.write(f'class num:  {class_num}\n')
    log_file.write(f'lr: {lr}\n')
    log_file.write(f'num_epoch: {num_epoch}\n')
    log_file.write(f'batch_size: {batch_size}\n')
    log_file.write(f'crop_size: {crop_size}\n')
    log_file.write(f'data_path: {data_path}\n')
    log_file.write(f'folder_name: {folder_name}\n')
    log_file.write("========================================\n")
    log_file.flush()
    
def main():
    """
    Args: 
        crop_size : [height, width] 
        data_path : path of dataset
        class_num : number of class (include background)
    """
    #==========================================================================================
    #hyper parameter
    discord_send=False
    lr = 0.0005
    num_epoch = 1000
    batch_size = 4
    crop_size=[512,1280]#[512,1280]# [height, width]
    class_num = 3
    data_path = '/Dataset/nachi_test'
    folder_name = data_path.split('/')[-1]
    train_loader, validation_loader = data_loder(data_path,batch_size,crop_size,'augmentation')
    
    #device and model
    device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = DeepLabv3_plus(nInputChannels=3, n_classes=class_num, os=16, pretrained=True, _print=True).to(device)
    #==========================================================================================
    
    criterion = nn.CrossEntropyLoss()#nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    pth_i=check_pth(model, folder_name)
    best_loss=21545478
    with open(f'./pth_/log/log_BEST_{folder_name}_{pth_i}.txt', 'w') as log_file:
        write_log_file(log_file, lr, num_epoch, batch_size, crop_size, data_path, folder_name, pth_i, class_num)
        
        for epoch in range(0, num_epoch):
            train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device, epoch, crop_size)
            val_loss, val_accuracy = validation(model, validation_loader, criterion, device, best_loss, crop_size)
            
            if val_loss <= best_loss:
                best_loss = val_loss
                best_epoch = epoch + 1
                print("BEST LOSS:", best_loss)
                
                torch.save(model.state_dict(), f"./pth_/BEST_{folder_name}_{pth_i}.pth")
                print(f"./pth_/BEST_{folder_name}_{pth_i}.pth SAVED")
                log_file.write(f"BEST LOSS: {best_loss}  ./pth_/BEST_{folder_name}_{pth_i}.pth SAVED \n")
                
            log_message = f'Epoch: {epoch + 1}/{num_epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% \
                Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}% BEST EPOCH: {best_epoch}\n'
            if epoch==0:
                if discord_send:
                    asyncio.run(send_message(f"learning start, 'pth_file: ./pth_/BEST_{folder_name}_{pth_i}.pth"))
                torch.save(model.state_dict(), f"./pth_/BEST_{folder_name}_{pth_i}_epoch0.pth")
            #  print(log_message)
            print(log_message)
            log_file.write(log_message)
            log_file.flush()
    if discord_send:
        asyncio.run(send_message(f'Epoch: {epoch+1}/{num_epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%  \
                Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, \n best_epoch: {best_epoch}'))
if __name__=='__main__':
    main()