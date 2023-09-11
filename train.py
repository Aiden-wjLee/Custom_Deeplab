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

from Discordsend import *
gc.collect()
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def data_loder(dataset_path, batch_size, crop_size):
    ''''
    transform = transforms.Compose([
        #transforms.Resize((1024, 512)),
        transforms.Resize((crop_size[0], crop_size[1])),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_nearest = transforms.Compose([
        transforms.Resize((crop_size[0], crop_size[1]),interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])
    '''
    train_dataset = SegmentationDataset(f'{dataset_path}/train/',crop_size)#, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) #drop_last : 마지막 배치가 batch_size보다 작을 때 drop할지 여부

    validation_dataset = SegmentationDataset(f'{dataset_path}/validation/',crop_size)#, transform_nearest)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, drop_last=True)
    print("train_loader_len: ",train_loader.dataset.__len__(), "validation_loader_len: ",validation_loader.dataset.__len__())
    #print("loader_shape: ",train_loader.dataset.__getitem__(0)[0].shape, validation_loader.dataset.__getitem__(0)[0].shape)
    '''
    train_dataset = SegmentationDataset(f'{dataset_path}\\train\\', transform)

    # split train dataset into train and validation dataset
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(dataset_size * 0.8) # split index
    np.random.shuffle(indices) # shuffle indices
    train_indices, validation_indices = indices[:split], indices[split:]

    # define samplers for train loader and validation loader
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)

    # create train loader and validation loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=validation_sampler)
    '''
    #test_dataset = SegmentationDataset(f'{dataset_path}/test', transform)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, validation_loader#, test_loader

def train(model, train_loader, criterion, optimizer, device, epoch,crop_size):
    model.train()
    train_loss = 0; train_correct = 0
    total_batches = len(train_loader)
    one_fifth_batches = total_batches // 5
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.long().to(device) #target.float().to(device)
        #print("data_shape: ",data.shape, target.shape)
        optimizer.zero_grad()
        output = model(data)
        #Calculate loss and backprop
        loss = criterion(output, target.squeeze(1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True) #각 dim=1에 대해 최댓값을 구하고, 출력의 형태는 output의 형태와 같도록 설정.
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        
        #show progress
        if (batch_idx + 1) % one_fifth_batches == 0:
            print(f'Progress: {((batch_idx + 1) / total_batches) * 100:.1f}%')
            
    train_loss /= len(train_loader)
    train_accuracy = 100. * train_correct / (crop_size[0]*crop_size[1]*len(train_loader.dataset))
    if epoch%20==0:
        if not os.path.exists("./pth_/BEST_ju"):
            os.mkdir("./pth_/BEST_ju")
        torch.save(model.state_dict(), f"./pth_/BEST_ju/{epoch}.pth")
    return train_loss, train_accuracy#100. * train_correct / (256*256*len(train_loader.dataset)) 

def validation(model, val_loader,criterion, device, best_loss,crop_size):
    print("validation")
    val_loss=0 ; val_correct=0
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.long().to(device)
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
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target)

            pred = output.argmax(dim=1, keepdim=True) #각 dim=1에 대해 최댓값을 구하고, 출력의 형태는 output의 형태와 같도록 설정.
            test_correct += pred.eq(target.view_as(pred)).sum().item() #pred값과 target이 같은 값의 개수 

    return test_loss / len(test_loader), test_correct / len(test_loader.dataset)

def check_pth(folder_name):
    base_path = f"./pth_/BEST_{folder_name}"
    ext = ".pth"
    idx = 0
    # 파일이 존재하는지 검사
    while os.path.isfile(base_path + ext):
        idx += 1
        base_path = f"./pth_/BEST_{folder_name}_{idx}"
    return idx
def main():
    """
    method: 
        crop_size: [height, width] 를 조정.
        data_path : 데이터 셋의 경로 조정.
        model = UNet(in_channels=3,out_channels=2).to(device) 에서 out_channels를 클래스의 개수에 맞게 조정.
    """
    #학습 관련 변수
    lr = 0.0005
    num_epoch = 2000
    batch_size = 2
    crop_size=[512,512]#[512,1280]# [height, width]
    
    ###train_loader, validation_loader = data_loder('D:\\Dataset\\Water_green',batch_size) #edit 0722#
    #train_loader, validation_loader = data_loder('D:\\Dataset\\nachi_2560_1024',batch_size,crop_size)
    #data_path = 'D:\\Dataset\\Water_real_floor'
    data_path = '/Dataset/Water_real_floor'
    folder_name = data_path.split('/')[-1]
    print("folder_name: ",folder_name)
    print("crop size: ",crop_size)
    train_loader, validation_loader = data_loder(data_path,batch_size,crop_size)

    #디바이스 및 네트워크 
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepLabv3_plus(nInputChannels=3, n_classes=2, os=16, pretrained=True, _print=True).to(device)
    #UNet(in_channels=3,out_channels=2).to(device) #nachi 3, water 2
    
    criterion = nn.CrossEntropyLoss()#nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    pth_i=check_pth(folder_name)
    best_loss=21545478
    for epoch in range(0, num_epoch):
        train_loss, train_accuracy= train(model, train_loader, criterion, optimizer, device, epoch,crop_size)
        val_loss,val_accuracy =validation(model, validation_loader,criterion, device,best_loss,crop_size)
        if val_loss<=best_loss:
            best_loss = val_loss
            best_epoch = epoch+1
            print("BEST LOSS: ", best_loss)
            if os.path.isdir(f"./pth_")==False:
                os.mkdir(f"./pth_")
            torch.save(model.state_dict(), f"./pth_/BEST_{folder_name}_{crop_size[0]}_{crop_size[1]}_{pth_i}.pth")
            print(f"./pth_/BEST_{folder_name}_{crop_size[0]}_{crop_size[1]}_{pth_i}.pth SAVED")
        print(f"pth file name: ./pth_/BEST_{crop_size[0]}_{crop_size[1]}_{pth_i}.pth")
        #test_loss, test_accuracy = test(model, test_loader, criterion, device)
        print(f'Epoch: {epoch+1}/{num_epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%  \
            Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%  BEST EPOCH: {best_epoch}')
        #Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    asyncio.run(send_message(f'Epoch: {epoch+1}/{num_epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%  \
            Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, \n best_epoch: {best_epoch}'))
if __name__=='__main__':
    main()