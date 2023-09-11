import torch
import torchvision.transforms as transforms
from PIL import Image
from Deeplab_resnet import DeepLabv3_plus
import cv2
import numpy as np
import os
import argparse
#from poly import poly

def load_model(model_path):
    model = DeepLabv3_plus(nInputChannels=3, n_classes=3, os=16, pretrained=True, _print=True)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((512, 1280)), #512, 1024 세로 가로 
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def close_holes(mask, kernel_size=5):
    """Close small holes in a binary image using morphological operation"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed_mask
def parse_args():
    parse_args = argparse.ArgumentParser()
    parse_args.add_argument('--image_num', type=str, default='4')
    return parse_args.parse_args()
def main():
    #load model
    model_path = "./pth_/BEST_nachi_LR_3.pth"
    print("model_path: ", model_path)
    args = parse_args()
    #image_path ="/Dataset/nachi_2560_1024_LR/validation/img/15.png"
    image_path = f"/home/result/{args.image_num}.png"
    model = load_model(model_path)
    
    
    #predict``
    with torch.no_grad():
        image_tensor = preprocess_image(image_path)
        output = model(image_tensor)
    print("predicted_max: ", torch.max(output))
    predicted_labels = output.argmax(dim=1, keepdim=True)
    #predicted_labels = torch.sigmoid(output).round().float()
    predicted_labels_image = predicted_labels.cpu().numpy().squeeze()
    print("max of predicted_labels_image:", np.max(predicted_labels_image))
    
    print("predicted_labels_image_value: ", np.unique(predicted_labels_image))
    
    predicted_labels_image = predicted_labels_image.astype('float32')
    if np.max(predicted_labels_image) > 1:
        predicted_labels_image = (predicted_labels_image - predicted_labels_image.min()) / (predicted_labels_image.max() - predicted_labels_image.min())
        
    #kernel_size = 10  # 이 값을 조정하여 구멍의 크기에 따라 적절한 kernel size를 선택하세요.
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    #predicted_labels_image = cv2.morphologyEx(predicted_labels_image, cv2.MORPH_CLOSE, kernel)

    stack = np.dstack([predicted_labels_image]*3)
    #cv2.imshow("mask", predicted_labels_image)
    cv2.imwrite("./result/mask.png", predicted_labels_image*255)
    
    #show original image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (predicted_labels_image.shape[1], predicted_labels_image.shape[0]))#.astype('float32') #(width, height)
    image = image.astype('float32')/255
    #cv2.imshow("original", image)
    cv2.imwrite("./result/original.png", image*255)
    print("stack_value: ", np.unique(stack))
    print(image.shape, stack.shape)
    
    #show integrated image
    image_integrated= cv2.addWeighted(image, 1, stack, 0.5, 0, dtype=cv2.CV_32F)
    #cv2.imshow("integrated", image_integrated)
    cv2.imwrite("./result/integrated.png", image_integrated*255)
    #cv2.waitKey(0)
    
if __name__ == '__main__':
    main()