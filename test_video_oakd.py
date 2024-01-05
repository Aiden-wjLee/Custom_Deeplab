import torch
import torchvision.transforms as transforms
from PIL import Image
from Deeplab_resnet import DeepLabv3_plus
#from unet_model import UNet
import cv2
import numpy as np
import os
import pyrealsense2 as rs
import depthai as dai
import time
import argparse

torch.cuda.empty_cache()
def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use sdk python api')
    parser.add_argument('--device_name', default='cuda' ,help='name of device, cuda or cpu')
    parser.add_argument(
        '--model_path', default='/root/workspace/mmdeploy/mmdeploy_models/mmseg/ort2',
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('--image_path', default='/root/workspace/mmdeploy/demo/resources/cityscapes.png',
                        help='path of an image')
    args = parser.parse_args()
    return args


def load_model(model_path):
    model = DeepLabv3_plus(nInputChannels=3, n_classes=3, os=16, pretrained=True, _print=True)
    device = torch.device("cuda:0")  # Adjust if multiple GPUs are available
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((512, 1280)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def main(args):
    #load model
    model_path = "./pth/BEST_nachi_grid_integrated_12_4_0.pth"
    #BEST_green_real_floor2.pth"
    model = load_model(model_path)
    
    with dai.Device(pipeline) as device:
        device.startPipeline()
        qRgb = device.getOutputQueue(name="RGB", maxSize=4, blocking=False)

        frame_count = 0
        start_time = time.time() 
        # Process each frame
        while True:
            inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
            frame_count += 1

            # Calculate FPS
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time

            # Display FPS on frame
            frame = inRgb.getCvFrame()
            frame = cv2.resize(frame,(1024,512))

            # Preprocess image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = preprocess_image(image)
            
            #predict
            with torch.no_grad():
                output = model(image_tensor)
            #predicted_labels = output.round().float()*255
            predicted_labels = output.argmax(dim=1, keepdim=True)
            predicted_labels_image = predicted_labels.cpu().numpy().squeeze()
            print(np.max(predicted_labels_image), np.min(predicted_labels_image))
            print(predicted_labels_image.shape)
            cv2.imshow("original", frame)
            frame_show = cv2.resize(frame, (predicted_labels_image.shape[0], predicted_labels_image.shape[1]))
                
            predicted_labels_image_3d = np.stack([predicted_labels_image[:,:]]*3, axis=-1)
            predicted_labels_image=predicted_labels_image*255
            predicted_labels_image=predicted_labels_image.astype(np.uint8)
            print("max of predicted: ", np.max(predicted_labels_image))
            cv2.imshow("predicted", predicted_labels_image)
            
            print("max_frame_show: ", np.max(frame_show),"max_predicted_labels_image_3d: ", np.max(predicted_labels_image_3d))
            #integrated=cv2.addWeighted(frame_show/255, 1, predicted_labels_image_3d, 0.5, 0, dtype=cv2.CV_32F)
            #cv2.imshow("integrated", integrated)
            
            print("=========================")
            # Check for key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    
if __name__ == '__main__':
    pipeline = dai.Pipeline()

    camRgb = pipeline.createColorCamera()
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setFps(60)

    rgbout = pipeline.createXLinkOut()
    rgbout.setStreamName("RGB")
    camRgb.video.link(rgbout.input)
    args = parse_args()
    main(args)