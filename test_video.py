import torch
import torchvision.transforms as transforms
from PIL import Image
from Deeplab_resnet import DeepLabv3_plus
#from unet_model import UNet
import cv2
import numpy as np
import os
import pyrealsense2 as rs

def load_model(model_path):
    model = DeepLabv3_plus(nInputChannels=3, n_classes=2, os=16, pretrained=True, _print=True)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def main():
    #load model
    model_path = "./pth_/BEST_water_0806_0.pth"
    #BEST_green_real_floor2.pth"
    model = load_model(model_path)
    
     # Open webcam
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    
    pipeline.start(config)
    
    # Process each frame
    while True:
        # Read next frame from webcam
        # ret, frame = video_capture.read()
        # if not ret:
        #     break
        
        frame = pipeline.wait_for_frames()
        frame = frame.get_color_frame()
        frame = np.asanyarray(frame.get_data())
        
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
        #show original image
        #image = cv2.resize(image, (predicted_labels_image.shape[2], predicted_labels_image.shape[1]))
        cv2.imshow("original", frame)
        frame_show = cv2.resize(frame, (predicted_labels_image.shape[0], predicted_labels_image.shape[1]))
              
        #show predicted image
        #predicted_labels_image_3d = np.stack([predicted_labels_image[0,:,:]]*3, axis=-1)
        predicted_labels_image_3d = np.stack([predicted_labels_image[:,:]]*3, axis=-1)
        #image_integrated = np.clip(frame_show.astype(float)*255 + predicted_labels_image_3d.astype(float) * 255, 0, 255).astype(np.uint8)
        #cv2.imshow("integrated", image_integrated)
        predicted_labels_image=predicted_labels_image*255
        predicted_labels_image=predicted_labels_image.astype(np.uint8)
        print("max of predicted: ", np.max(predicted_labels_image))
        cv2.imshow("predicted", predicted_labels_image)
        
        print("max_frame_show: ", np.max(frame_show),"max_predicted_labels_image_3d: ", np.max(predicted_labels_image_3d))
        integrated=cv2.addWeighted(frame_show/255, 1, predicted_labels_image_3d, 0.5, 0, dtype=cv2.CV_32F)
        cv2.imshow("integrated", integrated)
        
        print("=========================")
        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
if __name__ == '__main__':
    main()