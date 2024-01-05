sudo docker run -it --network host --gpus all --ipc=host --privileged \
    -e DISPLAY=$DISPLAY -v $(pwd):/home -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /media/lee/90182A121829F83C2/Dataset:/Dataset \
    -v /dev/usb:/dev/usb \
    --name deeplab_used deeplab_used