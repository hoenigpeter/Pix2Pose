xhost +
docker run \
    --gpus all \
    --net=host \
    --privileged \
    -e QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    -it \
    -v /home/hoenig/BOP/Pix2Pose:/Pix2Pose \
    --name=pix2posev0 pix2pose \
    bash