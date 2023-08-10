xhost +
docker run \
    --gpus all \
    --net=host \
    --privileged \
    -e QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    -it \
    -v /home/hoenig/BOP/Pix2Pose:/Pix2Pose \
    --name=pix2pose_tf2v1 pix2pose_tf2 \
    bash
    #--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \