# MFTP
MFTP models in mmdetection

In this git, MFPT-FPNOF3 applying ResNet-50 as the backbone is given.

Firstly, you should install the mmdetection as https://github.com/open-mmlab/mmdetection.

Then running the order in 'move.txt' to apply our codes.

Also, you should add the module in the corresponding __init__.py.

"python setup develop".

Happly running our code!

Training 

bash tools/dist_train.sh config/xxxx N

Testing

python tools/test.py config/xxxx workdirs/xxxx --eval bbox

