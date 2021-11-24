# STOD
STOD models in mmdetection

In this git, STOD-DDE3X applying swin-transformer as the backbone is given.

Firstly, you should install the mmdetection as https://github.com/open-mmlab/mmdetection.

Then running the order in 'move.txt' to apply our codes.

"python setup develop".

Happly running our code!

Training 

bash tools/dist_train.sh config/xxxx N

Testing

python tools/test.py config/xxxx workdirs/xxxx --eval bbox

The 50.0 mAP and 70.7 AP50 model can be download at https://drive.google.com/file/d/10Iv4h1eIbxksI7Zq9d48EkUharT13ojj/view?usp=sharing
