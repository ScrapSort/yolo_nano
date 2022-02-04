# PyTorch-YOLO_Nano

Still working on customizing to our ```trash_dataset``` 

```Updated the network, now network size is 12 mb from 22 mb with similar performance```

A minimal PyTorch implementation of YOLO_Nano
- [Yolo_Nano](https://arxiv.org/abs/1910.01271)

#### Download COCO
    $ cd data/
    $ bash get_coco_dataset.sh
## Module Pipeline

## training
```
./train_cus.sh
```

Better Para:  
   --epochs 120  
   --batch_size 8  
   --model_def ./config/yolo-nano.cfg  
   --lr 2.5e-4  
   --fix_up True  
   --lr_policy cosine
```
## Testing
```bash
python test.py --data_config ./config/coco.data --model_def ./config/yolo-nano.cfg --weights_path [checkpoint path]
```
## Result
we compare with yolov-3，yolo-tiny. We got competitive results.  

Methods |mAP@50|mAP|weights|FPS| Model 
yolo-nano          | 55.6 |27.7 | 22.0M 
 
 for trained model : check ``` checkpoints/``` folder
