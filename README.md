# PyTorch-YOLO_Nano

codebase is ready for custom dataset  

A minimal PyTorch implementation of YOLO_Nano
- [Yolo_Nano](https://arxiv.org/abs/1910.01271)

#### Download COCO
    $ cd data/custom
#### Annotations
    $ cd annotations/
    place annotations here, for train, val, test here in COCO format
#### Images
    $ cd images
    place train/val/test images here in the respective folder named
    $ cd train_images/ --> training images
    $ cd val_images/ --> validation images
#### Names of labels
    $ vim custom.names
    write the list of all the class names here, the length of this files decides the number of classes in the final prediction layer in the network.
#### List of path to images
    Run file 
    $ python train_val_imgpaths.py
    This will generate the "train.txt" and "val.txt" files to be used in training
#### Model configuration
    $ cd config/
    $ vim custom.data  --> update this file with paths to the desired fields needed
    $ vim yolo-nano_cus.cfg --> update this file with the desired files.
    
## Module Pipeline

## training
```
update the batch size and number of workers here.

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
