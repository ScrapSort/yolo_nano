import json
import os
import shutil
import glob

train_file = "./annotations/annotations_train.json"
val_file = "./annotations/annotations_val.json"

def createCompletePath(file_path, img_dir, out_file):
	f = open(out_file,'w')

	data = json.load(open(file_path))
	for img_meta in data["images"]:
		f.writelines(os.path.join(os.getcwd(), img_dir, img_meta["file_name"])+'\n')
	
	f.close()

if __name__ == '__main__':
	createCompletePath(train_file, img_dir="images/train_images", out_file="train.txt")
	createCompletePath(val_file, img_dir="images/val_images", out_file="val.txt")
