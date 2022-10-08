# Hack to Run this Project 

## Step 1: 

```
conda create -n oln python=3.7 -y
conda activate oln
conda install pytorch=1.7.0 torchvision cudatoolkit=11.0 -c pytorch -y
pip install -r requirements.txt
pip install mmcv-full==1.2.7
pip install -v -e . 
```

## Step 2: 
Pre-trained models are available for download [here](https://drive.google.com/uc?id=1uL6TRhpSILvWeR6DZ0x9K9VywrQXQvq9). Place it under `trained_weights/latest.pth` and run the following commands to test OLN on COCO dataset.

## Step 3: 

### Multi-GPU distributed testing
```
python tools/test.py configs/oln_box/oln_box.py trained_weights/latest.pth --eval bbox
```
## Step 4: 

### Visualize Predictions made my Model 
```
python ../DetVisGUI/DetVisGUI_test.py configs/oln_box/oln_box.py trained_weights/latest.pth data/coco/val2017
```

## Step 5:

### Visualize on a given image
```
python extras/image_testing.py data/custom_images/image1.jpg configs/oln_box/oln_box.py trained_weights/latest.pth
```

Note: For custom image testing, run - 

```
python ../DetVisGUI/DetVisGUI_test.py configs/oln_box/oln_box.py trained_weights/latest.pth data/custom_images
```

Single image: 
```
python extras/image_testing.py data/custom_images/image1.jpg configs/oln_box/oln_box.py trained_weights/latest.pth --outfile output/oimage1.jpg
```

Web Scraper
```
python extras/web_scraper.py mango 20
```

For more on configs, read Tutorial 1.

