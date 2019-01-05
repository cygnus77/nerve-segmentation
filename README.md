# Semantic Segmentation in PyTorch: Ultrasound Nerve Segmentation

<b>Ultrasound Nerve Segmentation</b> is a Kaggle challenge to build a model that can identify nerve structures in a dataset of ultrasound images of the neck.
The dataset in this challenge is a great resource for learning and testing semantic segmentation algorithms. Here, I use PyTorch and Keras to explore semantic segmentation on this dataset.

## Sample Results

- **Purple**: True Positive (model prediction matches nerve area marked by human)
- **Yellow**: False Negative (nodel missed nerve in the area)
- **Green**: False Positive (model incorrectlt predicted nerve in the area)

![](out-good1.jpg)
![](out-good2.jpg)
![](out-miss1.jpg)

## The Data
Ultrasound images are provided as 8bit/pixel grascale, LZW compressed TIFF images of dimension 580x420. For each ultrasound image, a mask image of the same type and dimensions as the ultrasound is provided. Each mask image contains one or more manually marked areas representing location of the nerve segments (if any) in the corresponding ultrasound image.

| Ultrasound image| Label mask |
|---|---|
| ![ultrasoundimage](1_1.jpg) | ![mask](1_1_mask.jpg) |
| | |

OpenCV decodes each image into a 580x420x3 numpy uint8 array. Though the image is grayscale, I retained the 3 channels as pre-trained VGG model expects 3 channel input.

For smooth down and up sampling, I cropped the images to 576x416 (multiples of 32).

I developed a tool [image-analysis.py](./image-analysis.py) to:
- Identify duplicate images, with possibly differing masks (labeling errors).
- View duplicate images and make corrections.
- Distribution of frames with and without nerve segments to help balance dataset prior to training.

### Duplicates:
Output of [image-analysis.py](./image-analysis.py)


![](duplicate_images/dup1.png)
![](duplicate_images/dup2.png)
![](duplicate_images/dup3.png)
![](duplicate_images/dup4.png)


[image-analysis.py](./image-analysis.py) works in two modes: scanning and analysis. The '-scan' option scans all data files to spot duplicates by computing image differences for every possible pair of images and stores the results in a file. In the second step, it can show side-by-side comparisons of duplicates, histogram of difference values and allow user to enter corrections.

- 63 duplicates images with difference < 100, 52 of which have mismatching masks
- 131 duplicates images with difference < 1000, 107 of which have mismatching masks


### Histogram of image differences

(log scale)

![](./image-similarity.png)


## Neural Net Architecture
VGG-16 is a fairly simple deep network that is commonly used for image segmentation. Though VGG-16 is less accurate than the larger Resnet or Inception networks and slower than Mobilenets, its simple architecture lends itself to extension by adding additional layers, introducing skip-connections, etc.


Model

Loss Functions and Metrics

Training

Evaluation

