# Introduction
 Rapid and accurate evaluation of seed vigor is of great significance for seed industry development and market stability. As an emerging non-destructive testing method with high sensitivity and accuracy, multispectral imaging technology has a large potential for application in seed quality evaluation. Different maturity degrees of seeds will further affect seed vigor by affecting seed germination rate and germination potential. To overcome this problem, we develop a convenient bioinformatics software tool called nCDA-CNN by combining spectral information with convolutional neural network, which significantly improved the recognition accuracy. nCDA-CNN pipeline is a powerful and effective tool for seed prediction and can be widely applied on seed examination and identification. 
# Dependencies required
 nCDA-CNN could be run in Windows, Mac and Linux, after the following dependencies are installed: 1. Python; 2. TensorFlow. 
# Running procedures and Input 
1. Preparing single-seed images on the pairwise conditions. The example data was generated with nCDA in VideometerLab4, and stored in two image folders, i.e., milk and dough. The image format inside the folder should be in jpg format. The other two corresponding empty folders should be also created to store the transformed images with uniform image resolution, for example, milk_128 and dough_128.
  
2. Seed images would be transformed with 128*128 resolution, and stored in the two empty folders, i.e., milk_128 and dough_128. The higher resolution might be set up, although it will require more running time.
  
3. Setting the parameters in the python script “CNN-nCDA”. The parameters are shown as followed.
(1) working directory (default: D:/CNN/) following the four folders prepared above on the lines 12-16 should be set up firstly. 
(2) two sample labels on line 19 (default: 'milk', 'dough') could be assigned. 
(3) resolution of 128*128 pixels on lines 27, 84, 90, 100 could be changed for better resolution. 
(4) training and test images could be set up on the line 25, e.g., “test_size=0.2” for 20% test and 80% training data for the default. 
(5) “a, b = 5, 6” on the line 40, to show the results for the preferred 30 training images. The quantity could be adjusted, e.g., “a, b = 3, 4” for 12 training images. 

4. Running CNN-nCDA.py in command line: python CNN-nCDA.py
# Output
There are three output results, including accuracy, training images, test images with prediction results. 
1. The accuracy with epochs is shown on the command line.
![image](https://github.com/CAUhx/nCDA-CNN/blob/main/readme%20images/1.png)
2. The first 30 images used for training are shown.
![image](https://github.com/CAUhx/nCDA-CNN/blob/main/readme%20images/2.png)
3. All the test images are shown finally, with the prediction for each image. The example data outputs all the 40 images as followed. 
![image](https://github.com/CAUhx/nCDA-CNN/blob/main/readme%20images/3.png)
