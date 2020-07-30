# Lung-Nodule-Detector in Pytorch

My first unguided deep-learning project to tackle my everyday problem in radiology.

The detector is supposed to create a mask upon the chest computer tomography images, marking the place where it thinks the nodules are.

The model trains on [LUNA 16 dataset](https://luna16.grand-challenge.org/), which CT images are stored as simple ITK files and coordinates of the nodules, their diameters were given, as well as, the coordinates for false positive candidates (false nodules they got from some model) to reduce false positive rate.

First, I import the data using tutorial provided on LUNA16 page and create nodule masks, assuming that they have spherical shape using this [tutorial](https://www.kaggle.com/arnavkj95/candidate-generation-and-luna16-preprocessing), to get the (Z,512,512) dimension images, since the Z dimensions in each CT scan are not of the same size (depends on the technicians, they might extend the image to include a varrying bits of upper abdomen), using [MIP](https://radiopaedia.org/articles/maximum-intensity-projection) to reduce the dimension Z dimension to 128. After normalizing can centering the image, I can chunk the entire CT in to 1024 chunks of (32,32,32) smaller images, which were then used to train the model.

In real-life scenario, converting the image from DICOM to simple ITK and then chunk it using 'fold' and 'unfold' method in Pytorch will preserve the chunk position within the image, allowing you to assemble it into one large CT image again, after inputting the chunks into the model. 

#### Note that I did not segment out the mediastinum, vessels and chest wall in the preprocessing part, because I'm still learning how to do so.

## Files included
- The model
- Others such as preprocessing and testing files will perhaps be added at a later date..

## Packages
- Pytorch
- Numpy
- matplotlib
- ipywidgets
- Scikit-image
