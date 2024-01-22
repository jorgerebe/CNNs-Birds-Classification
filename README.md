# Convolutional Neural Networks (CNNs) for a bird species classification problem
In this repository you can find a full deep learning problem solved. Target is to classificate as accurately as possible bird species given an image, using convolutional neural networks.

To achieve it, pretrained models used to classify images from fastai (Pytorch) will be retrained with the images of the dataset of the problem we are trying to solve.

## Dataset Creation

As there are not very good existing datasets on the Internet of the bird species we are looking for, I created the dataset with web scraping techniques. Using Selenium, images has been scraped from Google Images, 100 for each class.

### Data Cleaning

Once we have got the images, as they are from the Internet (searched automatically in Google Images), there are going to be some that do not belong to any class, so they must be deleted.

### Train-Validation-Test Split

Once we have a good dataset, it must be splitted in train, validation and test sets. For this problem, 70% of the dataset has been used to train, 15% to validate and 15% to test.

## CNNs architectures used

Several models of convolutional neural networks have been studied and tested. Some basics such as AlexNet, advancing to GoogLeNet, DenseNet, VGGNet, ResNet and ResNeXt. Some are very expensive to train, so it is difficult to do several experiments to make sure that the model works right for our problem.

## CNN Architecture chosen

Finally, a **resnext50_32x4d** model has been chosen. Its training is pretty fast and it works reasonably well compared to the other architectures studied, as shown in its original paper. Inside resnext, the model chosen has a depth of 50 layers with a cardinality of 32 and bottleneck = 4d.

## Training the Model

The model was trained with the train set, and validated against the validation test during 10 epochs. During this training, only the weights of the last layer are modified.

### Callbacks
Pretrained models from fastai train pretty fast, so in order to prevent overfitting, callbacks such as EarlyStopping,ReduceLROnPlateau and SaveModelCallback are used.

![callbacks_1](https://github.com/jorgerebe/CNNs-Birds-Classification/assets/48808378/05f949d8-2f43-43fa-9e70-1f831f43354b)


## Fine Tuning

In order to get the best model possible it is not enough with modifying the weights of the last layer. During this second training we will modify the weights of the rest of the layer, using the _fine_tune_ function from fastai.

### Callback

Same callbacks as when training the model were used, but with different parameters.

![callbacks_2](https://github.com/jorgerebe/CNNs-Birds-Classification/assets/48808378/ce9b0655-68b3-4242-9076-990cc9f43a69)


## Testing the Model
Once the model has been trained, it must be tested against the test set. You can see the results in the Jupyter Notebook.
