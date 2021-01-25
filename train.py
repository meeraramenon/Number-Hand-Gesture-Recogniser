# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Step 1 - Building the CNN

# Initializing the CNN
classifier = Sequential()
#we are using the sequential class to initialise the classifier object, this is the neural network and we will add our layers into this

# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu')) #32 is the number of filters, (3,3) -> size of convolution window
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten()) #after conv and pooling we will be left with layers in the form of 2d arrays, so here we convert it into a 1d array

# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu')) #128 number of neurons
classifier.add(Dense(units=6, activation='softmax')) # softmax for more than 2
#output layers 6 is the no. of classes we have 0 1 2 3 4 5

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2


# Step 2 - Preparing the train/test data and training the model

# Code copied from - https://keras.io/preprocessing/image/
#basically takes in the image, adds shear, zoom so that there is some variety
#then it generates the images
#then we have mentioned the training and testing directory
#it takes the images and feeds it to neural network
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(64, 64),
                                                 batch_size=5,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(64, 64),
                                            batch_size=5,
                                            color_mode='grayscale',
                                            class_mode='categorical') 

#we use the fit generator method to train the ds
classifier.fit_generator(
        training_set,
        steps_per_epoch=600, # No of images in training set
        epochs=10, #set this to larger value if we are training on rgb images
        validation_data=test_set,
        validation_steps=30)# No of images in test set


# Saving the model and weightts
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model-bw.h5')
#these are loaded to predict.py to make the predictions

