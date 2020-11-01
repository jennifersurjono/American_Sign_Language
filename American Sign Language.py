#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import all necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import accuracy_score


# In[2]:


#We are going to import two datasets : train and validation sets
#The train set will be split into train and test sets. Also, this set is used for our modelling initiation
#The validation set is the set we are going to test how well the applied model will work.

#Import the train and validation data
sl = pd.read_csv("D:/datacamp/Python/Introduction to Tensorflow in Python/sign_mnist_train.csv")
sl_validate = pd.read_csv("D:/datacamp/Python/Introduction to Tensorflow in Python/sign_mnist_test.csv")


# In[3]:


#Take a look at the first 5 rows of the train set
sl.head()


# In[4]:


#Check out the shape of the train set
sl.shape


# In[5]:


#The train set had 27,455 rows and 785 columns
#The first column represents the image's label and the rest of the columns represent the pixels of the images

#Take a look at the first 5  columns of validation set
sl_validate.head()


# In[6]:


#Check out the shape of the validation set
sl_validate.shape


# In[7]:


#The validation set has 7172 images

#Now, we want to know which label represents which American sign language
#Import the picture of 24 American sign languages 
Image("D:/datacamp/Python/Introduction to Tensorflow in Python/amer_sign2.png")


# In[8]:


#Now, let's define our train target which, is the label column, as an array. Then, print the unique values of the array 
target = np.array(sl["label"])
np.unique(target)


# In[9]:


#Also, define our validation target as an array and print the unique values
target_validate = np.array(sl_validate["label"])
np.unique(target_validate)


# In[10]:


#As we can see, the target values represent the sign languages types as the following :
#0-8 : A-I
#10-24 : K-Y
#Sign J and Z is not shown in this dataset

#We are entering a brief EDA. We want to find out the number of each sign language in both train and validate targets
sns.countplot(target)


# In[11]:


sns.countplot(target_validate)


# In[12]:


#Now, we are in the pre-processing step
#For modelling purpose, we need to transform our target into binary type
#Define the label binarizer
lb = LabelBinarizer()


# In[13]:


#Transform the train and validation targets into binary type
target = lb.fit_transform(target)
target


# In[14]:


target_validate = lb.fit_transform(target_validate)
target_validate


# In[15]:


#Define our train features, which are the pixels of the image, in an array as float32 type
features = np.array(sl.drop('label',axis=1), np.float32)
features


# In[16]:


#Define our validation features, which are the pixels of the image, in an array as float32 type
features_validate = np.array(sl_validate.drop('label',axis=1), np.float32)
features_validate


# In[17]:


#The features' value are still in range of 0-255, so we need to normalize them into a range of 0-1  
features = features/255
features_validate = features_validate/255


# In[18]:


#Now, let's see some of the dataset's images in black and white style
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.grid(False)
    plt.imshow(features[i].reshape(28,28), cmap = plt.cm.binary)
plt.show()


# In[19]:


#Split the train set into 70% train and 30% test sets
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=42)


# In[20]:


#Check out the train features' shape. It should have 784 columns
features_train.shape


# In[21]:


#Also, check out the train targets' shape. It should have 24 columns (represents the labels in binary type)
target_train.shape


# In[22]:


#Now, we are in modelling step
#We are going to apply neural networks model for our prediction
#In the model, we are assigning 2 hidden layers and 1 output layers
#We are applying dropouts for the hidden layers to avoid overfitting
#We are also applying adam as optimizer, categorical crossentropy as loss and accuracy as metrices
model=keras.Sequential()
model.add(keras.layers.Dense(256, activation='sigmoid', input_shape=(784,)))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.15))
model.add(keras.layers.Dense(24, activation='softmax'))
model.compile(optimizer = 'adam', loss='categorical_crossentropy',metrics=["accuracy"])
model.summary()


# In[23]:


#Now, let's run the model and check out the accuracy of the train set, test set and validation set
model.fit(features_train, target_train, epochs=30)
test_loss, test_acc = model.evaluate(features_test, target_test, verbose=2)
val_loss, val_acc = model.evaluate(features_validate, target_validate, verbose=2)
print('\nTest accuracy:', test_acc)
print('\nValidation accuracy', val_acc)


# In[24]:


#The model work well on the train and test set as it does not overfit the test set.
#For the validation set, it had 83% accuracy which is good enough in making prediction

#Let's see our predictions of the image.
#The output will be in the form of probability of each label. The highest probability means the the label of each image
y_pred = model.predict([features_validate])
y_pred


# In[25]:


#To compare the predicted images against the actual ones, we must transform the predicted and actual targets into original type
#Then, transform them into dataframe
y_pred = lb.inverse_transform(y_pred)
y_pred = pd.DataFrame(y_pred, columns=["Predicted"])
y_test = lb.inverse_transform(target_validate)
y_test = pd.DataFrame(y_test, columns=["Actual"])


# In[26]:


#Combine the actual and predicted image labels
results = pd.concat([y_test,y_pred], axis=1)


# In[28]:


#Save the result to csv file
results.to_csv(r"D:/datacamp/Python/Introduction to Tensorflow in Python/American Sign Language",header=True)


# In[ ]:




