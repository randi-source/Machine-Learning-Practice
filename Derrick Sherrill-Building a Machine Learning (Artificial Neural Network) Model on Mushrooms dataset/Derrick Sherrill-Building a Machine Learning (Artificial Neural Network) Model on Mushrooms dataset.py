#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


# In[2]:


mushroom_df= pd.read_csv('D:\\Visual Studio Code\\Neural Network Applied\\mushrooms.csv')


# In[3]:


mushroom_df.head(5)


# In[4]:


mushroom_df.shape


# In[5]:


mushroom_df.describe()


# In[6]:


mushroom_df.groupby(['class','odor']).count()


# In[7]:


labels= mushroom_df['class']
features= mushroom_df.drop(columns=['class'])


# In[8]:


labels[0:5]


# In[9]:


features[0:5]


# In[10]:


labels.replace('p', 0, inplace=True)
labels.replace('e', 1, inplace=True)
labels[0:5]


# In[11]:


features=pd.get_dummies(features)
features[0:10]


# In[12]:


features=features.values.astype('float32')
labels=labels.values.astype('float32')


# In[13]:


features_train, features_test, labels_train, labels_test=train_test_split(features, labels, test_size=0.2)
features_train, features_validation, labels_train, labels_validation= train_test_split(features_train, labels_train, test_size=0.2)


# In[14]:


model= keras.Sequential([keras.layers.Dense(32, input_shape=(117,)),
                        keras.layers.Dense(20, activation=tf.nn.relu),
                        keras.layers.Dense(2, activation='softmax')])


# In[15]:


model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['acc'])


# In[16]:


history = model.fit(features_train, labels_train, epochs=20, validation_data=(features_validation, labels_validation))


# In[17]:


prediction_features = model.predict(features_test)
performance = model.evaluate(features_test, labels_test)
print(performance)


# In[18]:


history_dict = history.history
history_dict.keys()


# In[22]:


#Checking Overfit
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs= range(1, len(acc)+1)
#"bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
#b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Ephocs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:




