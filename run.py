import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import itertools
import numpy as np
from keras.layers import Dense
from scipy.spatial import distance
import streamlit as st
import cv2
from glob import glob
L = [i for i in glob('/home/findmine/Desktop/envatxel/pyimagesearch-ropa/Banana_Republic/Mens/Apparel/Shirt/*')]

st.title('Keras embedding vectors: cosine simmilarity')
model = tf.keras.models.load_model('model',compile=False)
IMAGE_SHAPE = (224, 224)
metric = 'cosine'
def extract(file):
  file = Image.open(file).convert('L').resize(IMAGE_SHAPE)

  file = np.stack((file,)*3, axis=-1)

  file = np.array(file)/255.0

  embedding = model.predict(file[np.newaxis, ...])
  vgg16_feature_np = np.array(embedding)
  flattended_feature = vgg16_feature_np.flatten()

  return flattended_feature

def compare(x,y):
  xb = extract(x)
  yb = extract(y)
  cosineDistance = distance.cdist([xb], [yb], metric)[0]
  return cosineDistance

def main():
    st.write('Resultados 1:1')
    for a, b in itertools.combinations_with_replacement(L[:3], 2):
        #print(D)
        D = compare(a, b)
        # Read First Image
        img1 = cv2.imread(a)
        img1 = cv2.resize(img1, (224,244))

        # Read Second Image
        img2 = cv2.imread(b)
        img2 = cv2.resize(img2, (224,244))
        
        # concatenate image Horizontally
        Hori = np.concatenate((img1, img2), axis=1)
        # concatenate image Vertically
        st.image(Hori,caption=f'DISTANCE:  {D}')

if __name__ == '__main__':
    main()

