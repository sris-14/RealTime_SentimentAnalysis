# from keras.models import load_model

# model = load_model('sequential_saved_model.h5')
# print(model.summary())

from keras.models import load_model

try:
    model = load_model('sequential_saved_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error: {e}")


# import tensorflow as tf
# import keras as k 
# print(tf.__version__)
# print(k.__version__)