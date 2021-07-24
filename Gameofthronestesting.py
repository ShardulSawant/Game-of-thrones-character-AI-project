from keras.models import load_model
from keras.preprocessing import image
import numpy as np

cnn = load_model('models/gotmodelfinal.h5')

test_image = image.load_img('gameofthrones/validation/dani.jpg', target_size=(128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
print('Predicted character')
result = np.argmax(cnn.predict(test_image), axis=-1)
print(result)
if result[0] == 0:
    prediction = 'Arya Stark'
elif result[0] == 1:
    prediction = 'Daenerys Targaryen'
elif result[0] == 2:
    prediction = 'Jaime Lannister'
elif result[0] == 3:
    prediction = 'Jon Snow'
print(prediction)
