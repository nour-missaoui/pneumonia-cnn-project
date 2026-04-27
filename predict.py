import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("cnn_pneumonia_model.h5")

img_path = sys.argv[1]

img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    print(f"PNEUMONIA probability: {prediction:.2%}")
else:
    print(f"NORMAL probability: {(1 - prediction):.2%}")