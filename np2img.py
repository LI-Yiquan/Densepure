import numpy as np
from PIL import Image
import sys

# Load the NPZ file
path = '/home/yiquan/DensePure/purified_images/'
data = np.load(path+'testing_samples_100x32x32x3.npz')['arr_0']

# Iterate over the data and save each image
for i, img_data in enumerate(data):
    print(i)
    img = Image.fromarray(img_data.astype(np.uint8))
    img.save(f'{path}/image_{i}.png')
    