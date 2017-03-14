from PIL import Image
import numpy as np


def saveImage(image, i):
	data = ((image[i]/7.0+0.5)*255).astype(np.uint8)
	img = Image.fromarray(data, 'RGB')
	img.save('a' + str(i) + ".png")