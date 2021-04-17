from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def curate(image_name):
	# open the image and convert to gray scale:
	im = Image.open(image_name).convert('L')

	# convert to numpy array:
	im_array = np.array(im)

	# crop:
	dim = im_array.shape
	min_dimension = min(dim[0],dim[1])
	# get x and y offsets:
	x_offset = int(abs(min_dimension - dim[0])/2)
	y_offset = int(abs(min_dimension - dim[1])/2)
	# crop based on offsets:
	(left, upper, right, lower) = x_offset,y_offset,abs(x_offset-dim[0]), abs(y_offset-dim[1])
	im_crop = im.crop((left, upper, right, lower))

	# resize to be 28 by 28:
	im_crop.thumbnail((28,28))

	new_image_name = image_name[:-4] + "_gray_small.png"
	im_crop.save(new_image_name)

	# invert image:
	inverted_im = ImageOps.invert(im_crop)

	# Convert back to array so we can scale it down:
	im_array = np.array(inverted_im)

	# scale image between 0 and 1:
	final_im = im_array / 255.0

	# return the final image:
	return final_im

# parse the data:
im1 = curate("shirt.png")
im2 = curate("sandal.png")
im3 = curate("dress.png")
im4 = curate("pullover.png")

print(im1.shape)

plt.figure()
plt.subplot(2,2,1)
plt.imshow(im1)
plt.subplot(2,2,2)
plt.imshow(im2)
plt.subplot(2,2,3)
plt.imshow(im3)
plt.subplot(2,2,4)
plt.imshow(im4)
plt.grid(False)
# plt.show()

# Load Data from MNIST

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class labels:

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

# Set up the layers:

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model:

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Feed the model:

model.fit(train_images, train_labels, epochs=10)

# Make predictions:

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

# Make a prediction on the images:

images = np.array([im1,im2,im3,im4])

# predictions_single = probability_model.predict(images)

# # print out all the predictions:
# print(predictions_single)

# loop through images and print the classifications:
for i in range(len(images)):
	img = (np.expand_dims(images[i],0))
	predictions_single = probability_model.predict(img)

	print(predictions_single)

	clothing_index = np.argmax(predictions_single[0])
	print(class_names[clothing_index])