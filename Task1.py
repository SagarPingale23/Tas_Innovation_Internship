'''ARTIFICIAL INTELLIGENCE INTERNSHIP

TASK-1

Image Classification

Here are the simplified steps to perform a Image Classification:

1. Gather and prepare a diverse dataset of labeled images for training. 

2. Choose and implement a suitable machine learning model for image classification.

3. Train the model using the prepared dataset and optimize its parameters for accuracy.

4. Validate the model's performance by evaluating it on a separate validation dataset.

5. Fine-tune the model and iterate on the training process to improve classification accuracy.

6. Test the trained model on new, unseen images to assess its realworld performance.'''
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_images, train_labels, epochs=10, validation_split=0.2)


test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_accuracy)


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()


predictions = model.predict(test_images[:5])
for i in range(5):
    predicted_label = class_names[predictions[i].argmax()]
    true_label = class_names[test_labels[i][0]]
    print(f"Predicted: {predicted_label}, True: {true_label}")
