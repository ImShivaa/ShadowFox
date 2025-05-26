import os
import requests
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Create folders if not exist
def create_dirs():
    for split in ['train', 'val']:
        for label in ['cats', 'dogs']:
            path = f'dataset/{split}/{label}'
            os.makedirs(path, exist_ok=True)
            print(f"Created folder: {path}")

# 2. Download and save image from URL
def save_image(url, path):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize((150, 150))
        img.save(path)
        print(f"Saved image: {path}")
    except Exception as e:
        print(f"Error saving {url}: {e}")

# 3. Reliable URLs for sample cat and dog images
cat_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Felis_catus-cat_on_snow.jpg/320px-Felis_catus-cat_on_snow.jpg"
]

dog_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Golde33443.jpg/320px-Golde33443.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Golden_Retriever_Carlos_%2810543150996%29.jpg/320px-Golden_Retriever_Carlos_%2810543150996%29.jpg"
]

# 4. Setup folders
create_dirs()

# 5. Save images into train (70%) and val (30%) folders
def download_and_save(urls, label):
    split_index = int(len(urls) * 0.7)
    for i, url in enumerate(urls[:split_index]):
        save_image(url, f'dataset/train/{label}/{label}_{i}.jpg')
    for i, url in enumerate(urls[split_index:]):
        save_image(url, f'dataset/val/{label}/{label}_{i}.jpg')

download_and_save(cat_urls, 'cats')
download_and_save(dog_urls, 'dogs')

# 6. List saved images in folders
def list_images():
    base_dir = 'dataset'
    for split in ['train', 'val']:
        for label in ['cats', 'dogs']:
            path = os.path.join(base_dir, split, label)
            print(f"\nImages in {path}:")
            if os.path.exists(path):
                files = os.listdir(path)
                if files:
                    for img_file in files:
                        print("  ", img_file)
                else:
                    print("  No images found!")
            else:
                print("  Folder not found!")

list_images()

# 7. Parameters for data loading
img_height, img_width = 150, 150
batch_size = 2

# 8. Data generators with rescaling
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    'dataset/val',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# 9. Define simple CNN model
model = models.Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(2, activation='softmax')  # 2 classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 10. Train model
history = model.fit(train_gen, validation_data=val_gen, epochs=3)

# 11. Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
