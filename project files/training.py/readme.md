import os
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Dataset paths
csv_path = r"C:\Users\Mounica\Downloads\dataset-master\dataset-master\labels.csv"
image_folder = r"C:\Users\Mounica\Downloads\dataset-master\dataset-master\JPEGImages"

# 2. Load CSV and prepare DataFrame
df = pd.read_csv(csv_path)
df.columns = [col.strip().lower() for col in df.columns]
df.rename(columns={'image': 'id', 'category': 'label'}, inplace=True)

# Format filenames
df['filename'] = df['id'].apply(lambda x: f"BloodImage_{int(x):05d}.jpg")
df['filepaths'] = df['filename'].apply(lambda x: os.path.join(image_folder, x))

# Drop missing labels first
df = df.dropna(subset=['label'])

# Remove multi-label rows (with commas)
df = df[~df['label'].astype(str).str.contains(',')]

# Remove file rows that don't exist
df = df[df['filepaths'].apply(os.path.exists)].reset_index(drop=True)

# 3. Image Data Generators
img_gen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

train_gen = img_gen.flow_from_dataframe(
    df,
    x_col='filepaths',
    y_col='label',
    target_size=(224, 224),
    class_mode='categorical',
    subset='training',
    batch_size=32,
    shuffle=True
)

val_gen = img_gen.flow_from_dataframe(
    df,
    x_col='filepaths',
    y_col='label',
    target_size=(224, 224),
    class_mode='categorical',
    subset='validation',
    batch_size=32,
    shuffle=False
)

# 4. Compute Class Weights to handle imbalance
class_labels = train_gen.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(class_labels),
    y=class_labels
)
class_weight_dict = dict(enumerate(class_weights))

# 5. Build Transfer Learning Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(len(train_gen.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 6. Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Train Model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weight_dict
)

# 8. Save the Trained Model
output_path = r"C:\Users\Mounica\Desktop\hemotovision\model"
os.makedirs(output_path, exist_ok=True)
model.save(os.path.join(output_path, "Blood Cell.h5"))
print("✅ Training Complete. Model saved successfully.")

