import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
import os
import glob

tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), 'GPU')

def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(64, 3, padding='same'),
        tf.keras.layers.Conv2D(3, 3, padding='same')
    ])

def load_image(path):
    path = tf.convert_to_tensor(path, dtype=tf.string)
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [64, 64])
    return tf.image.convert_image_dtype(img, tf.float16)

def create_validation_dataset(base_val_dir):
    val_datasets = [
        os.path.join(base_val_dir, "BSD300", "test"),
        os.path.join(base_val_dir, "Kodak"),
        os.path.join(base_val_dir, "Set14")  
    ]
    
    extensions = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
    
    val_files = []
    for dataset_path in val_datasets:
        if not os.path.exists(dataset_path):
            print(f"Validation path missing: {dataset_path}")
            continue
        for ext in extensions:
            val_files.extend(glob.glob(os.path.join(dataset_path, f"*{ext}")))
    
    if not val_files:
        raise ValueError("No validation images found!")
    
    print(f"Loaded {len(val_files)} validation images")
    
    ds = tf.data.Dataset.from_tensor_slices(val_files)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds.batch(8).prefetch(tf.data.AUTOTUNE)

def create_dataset(data_dir, batch_size=8):
    patterns = [os.path.join(data_dir, ext) for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    
    if not file_list:
        raise ValueError(f"No images found in {data_dir}")
    
    ds = tf.data.Dataset.from_tensor_slices(file_list)
    ds = ds.map(
        lambda x: load_image(x), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.map(
        lambda x: (x, x),  
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def main():
    train_ds = create_dataset("./dataset/ImageNet", batch_size=16)
    val_ds = create_validation_dataset("./validation")

    model = create_model()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.MeanAbsoluteError()
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "denoise_model.h5", save_best_only=True
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=[early_stopping, lr_scheduler, checkpoint]
    )
if __name__ == "__main__":
    main()
