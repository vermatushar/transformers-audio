import os
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import joblib


# from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.models import Model
# import matplotlib.pyplot as plt
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# import numpy as np


# Image preprocessing
def preprocess(train_data_dir, valid_data_dir, test_data_dir):
    img_height, img_width = (224, 224)
    batch_size = 32

    '''train_data_dir = r"/content/drive/MyDrive/data/data_dir/cwt_scalograms_dataprocessed/train"
    valid_data_dir = r"/content/drive/MyDrive/data/data_dir/cwt_scalograms_dataprocessed/val"
    test_data_dir = r"/content/drive/MyDrive/data/data_dir/cwt_scalograms_dataprocessed/test"
    '''

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       validation_split=0.4)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')
    # set as training data

    valid_generator = train_datagen.flow_from_directory(
        valid_data_dir,  # same directory as training data
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')
    # set as validation data

    test_generator = train_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode='categorical',
        subset='validation')
    # set as test data

    return train_generator, test_generator, valid_generator


def model_trainer(epochs = None, **kwargs):
    train_data_dir = r"/Users/kayle/Projects/Python/audio/data_dir/cwt_scalograms_dataprocessed/train"
    test_data_dir = r"/Users/kayle/Projects/Python/audio/data_dir/cwt_scalograms_dataprocessed/test"
    valid_data_dir = r"/Users/kayle/Projects/Python/audio/data_dir/cwt_scalograms_dataprocessed/val"

    train_generator, test_generator, valid_generator = preprocess(
        train_data_dir=train_data_dir,
        test_data_dir=test_data_dir,
        valid_data_dir=valid_data_dir)

    x, y = test_generator.next()
    print(x.shape)
    print(f'Number of classification classes : {train_generator.num_classes}')

    base_model = ResNet50(include_top=False,
                          weights="imagenet")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(train_generator.num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint_callback = ModelCheckpoint('/Users/kayle/Projects/Python/audio/models/resnet/best_model.h5',
                                          monitor='val_loss', save_best_only=True, verbose=1)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    model.fit(
        train_generator,
        epochs=10,
        # steps_per_epoch=steps_per_epoch,
        validation_data=valid_generator,  # Replace with your validation dataset
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    '''
    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for i, (x_batch, y_batch) in enumerate(train_generator):
            try:
                # Train on the current batch
                loss, accuracy = model.train_on_batch(x_batch, y_batch)
                print(f"Batch {i + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            except Exception as e:
                # Handle the error (e.g., skip the problematic image)
                print(f"Error in batch {i + 1}: {str(e)}")

                problem_image_bytes = x_batch[0]  # Assuming the problematic image is the first in the batch
                problem_image = Image.open(io.BytesIO(problem_image_bytes))
                problem_image.save(f'problematic_image_{epoch + 1}_{i + 1}.png')

                continue
    '''
    return model


if __name__ == '__main__':

    scalogram_model = model_trainer(epochs = 10)
    scalogram_model.save('/Users/kayle/Projects/Python/audio/models/resnet/scalogram_model.h5')


    # Save the Keras model as a pickle file
    model_filename = '/Users/kayle/Projects/Python/audio/models/resnet/scalogram_model.pkl'
    joblib.dump(scalogram_model, model_filename)

    print(f'Model saved as {model_filename}')
