import os
from PIL import Image

"""
Clase Label mappings
{
  "other": 1,
  "murmur": 2,
  "normal": 3,
}
"""

def create_dataset(root_dir):
    """
    Creates a dataset dictionary for image classification
    Args:
        root_dir (str): The root directory containing subdirectories for each label.

    Returns:
        list: A list of dictionaries containing image data, file path, and labels.
    """
    dataset = []
    labels = os.listdir(root_dir)

    for label_id, label in enumerate(labels):
        label_dir = os.path.join(root_dir, label)
        if not os.path.isdir(label_dir):
            continue

        label_images = os.listdir(label_dir)

        for image_name in label_images:
            image_path = os.path.join(label_dir, image_name)
            if not image_path.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                continue

            # Open and process the image (you can resize or preprocess it here)
            image = Image.open(image_path)
            # For example, resizing the image to a fixed size (e.g., 224x224)
            image = image.resize((224, 224))

            # Create a dictionary entry for the image
            image_dict = {
                'image': image,
                'image_file_path': image_path,
                'label': label_id
            }

            dataset.append(image_dict)

    return dataset

root_directory = '/Users/kayle/Projects/Python/audio/data_dir/cwt_scalograms_data'
dataset = create_dataset(root_directory)

print(dataset[5::-1])

'''
{
  "other": 1,
  "murmur": 2,
  "normal": 3,
}

'''
heart_dataset = dataset #list of dictionaries

