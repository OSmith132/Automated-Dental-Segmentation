import os
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm  # Progress bar

# Define class IDs
class_mapping = {
    "braces":  1,
    "bridge":  2,
    "cavity":  3,
    "crown":   4,
    "filling": 5,
    "implant": 6,
    "lesion":  7,
    "null":    0,  # Default
}






# Define max ammount of images per class (undersampling) for this split:
max_images_per_class = [None,None,None]








def clear_directory(directory):
    """
    Clears all files in the given directory.

    Args:
        directory (str): Path to the directory to clear.
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Remove the file









def extract_individual_masks_from_coco(dataset_path, splits):
    """
    Extracts individual masks for each segmented object from COCO annotations.
    Saves each object's mask separately in the 'individual_masks' folder within the dataset split directory.

    Args:
        dataset_path (str): Path to the dataset folder.
        splits (list): List of dataset splits to process (e.g. ["train", "valid", "test"]).
    """
    for max_class_count_idx, split in enumerate(splits):
        # Define paths
        split_path = os.path.join(dataset_path, split)  # test, train, valid folders
        json_file = os.path.join(split_path, "_annotations.coco.json")  # COCO annotation file
        mask_dir = os.path.join(split_path, "individual_masks")  # Output directory for individual object masks
        os.makedirs(mask_dir, exist_ok=True)  # Ensure the directory exists


        # Clear existing files in the 'individual_masks' folder before extracting new ones (for undersampling)
        if os.path.exists(mask_dir):  # Check if the directory exists
            clear_directory(mask_dir)  # Remove all existing files


        # Define class IDs for this split
        class_count = [0,0,0,0,0,0,0]

        # Define max class count for this split:
        max_images = max_images_per_class[max_class_count_idx]



        # Load COCO annotations
        if not os.path.exists(json_file):
            print(f"Annotation file not found: {json_file}")
            continue

        with open(json_file, "r") as file:
            coco_data = json.load(file)

        print(f"Extracting individual masks for {split}...")

        # Parse image and annotation info
        images = {image['id']: image for image in coco_data['images']}
        annotations = coco_data['annotations']
        categories = {category['id']: category['name'] for category in coco_data['categories']}  # Map category_id to class name

        # Process each image
        for img_info in tqdm(images.values()):
            height, width = img_info['height'], img_info['width']

            # Filter annotations for the current image
            img_annotations = [ann for ann in annotations if ann['image_id'] == img_info['id']]

            # Process each object separately
            for obj_idx, annotation in enumerate(img_annotations):
                # Get the class name
                class_name = categories.get(annotation['category_id'], 'null')

                # Get the class ID (color)
                class_id = class_mapping.get(class_name, class_mapping["null"])


                



                # If undersampling skip all objects belonging to a class that is full
                if max_images:

                    if class_count[class_id - 1] >= max_images:
                        continue

                    class_count[class_id - 1] += 1
                    print("Saving Class: ",class_id - 1)




                # Create a black mask for this specific object
                obj_mask = np.zeros((height, width), dtype=np.uint8)

                # Decode segmentation polygons
                if 'segmentation' in annotation and isinstance(annotation['segmentation'], list):
                    for polygon in annotation['segmentation']:
                        pts = np.array(polygon, dtype=np.int32).reshape((-1, 2))
                        cv2.fillPoly(obj_mask, [pts], color=class_id)  # Fill the polygon with class_id or 1

                # Save mask with unique name
                if obj_mask.max() > 0:  # Check if there is any non-zero value
                    mask_filename = f"{os.path.splitext(img_info['file_name'])[0]}_obj{obj_idx}.png" # _class{class_id}
                    mask_path = os.path.join(mask_dir, mask_filename)
                    Image.fromarray(obj_mask).save(mask_path)

        print(f"{split} individual object masks saved in {mask_dir}")


# Root dataset folder
dataset_path = "Datasets/Dental project.v19i.coco-1"
splits = ["test", "train", "valid"]

# Call function to extract individual object masks
extract_individual_masks_from_coco(dataset_path, splits)