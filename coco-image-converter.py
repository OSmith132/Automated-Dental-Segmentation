import os
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm # Progress bar



# Define class IDs 
class_mapping = {
    "braces":  1,
    "bridge":  2,
    "cavity":  3,
    "crown":   4,
    "filling": 5,
    "implant": 6,
    "lesion":  7,
    "null":    0 ,  # Default
}



# Define max ammount of images per class (undersampling) for this split:
max_images_per_class = [None,None,None] # test, train, valid



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




# Creates segmentation masks from COCO annotations (output saved as PNG in the "masks" folder)
def create_masks_from_coco(dataset_path, splits):

    for max_class_count_idx, split in enumerate(splits):
        # Define paths
        split_path = os.path.join(dataset_path, split)                 # test, train, valid folders
        json_file = os.path.join(split_path, "_annotations.coco.json") # COCO annotation file
        mask_dir = os.path.join(split_path, "masks")                   # Output directory for masks
        os.makedirs(mask_dir, exist_ok=True)                           # Ensure the directory exists


        

         # Clear existing files in the 'individual_masks' folder before extracting new ones (for undersampling)
        if os.path.exists(mask_dir):  # Check if the directory exists
            clear_directory(mask_dir)  # Remove all existing files


        # Define class IDs for this split
        class_count = [0,0,0,0,0,0,0]

        # Define max class count for this split:
        max_images = max_images_per_class[max_class_count_idx]





        # Load COCO annotations
        if (not os.path.exists(json_file)):
            print(f"Annotation file not found: {json_file}")
            continue

        with open(json_file, "r") as file:
            coco_data = json.load(file)



        # Output loading message
        print(f"Creating masks for {split}...")


        # Parse image and annotation info
        images = {image['id']: image for image in coco_data['images']}
        annotations = coco_data['annotations']
        categories = {category['id']: category['name'] for category in coco_data['categories']}  # Map category_id to class name


        # loop through all images and create masks (with a loading bar!)
        for img_info in tqdm(images.values()):

            # Create black mask same size as imageS
            height, width = img_info['height'], img_info['width']
            mask = np.zeros((height, width), dtype=np.uint8)

            # Filter annotations for the current image
            img_annotations = [annotation for annotation in annotations if annotation['image_id'] == img_info['id']]


            # Sort annotations by area in descending order
            img_annotations.sort(key=lambda annotation: annotation['area'], reverse=True)


            # Apply all relevant annotations
            for annotation in img_annotations:

                # Get the class name
                class_name = categories.get(annotation['category_id'], 'null')

                # Get the class ID (colour)
                class_id = class_mapping.get(class_name, class_mapping["null"])






                # If undersampling skip all objects belonging to a class that is full
                if max_images:

                    if class_count[class_id - 1] >= max_images:
                        continue

                    class_count[class_id - 1] += 1
                    print("Saving Class: ",class_id - 1)





                # Decode segmentation polygons
                if ('segmentation' in annotation and isinstance(annotation['segmentation'], list)):
                    for polygon in annotation['segmentation']:
                        pts = np.array(polygon, dtype=np.int32).reshape((-1, 2))

                        # Fill the polygon with correct colour 
                        cv2.fillPoly(mask, [pts], color=class_id)


            if mask.max() > 0:
                # Save mask 
                mask_filename = os.path.splitext(img_info['file_name'])[0] + "-segmentation-mask.png"
                mask_path = os.path.join(mask_dir, mask_filename)
                Image.fromarray(mask).save(mask_path)

        print(f"{split} masks successfully createdd and saved to {mask_dir}")




# Root dataset folder
dataset_path = "Datasets/Dental project.v19i.coco-1"
splits = ["test", "train", "valid"]

# Call function to create masks
create_masks_from_coco(dataset_path, splits)