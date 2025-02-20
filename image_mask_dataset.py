from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm 
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import SamProcessor
from segment_anything import sam_model_registry




# Gives access to a dataset split of images. Uses on-the-fly loading.
# Extends torch dataset so that it can be used by torch dataloaders
class ImageMaskDataset(Dataset):
    def __init__(self, dataset_path, split, processor):
        
        self.processor         = processor
        self.dataset_path      = dataset_path
        self.split             = split
        
        self._return_as_tensor = False # Preprocess image and mask before returng for use in fine-tuning
        self._resize_mask      = False # Resize the mask to 256x256 for comparison with output from SAM inference

        # Get the dirs for images and masks
        self.split_dir = os.path.join(dataset_path, split)
        self.mask_dir = os.path.join(self.split_dir, "masks")
        
        
        # Skip if directory doesn't exist
        if not os.path.exists(self.split_dir) or not os.path.exists(self.mask_dir):
            print(f"No directory found {self.split_dir}")
            return
        
        # Get list of image-mask pairs
        self.image_mask_pairs = []

        # Store each image path
        self.image_mask_pairs = [
            (os.path.join(self.split_dir, filename), os.path.join(self.mask_dir, filename.replace(".jpg", "-segmentation-mask.png")))
            for filename in tqdm(sorted(os.listdir(self.split_dir)))
            if filename.endswith(".jpg") and os.path.isfile(os.path.join(self.split_dir, filename))
        ]
        


    #########################
    ### GETTERS & SETTERS ###
    #########################


    @property 
    def resize_mask(self):
        return self._resize_mask

    @resize_mask.setter
    def resize_mask(self, value):
        if not isinstance(value, bool):
            raise ValueError("variable must be a boolean value")
        self._resize_mask = value


    @property
    def return_as_tensor(self):
        return self._return_as_tensor

    @return_as_tensor.setter
    def return_as_tensor(self, value):
        if not isinstance(value, bool):
            raise ValueError("variable must be a boolean value")
        self._return_as_tensor = value


    #########################
    ###      Methods      ###
    #########################



    # Return number of images in the dataset split
    def __len__(self):
        return len(self.image_mask_pairs)
    

    # Returns a seperate mask for each object. Masks created by individual_mask_generator.py
    def _find_object_masks(self, img_path, mask_dir):

        # Extract filename without extension
        img_filename = os.path.splitext(os.path.basename(img_path))[0]

        # Find all relevant mask files
        object_mask_paths = sorted([
            os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
            if f.startswith(img_filename) and f.endswith(".png")
        ])

        if not object_mask_paths:
            raise FileNotFoundError(f"No masks found for {img_filename} in {mask_dir}")

        # Load masks as binary NumPy arrays
        object_masks = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in object_mask_paths]

        return object_masks
    


    #Get bounding boxes from mask. Taken from https://github.com/bnsreenu/python_for_microscopists/blob/master/331_fine_tune_SAM_mito.ipynb
    def get_bounding_box(self, ground_truth_map):
        
        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]

        return bbox



    # Returns the image tensor as pre-processed by the given SAM Processor
    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]

        # Load image as RGB (3 channels)
        image = cv2.imread(img_path)  # Loads the image in BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Load mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale directly
        

        # Resize mask to 256x256 if required
        if self._resize_mask:
            mask = cv2.resize(mask, (256,256))
        

        # Preprocess and return tensors for use in fine-tuning
        if self._return_as_tensor:

            # object_masks = self._find_object_masks(img_path, os.path.join(os.path.dirname(os.path.dirname(mask_path)), "individual_masks") ) # Get indiviidual biinary mask for each object
            # mask = torch.tensor(mask, dtype=torch.float32)

            binary_mask = (mask > 0).astype(np.float32)  # Convert to binary (0 or 1)

            bounding_boxes = self.get_bounding_box(binary_mask)

            inputs = self.processor(images=image, input_boxes=[[bounding_boxes]],  return_tensors="pt")  # RGB image (3 channels)           # segmentation_maps=[binary_mask],

            inputs = {k:v.squeeze(0) for k,v in inputs.items()} # Remove batch dimension added by default

            inputs["ground_truth_mask"] = binary_mask

            return inputs


        # Return as a dictionary
        return {
            "pixel_values": image,
            "ground_truth_mask": mask
        }





    def show_image_mask(self, idx):

        # Get image and mask path
        img_path, mask_path = self.image_mask_pairs[idx]

        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

        # Plot the image and mask
        plt.figure(figsize=(15, 6))
        plt.title(os.path.basename(img_path), pad=20, fontsize=12)
        plt.axis("off")  # Remove axis

        # Show image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title("Image", fontsize=10)
        plt.figtext(0.30, 0.05, f"Image Shape: {image.shape}", ha='center')

        # Show mask
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray') # viridis also works well
        plt.axis('off')
        plt.title("Mask", fontsize=10)
        plt.figtext(0.73, 0.05, f"Mask Shape: {mask.shape}", ha='center')

        plt.show()




    def compare_image_masks(self, idx, compare_masks):
 
        # Get image and mask path
        img_path, mask_path = self.image_mask_pairs[idx]

        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        #image = cv2.resize(image, (1024, 1024))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Show Original Image
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title("Original Image")

        # Show Ground Truth Mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].axis('off')
        axes[1].set_title("Ground Truth Mask")

        # Create an empty black image for the SAM mask
        sam_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)  # Black background

        # Sort the masks by area (largest first)
        sorted_anns = sorted(compare_masks, key=lambda x: x['area'], reverse=True)

        for ann in sorted_anns:
            segment = ann['segmentation']
            random_color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)  # Generate random color
            sam_mask[segment] = random_color  # Assign color to the segment

        # Show SAM-generated mask
        axes[2].imshow(sam_mask)
        axes[2].axis('off')
        axes[2].set_title("SAM Generated Mask")

        plt.tight_layout()
        plt.show()




    # Overlay segmentation mask over image. Taken directly from SAM's Github repo
    def show_anns(self,idx, anns):

        # Return if no annotations provided
        if len(anns) == 0:
            return

        # Get image
        img_path = self.image_mask_pairs[idx][0]

        plt.figure(figsize=(10,10))
        plt.imshow(cv2.imread(img_path))

        # Display largest areas first
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        # Create empty image
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0

        # Add all segments from largest to smallest
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask

        # Output
        ax.imshow(img)
        plt.axis('off')
        plt.show() 






















