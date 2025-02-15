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

# Holds information and gives access to a dataset spliit of images. Uses on-the-fly loading.
class ImageMaskDataset(Dataset):
    def __init__(self, dataset_path, split, processor):
        """
        Args:
            dataset_path (str): Path to the dataset.
            split (str): Split of the dataset (e.g., 'train', 'test').
            sam_checkpoint (str): Path to the local SAM model checkpoint.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.processor = processor
        self.dataset_path = dataset_path
        self.split = split

        self.transform = transforms.ToTensor()  # Apply default transformation

        # Get the dirs for images and masks
        self.split_dir = os.path.join(dataset_path, split)
        self.mask_dir = os.path.join(self.split_dir, "masks")
        
        # Get list of image-mask pairs
        self.image_mask_pairs = []

        # Skip if directory doesn't exist
        if not os.path.exists(self.split_dir) or not os.path.exists(self.mask_dir):
            print(f"No directory found {self.split_dir}")
            return

        # Process each image in the split directory
        self.image_mask_pairs = [
            (os.path.join(self.split_dir, filename), os.path.join(self.mask_dir, filename.replace(".jpg", "-segmentation-mask.png")))
            for filename in tqdm(sorted(os.listdir(self.split_dir)))
            if filename.endswith(".jpg") and os.path.isfile(os.path.join(self.split_dir, filename))
        ]
        

        

    # Return number of images in the dataset split
    def __len__(self):
        return len(self.image_mask_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]

        # Load image and mask on-the-fly
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

        # # Convert numpy arrays to PIL Images for transformations
        # image = Image.fromarray(image)
        # mask = Image.fromarray(mask)

        # # Preprocess image using SAM processor
        # inputs = self.processor(images=image, return_tensors="pt")  # Process the image to tensors
        # image = inputs["pixel_values"].squeeze(0)  # Remove batch dimension added by default

        # # Resize the mask to match the image size
        # resize_transform = transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.NEAREST)
        # mask = resize_transform(mask)

        # # Convert mask to tensor (no normalization)
        # mask = torch.tensor(np.array(mask), dtype=torch.long)  # Ensure mask is long type for segmentation

        # Return as a dictionary
        return {
            "pixel_values": image,
            "ground_truth_mask": mask
        }





    def show_image_mask(self, idx):
        """
        Display images and masks along with their shapes.
        """
        img_path, mask_path = self.image_mask_pairs[idx]

        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

        # Convert numpy arrays to PIL Images
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        # Apply transformation to the image and mask
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Convert image and mask back to numpy arrays for displaying
        image = image.permute(1, 2, 0).numpy()  # Convert tensor back to HWC (numpy) format
        mask = mask.squeeze(0).numpy()  # Squeeze to remove unused channel

        # Plot the image and mask
        plt.figure(figsize=(15, 5))
        plt.title(os.path.basename(img_path), pad=20, fontsize=12)
        plt.axis("off")  # Remove axis



        # Show image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title("Image", fontsize=10)
        plt.figtext(0.30, 0.15, f"Image Shape: {image.shape}", ha='center')

        # Show mask
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray') # viridis also works well
        plt.axis('off')
        plt.title("Mask", fontsize=10)
        plt.figtext(0.73, 0.15, f"Mask Shape: {mask.shape}", ha='center')

        plt.show()




    def compare_image_masks(self, idx, compare_masks):
        """
        Display the original image, ground truth mask, and SAM-generated mask 
        with a black background and randomly colored segments.
        """
        img_path, mask_path = self.image_mask_pairs[idx]

        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
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
        sam_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)  # Black background

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








    # Taken directly from SAM's Github repo
    def show_anns(self,idx, anns):

        img_path= self.image_mask_pairs[idx][0]

        plt.figure(figsize=(10,10))
        plt.imshow(cv2.imread(img_path))

        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)
        

        plt.axis('off')
        plt.show() 






















