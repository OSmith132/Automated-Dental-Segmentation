from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm 
import os
from PIL import Image
import matplotlib.pyplot as plt


# Holds information and gives access to a dataset spliit of images. Uses on-the-fly loading.
class ImageMaskDataset(Dataset):
    def __init__(self, dataset_path, split):
        """
        Args:
            dataset_path (str): Path to the dataset.
            split (str): Split of the dataset (e.g., 'train', 'test').
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.dataset_path = dataset_path
        self.split = split
        self.transform = transforms.ToTensor() #transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
        
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
    
    # Return item by index []
    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]

        # Load image and mask on-the-fly
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

        # Convert numpy arrays to PIL Images for transformations
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        # Apply transformation if defined
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        
        return image, mask


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






    def compare_image_masks(self,idx,compare_masks):
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



        # Plot the image and masks
        plt.figure(figsize=(15, 5))
        plt.axis("off")  # Remove axis

        # Show image
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title("Original Image")

        # Show mask
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray') # viridis also works well
        plt.axis('off')
        plt.title("Ground Truth Mask")

        # Show generated Mask
        plt.subplot(1, 3, 3)
        mask_array = compare_masks[0]['segmentation']
        plt.imshow(mask_array, cmap='gray')
        plt.axis("off")
        plt.title("SAM Generated Mask")

        plt.show()






















