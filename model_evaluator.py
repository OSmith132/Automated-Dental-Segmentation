from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader
from monai.losses import DiceLoss
from segment_anything import SamPredictor
import numpy as np
import torch

class ModelEvaluator:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.predictor = SamPredictor(model)
        self.device = device
        self.loss_fn = DiceLoss(sigmoid=True)
        

    def evaluate_metrics(masks, gt_masks):
        
        num_images = len(masks)
        num_classes = int(np.max([np.max(mask) for mask in gt_masks]) + 1)  # Find the number of classes
        
        # Initialize a dictionary to store metrics for each class
        class_metrics = {class_id: {
            "IoU": 0,
            "Precision": 0,
            "Recall": 0,
            "F1 Score": 0,
            "Dice Score": 0,
            "MCC": 0
        } for class_id in range(num_classes)}
        
        for mask, gt_mask in zip(masks, gt_masks):

            # Ensure both mask and gt_mask are NumPy arrays
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            if isinstance(gt_mask, torch.Tensor):
                gt_mask = gt_mask.cpu().numpy()

            # For multi-class masks, convert to class indices
            if mask.ndim == 3:  # For multi-channel masks (e.g., RGB or multi-class channels)
                mask = np.argmax(mask, axis=-1)  # Convert softmax output to class indices
            gt_mask = np.argmax(gt_mask, axis=-1)  # Ensure ground truth is also a single class per pixel

            # Calculate metrics for each class
            for class_id in range(num_classes):
                # Get binary masks for the current class
                mask_class = (mask == class_id).astype(int)
                gt_mask_class = (gt_mask == class_id).astype(int)
                
                # Calculate IoU
                intersection = np.sum((mask_class == 1) & (gt_mask_class == 1))
                union = np.sum((mask_class == 1) | (gt_mask_class == 1))
                iou = intersection / union if union != 0 else 0
                class_metrics[class_id]["IoU"] += iou

                # Calculate Precision
                precision = precision_score(gt_mask_class.flatten(), mask_class.flatten())
                class_metrics[class_id]["Precision"] += precision

                # Calculate Recall
                recall = recall_score(gt_mask_class.flatten(), mask_class.flatten())
                class_metrics[class_id]["Recall"] += recall

                # Calculate F1 Score
                f1 = f1_score(gt_mask_class.flatten(), mask_class.flatten())
                class_metrics[class_id]["F1 Score"] += f1

                # Calculate Dice Score
                intersection = np.sum((mask_class == 1) & (gt_mask_class == 1))
                dice = 2 * intersection / (np.sum(mask_class) + np.sum(gt_mask_class)) if (np.sum(mask_class) + np.sum(gt_mask_class)) != 0 else 0
                class_metrics[class_id]["Dice Score"] += dice

                 # Calculate MCC (Matthews Correlation Coefficient)
                mcc = matthews_corrcoef(gt_mask_class.flatten(), mask_class.flatten())
                class_metrics[class_id]["MCC"] += mcc
        

        


























    
    def evaluate_metrics_binary(masks, gt_masks):
        """
        Evaluate the model's performance on the given masks and ground truth masks.

        Args:
            masks (list of np.ndarray): List of predicted masks.
            gt_masks (list of np.ndarray): List of ground truth masks.
        
        Returns:
            dict: A dictionary containing the average metrics for IoU, Precision, Recall, F1 Score, Dice Score, and MCC.
        """
        num_images = len(masks)
        metrics = {
            "IoU": 0,
            "Precision": 0,
            "Recall": 0,
            "F1 Score": 0,
            "Dice Score": 0,
            "MCC": 0
        }
        
        for mask, gt_mask in zip(masks, gt_masks):

            # Ensure both mask and gt_mask are NumPy arrays
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            if isinstance(gt_mask, torch.Tensor):
                gt_mask = gt_mask.cpu().numpy()

            if mask.ndim == 3:  # For multi-channel masks (e.g., RGB or multi-class channels)
                mask = np.argmax(mask, axis=-1)  # Convert softmax output to class indices
            gt_mask = np.argmax(gt_mask, axis=-1)  # Ensure ground truth is also a single class per pixel

            

            # Calculate IoU
            intersection = np.sum((mask == 1) & (gt_mask == 1))
            union = np.sum((mask == 1) | (gt_mask == 1))
            iou = intersection / union if union != 0 else 0
            metrics["IoU"] += iou

            # Calculate Precision
            precision = precision_score(gt_mask.flatten(), mask.flatten())
            metrics["Precision"] += precision

            # Calculate Recall
            recall = recall_score(gt_mask.flatten(), mask.flatten())
            metrics["Recall"] += recall

            # Calculate F1 Score
            f1 = f1_score(gt_mask.flatten(), mask.flatten())
            metrics["F1 Score"] += f1

            # Calculate MCC (Matthews Correlation Coefficient)
            mcc = matthews_corrcoef(gt_mask.flatten(), mask.flatten())
            metrics["MCC"] += mcc

            # Calculate Dice Score
            intersection = np.sum((mask == 1) & (gt_mask == 1))
            dice = 2 * intersection / (np.sum(mask) + np.sum(gt_mask)) if (np.sum(mask) + np.sum(gt_mask)) != 0 else 0
            metrics["Dice Score"] += dice

        # Calculate the average for each metric
        for key in metrics:
            metrics[key] /= num_images

        return metrics





