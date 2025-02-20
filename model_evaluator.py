import torch
import numpy as np
from tqdm import tqdm
from monai.metrics import DiceMetric
from torchmetrics import JaccardIndex, Precision, Recall, F1Score, MatthewsCorrCoef



class ModelEvaluator:
    def __init__(self, model, processor, test_dataset):
        self.model = model
        self.processor = processor
        self.test_dataset = test_dataset

        # Initialize metrics
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.iou_metric = JaccardIndex(task="binary").to("cuda")
        self.precision_metric = Precision(task="binary").to("cuda")
        self.recall_metric = Recall(task="binary").to("cuda")
        self.f1_metric = F1Score(task="binary").to("cuda")
        self.mcc_metric = MatthewsCorrCoef(task="binary").to("cuda")

        # Store all metric results
        self.iou_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []
        self.dice_scores = []
        self.mcc_scores = []


    # Clear all metrics
    def _clear_metrics(self):
        self.iou_scores.clear
        self.precision_scores.clear
        self.recall_scores.clear
        self.f1_scores.clear
        self.dice_scores.clear
        self.mcc_scores.clear


    # Evaluate SAM and SAM based models
    def evaluate_sam_model(self):

        # Clear prev metrics
        self._clear_metrics()

        # Set model to evaluation mode so it stops traininig
        self.model.eval()

        # Iterate over test dataset
        for idx in tqdm(range(len(self.test_dataset))):

            # Get image and gt mask
            test_image = self.test_dataset[idx]["pixel_values"]
            ground_truth_mask = np.array(self.test_dataset[idx]["ground_truth_mask"])

            # Get bounding box for input
            prompt = self.test_dataset.get_bounding_box(ground_truth_mask)

            # Prepare image & box prompt
            inputs = self.processor(test_image, input_boxes=[[prompt]], return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs, multimask_output=False)

            # Apply sigmoid to get probability mask
            medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1)).cpu().numpy().squeeze()

            # Convert to binary mask
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)  # Ensure binary GT mask

            # Convert to tensors for MONAI & torch
            medsam_seg_tensor = torch.tensor(medsam_seg, dtype=torch.float32).unsqueeze(0).to("cuda")
            ground_truth_mask_tensor = torch.tensor(ground_truth_mask, dtype=torch.float32).unsqueeze(0).to("cuda")

            # Compute metrics
            self.dice_scores.append(self.dice_metric(medsam_seg_tensor.unsqueeze(0), ground_truth_mask_tensor.unsqueeze(0)).item())
            self.iou_scores.append(self.iou_metric(medsam_seg_tensor, ground_truth_mask_tensor).item())
            self.precision_scores.append(self.precision_metric(medsam_seg_tensor, ground_truth_mask_tensor).item())
            self.recall_scores.append(self.recall_metric(medsam_seg_tensor, ground_truth_mask_tensor).item())
            self.f1_scores.append(self.f1_metric(medsam_seg_tensor, ground_truth_mask_tensor).item())
            self.mcc_scores.append(self.mcc_metric(medsam_seg_tensor, ground_truth_mask_tensor).item())

        return self.compute_average_metrics()


    # Average and returnn all 
    def compute_average_metrics(self):
        avg_metrics = {
            "IoU":        np.mean(self.iou_scores),
            "Precision":  np.mean(self.precision_scores),
            "Recall":     np.mean(self.recall_scores),
            "F1 Score":   np.mean(self.f1_scores),
            "Dice Score": np.mean(self.dice_scores),
            "MCC":        np.mean(self.mcc_scores),
        }

        return avg_metrics


    # Output stored results
    def print_results(self):
        avg_metrics = self.compute_average_metrics()
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.4f}")



    # SAM mask generator evaluation
    def evaluate_mask_generator(self):
        self.iou_scores.clear()
        self.precision_scores.clear()
        self.recall_scores.clear()
        self.f1_scores.clear()
        self.dice_scores.clear()
        self.mcc_scores.clear()
        
        for idx in tqdm(range(len(self.test_dataset))):
            test_image = self.test_dataset[idx]["pixel_values"]
            ground_truth_mask = np.array(self.test_dataset[idx]["ground_truth_mask"])

            # Generate masks using the automatic mask generator
            generated_masks = self.model.generate(test_image)
            
            # Convert generated masks into a single binary mask
            generated_mask = np.zeros_like(ground_truth_mask, dtype=np.uint8)
            for mask in generated_masks:
                generated_mask = np.logical_or(generated_mask, mask["segmentation"]).astype(np.uint8)

            # Ensure binary format for ground truth
            ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)
            
            # Convert to tensors for MONAI & TorchMetrics
            generated_mask_tensor = torch.tensor(generated_mask, dtype=torch.float32).unsqueeze(0).to("cuda")
            ground_truth_mask_tensor = torch.tensor(ground_truth_mask, dtype=torch.float32).unsqueeze(0).to("cuda")
            
            # Compute metrics
            self.dice_scores.append(self.dice_metric(generated_mask_tensor.unsqueeze(0), ground_truth_mask_tensor.unsqueeze(0)).item())
            self.iou_scores.append(self.iou_metric(generated_mask_tensor, ground_truth_mask_tensor).item())
            self.precision_scores.append(self.precision_metric(generated_mask_tensor, ground_truth_mask_tensor).item())
            self.recall_scores.append(self.recall_metric(generated_mask_tensor, ground_truth_mask_tensor).item())
            self.f1_scores.append(self.f1_metric(generated_mask_tensor, ground_truth_mask_tensor).item())
            self.mcc_scores.append(self.mcc_metric(generated_mask_tensor, ground_truth_mask_tensor).item())
            
        return self.compute_average_metrics()


