import torch
import numpy as np
from tqdm import tqdm
from monai.metrics import DiceMetric
from torchmetrics import JaccardIndex, Precision, Recall, F1Score, MatthewsCorrCoef
import torch.nn.functional as F


class ModelEvaluator:
    def __init__(self, model, processor, dataset):
        self.model = model
        self.processor = processor
        self.dataset = dataset

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

        self.dataset.return_as_sam = True

        # Clear prev metrics
        self._clear_metrics()

        # Set model to evaluation mode so it stops traininig
        self.model.eval()

        # Iterate over test dataset
        for idx in tqdm(range(len(self.dataset))):

            # Get image and gt mask
            test_image = self.dataset[idx]["pixel_values"]
            ground_truth_mask = np.array(self.dataset[idx]["ground_truth_mask"])

            # Get bounding box for input
            prompt = self.dataset.get_bounding_box(ground_truth_mask)

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


    # Average and return all 
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

        self.dataset.return_as_sam = True

        self.iou_scores.clear()
        self.precision_scores.clear()
        self.recall_scores.clear()
        self.f1_scores.clear()
        self.dice_scores.clear()
        self.mcc_scores.clear()
        
        for idx in tqdm(range(len(self.dataset))):
            test_image = self.dataset[idx]["pixel_values"]
            ground_truth_mask = np.array(self.dataset[idx]["ground_truth_mask"])

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






    @torch.no_grad()
    def medsam_inference(self, img_embed, bboxes, H, W):
        """Perform inference using MedSAM model to generate segmentation masks."""
        box_torch = torch.as_tensor(bboxes, dtype=torch.float, device=img_embed.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=self.model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

        low_res_pred = F.interpolate(
            low_res_pred,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, gt.shape)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        return medsam_seg
    




    def evaluate_medsam_model(self):
        """Evaluates MedSAM model on the test dataset."""

        # Ensure dataset returns data in MedSAM format
        self.dataset.return_as_medsam = True  

        self._clear_metrics()
        self.model.eval()  # Set model to evaluation mode

        for idx in tqdm(range(len(self.dataset)), desc="Evaluating MedSAM"):
            
            # Get correct preprocessing
            self.dataset.return_as_medsam = True

            # Get tensors
            img_np, box_np, gt_masks, bounding_boxes = self.dataset[idx].values()

            # Get original image
            self.dataset.return_as_medsam = False
            img_original = self.dataset[idx]["pixel_values"]
            W, H, _ = img_original.shape

            # image embedding
            with torch.no_grad():
                image_embedding = self.model.image_encoder(img_np)

            # Run inference for all boxes in a batch
            with torch.no_grad():
                seg_masks = self.medsam_inference(image_embedding, box_np, H, W)  # List of 5 masks


            # If only one mask then wrap in array
            if len(seg_masks.shape) == 2:
                seg_masks = [seg_masks]



            # ground_truth_mask_tensor = gt_masks


            for (seg_mask, ground_truth_mask_tensor) in zip(seg_masks, gt_masks):
                medsam_seg_tensor = torch.tensor(seg_mask, dtype=torch.float32).unsqueeze(0).to("cuda")
                ground_truth_mask_tensor = torch.tensor(ground_truth_mask_tensor, dtype=torch.float32).unsqueeze(0).to("cuda")


                # Compute metrics
                self.dice_scores.append(self.dice_metric(medsam_seg_tensor.unsqueeze(0), ground_truth_mask_tensor.unsqueeze(0)).to("cpu"))
                self.iou_scores.append(self.iou_metric(medsam_seg_tensor, ground_truth_mask_tensor).to("cpu"))
                self.precision_scores.append(self.precision_metric(medsam_seg_tensor, ground_truth_mask_tensor).to("cpu"))
                self.recall_scores.append(self.recall_metric(medsam_seg_tensor, ground_truth_mask_tensor).to("cpu"))
                self.f1_scores.append(self.f1_metric(medsam_seg_tensor, ground_truth_mask_tensor).to("cpu"))
                self.mcc_scores.append(self.mcc_metric(medsam_seg_tensor, ground_truth_mask_tensor).to("cpu"))

               # print(self.dice_metric(medsam_seg_tensor.unsqueeze(0), ground_truth_mask_tensor.unsqueeze(0)).to("cpu"))



        return self.compute_average_metrics()
