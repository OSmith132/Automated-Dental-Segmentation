import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2

# Class IDs
class_mapping = {
    "braces":  1,
    "bridge":  2,
    "cavity":  3,
    "crown":   4,
    "filling": 5,
    "implant": 6,
    "lesion":  7,
}



# Function to analyze the dataset
def analyze_dataset(dataset_path, splits):

    # dicts for each graph
    class_counts = {class_name: 0 for class_name in class_mapping.keys()}
    pixel_coverage = {class_name: 0 for class_name in class_mapping.keys()}
    co_occurrence = {class1: {class2: 0 for class2 in class_mapping.keys()} for class1 in class_mapping.keys()} # nested dictionary for each class and class
    split_class_counts = {split: {class_name: 0 for class_name in class_mapping.keys()} for split in splits}    # store class occurrence for each split
 
  
    for split in splits:

        # Get dir of data annotations
        split_path = os.path.join(dataset_path, split)
        json_file = os.path.join(split_path, "_annotations.coco.json")

        if not os.path.exists(json_file):
            print(f"Annotation file not found: {json_file}")
            continue # move onto next split

        # Read annotation file
        with open(json_file, "r") as file:
            coco_data = json.load(file)


        print(f"Analyzing dataset for {split}...")


        annotations = coco_data['annotations']
        categories = {category['id']: category['name'] for category in coco_data['categories']}

        for annotation in tqdm(annotations):
            class_name = categories.get(annotation['category_id'], 'null')
            if class_name not in class_mapping:
                continue

            # Increment the count for the class
            class_counts[class_name] += 1
            split_class_counts[split][class_name] += 1

            # Calculate pixel coverage
            if 'segmentation' in annotation and isinstance(annotation['segmentation'], list):
                for polygon in annotation['segmentation']:
                    pts = np.array(polygon, dtype=np.int32).reshape((-1, 2))
                    area = cv2.contourArea(pts)
                    pixel_coverage[class_name] += area

        # Calculate co-occurrence for images
        image_annotations = {}
        for annotation in annotations:
            image_id = annotation['image_id']
            class_name = categories.get(annotation['category_id'], 'null')
            if class_name not in class_mapping:
                continue
            if image_id not in image_annotations:
                image_annotations[image_id] = set()
            image_annotations[image_id].add(class_name)

        for classes in image_annotations.values():
            for class1 in classes:
                for class2 in classes:
                    co_occurrence[class1][class2] += 1

    # Normalize pixel coverage to average per item
    for class_name in pixel_coverage:
        if class_counts[class_name] > 0:
            pixel_coverage[class_name] /= class_counts[class_name]

    return class_counts, pixel_coverage, co_occurrence, split_class_counts






def visualize_all(class_counts, pixel_coverage, co_occurrence, split_class_counts):

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 18)) 
    plt.subplots_adjust(hspace=0.5, wspace=0.25)
    axes = axes.flatten() # for indexing




    # Class Balance Plot
    class_names = list(class_counts.keys())
    class_values = list(class_counts.values())
    ax = sns.barplot(ax=axes[0], x=class_names, y=class_values, palette='viridis')

    # Annotate bars
    max_height = max(class_values)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    fontsize=10, color='black',
                    xytext=(0, 5), textcoords='offset points')
        
    axes[0].set_ylim(0, max_height * 1.1) # Add 10% extra space above the tallest bar
    axes[0].set_title("Class Balance in Dataset")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Number of Instances")
    axes[0].tick_params(axis='x', rotation=45)




    # Pixel Coverage Plot
    pixel_values = list(pixel_coverage.values())
    ax = sns.barplot(ax=axes[1], x=class_names, y=pixel_values, palette='magma')

    # Annotate bars and adjust y-axis limits
    max_pixel_coverage = max(pixel_values)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    fontsize=10, color='black',
                    xytext=(0, 5), textcoords='offset points')
        
    axes[1].set_ylim(0, max_pixel_coverage * 1.1) # Add 10% extra space above the tallest bar
    axes[1].set_title("Average Pixel Coverage per Class")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Average Pixel Coverage")
    axes[1].tick_params(axis='x', rotation=45)




    # Co-occurrence Matrix
    co_occurrence_matrix = np.array([[co_occurrence[class1][class2] for class2 in class_mapping.keys()] for class1 in class_mapping.keys()])
    sns.heatmap(co_occurrence_matrix, annot=True, fmt="d", ax=axes[2], xticklabels=class_mapping.keys(), yticklabels=class_mapping.keys(), cmap="coolwarm", linewidths=0.5)

    axes[2].tick_params(axis='y', rotation=0)
    axes[2].set_title("Class Co-occurrence Matrix")
    axes[2].set_xlabel("Class")
    axes[2].set_ylabel("Class")

    # Class Distribution Across Splits
    split_names = list(split_class_counts.keys())
    split_data = {split: [split_class_counts[split].get(cls, 0) for cls in class_names] for split in split_names}




    # Create a grouped bar chart
    bar_width = 0.25  # Width of each bar
    x = np.arange(len(class_names))  # Class indices
    for i, split in enumerate(split_names):
        axes[3].bar(x + i * bar_width, split_data[split], width=bar_width, label=split)

    # Add labels and legend
    axes[3].set_title("Class Distribution Across Splits")
    axes[3].set_xlabel("Class")
    axes[3].set_ylabel("Number of Instances")
    axes[3].set_xticks(x + bar_width * (len(split_names) - 1) / 2)
    axes[3].set_xticklabels(class_names, rotation=45)
    axes[3].legend(title="Splits")

    plt.show()






# Root dataset folder
dataset_path = "Datasets/Dental project.v19i.coco-1"
splits = ["test", "train", "valid"]


# Analyze the dataset
class_counts, pixel_coverage, co_occurrence, split_class_counts = analyze_dataset(dataset_path, splits)

# Visualize all the data
visualize_all(class_counts, pixel_coverage, co_occurrence, split_class_counts)
