
# Semantic Segmentation Model for 2D Dental Image Analysis. <br> Oliver Smith UoL Dissertation Project 2024/25.


## Background Rationale:

Radiology is integral to modern healthcare, supporting diagnosis and treatment across various medical fields. However, the demand for radiological services has surged, outpacing the availability of trained professionals. This intense workload increases the risk of diagnostic errors while limiting global access to medical imaging.  

AI-powered solutions like this project aim to address these challenges by automating image analysis, reducing workload, and expanding access to diagnostic tools worldwide. The proposed model will use CNN-based segmentation for precise identification of dental structures. Through robust data preparation and augmentation techniques, the model seeks to generalize effectively to real-world clinical environments.  

---

## Aim:
Develop a semantic segmentation model using U-Net architecture, which identifies and categorises teeth, fillings, implants, and other dental structures from 2D dental images. This project addresses the need for automated dental analysis, offering a reliable tool for dental professionals to aid in diagnosis and treatment. The system will be trained on labelled dental images and validated through metrics like pixel accuracy and IoU score to ensure strong performance and reliability in clinical applications.

---

## Objectives:
1. **Data Collection:** Collect and preprocess a dataset of at least 200 labeled 2D dental images containing teeth, fillings, implants, and other dental structures.  
2. **Model Development:** Develop a U-Net-based semantic segmentation model with a target pixel accuracy of >80%.  
3. **Data Augmentation:** Apply techniques such as scaling, blurring, and exposure adjustment to improve model generalization.  
4. **Hyperparameter Tuning:** Optimize segmentation accuracy through loss function adjustments, optimizer selection, and convolutional filter tuning.  
5. **User Testing:** Organize evaluation sessions with dental professionals to assess practical value and gather feedback.  
6. **Literature Review:** Explore existing research on deep learning models applied to dental images, highlighting current solutions and limitations.  
7. **Documentation:** Present findings from development, testing, and evaluation in the final dissertation paper.  

---



## Dataset Used:
* https://universe.roboflow.com/dental-ai-psmzh/dental-project-kzwsz/dataset/19  (COCO JSON)
