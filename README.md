# Automated Safety Risk Assessment for Crane Operations Using Cascade Learning
**Abstract**\
Construction machinery enhances productivity and ensures project timelines. However, equipment failure poses significant risks, including injuries, fatalities, and financial losses. Traditional safety assessments rely on manual reporting and are prone to errors, delays, and inconsistencies. This study introduced a cascade learning technique for automated safety risk assessment in crane operations, ensuring reliable, accurate, and adaptable evaluations. The cascade model detects cranes, classifies safety statuses and activities, and computes risk values using confidence scores and impact factors. A risk threshold of 0.52 triggers real-time alerts for intervention. Video-feed analysis supports continuous monitoring and documentation. Expert validation confirmed the practicality of the risk-quantification models. The model achieved 92.10% precision in crane detection, 99.25% accuracy in safety classification, and 99.47% accuracy in activity classification, with an inference time of 0.70 seconds. This approach enhances Smart Site Safety System (4S) technologies, automates safety assessments, and contributes to improved construction safety standards.\

**Keywords:** Smart site safety system, Cascade learning, Computer vision, Crane operational risk classification, Safety risk assessment.

**Project Page and Details**\
This repository contains scripts for all models in a cascaded pipeline (SRGAN, RT-DETR-L, DINOv2, and ViT) used in our study, "Automated Safety Risk Assessment for Crane Operations Using Cascade Learning." Our safety risk assessment encompasses risk identification, assessment, evaluation, control and mitigation, monitoring and review, and documentation. Additionally, we validated our models by comparing them with state-of-the-art models and risk values obtained through an online survey of crane operators.

#### Our safety risk assessment involves the following steps:

**a) Image Pre-processing (Optional)**\
In this step, we used the SRGAN model to enhance the resolution of CCTV images, which improved subsequent risk detection and classification.

**b) Risk Identification**\
In this step, we initially used the RT-DETR-L model to detect mobile and tower cranes. Then, we used DINOv2 to classify them into safe operations, unsafe operations, near-miss incidents, and incidents. Finally, we used the ViT model to classify these four categories into 19 classes/activities.

**c) Risk Assessment/Analysis**\
In this step, we converted confidence levels into probabilities and multiplied them by impact or consequence values.

**d) Risk Evaluation**\
In this step, we determined threshold values for all 19 activities. Activities with values greater than the threshold require control and mitigation.

**e) Risk Control/Mitigation**\
In this step, real-time warning signals were issued to crane operators. Originally, these signals were in text form, but we converted them into warning images.

**f) Risk Monitoring and Review**\
In this step, we implemented our cascaded pipeline on unseen videos to predict crane operational risks at construction sites. An example video can be viewed at [YouTube Video](https://youtu.be/xSDrxOv0iaE).

**g) Documentation**\
Finally, we documented and recorded all steps properly.

**h) Validation**\
We validated the models and risk values by comparing them with state-of-the-art models and through an online survey of crane operators.

---

## Repository Structure
```
safety-risk-assessment  
│  
├── ActivityViTModule  
│ ├── VIT_Mobile_crane_training.ipynb  
│ ├── VIT_Tower_Crane_training.ipynb  
│ ├── ViT_Tower_Crane_Confusion_Matrix_count.png  
│  
├── DetectionModule  
│ ├── Detr_training.ipynb  
│ ├── Yolov11_training.ipynb  
│ ├── yolov10_training.ipynb  
│ ├── yolov8_training.ipynb  
│ ├── yolov9_training.ipynb  
│ ├── detr_model.yaml  
│ ├── custom_data_yolo11.yaml  
│ └── detr-results.ipynb  
│  
├── PipelineCode  
│ └── safety_risk_assessment_piepline_code.ipynb  # Unified pipeline for running all models on an image  
│  
├── SRGanModule  
│ ├── srgan_code_training.ipynb  
│ ├── srgan_model.ipynb  
│ ├── Srgan_data_generation.ipynb  
│ └── srgan_code_results.ipynb  
│  
├── SafetyStatusDinoModule  
│ ├── training_dino_multi_class_classification.ipynb  
│ ├── testing_dinov2_multi_class_classification.ipynb  
│ ├── confusion_matrix_multi-class.png  
│ └── classification_dataset/  
│  
├── inference  
│ ├── core/  
│ ├── models/  
│ ├── utils/  
│ ├── .env  
│ ├── download_models.py  
│ └── inference.py  
│  
├── .gitignore  
├── README.md  
├── requirements.txt  
```
Each module contains dedicated notebooks for **model-specific training, testing, and visualization**.  
The `inference/` folder provides a unified inference pipeline for testing trained models on videos.

---

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Mudasir1214/safety-risk-assessment.git
   cd safety-risk-assessment
   ```
2. **Create and Activate Conda Environment**
   ```bash
   conda create -n sra_env python=3.10
   conda activate sra_env
   ```
3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```
---

## Usage
#### Training
For training each model, we have provided separate modules with their respective training notebook files:

- **SRGAN:** `SRGanModule/srgan_code_training.ipynb`

- **Detection (RT-DETR / YOLO):** `DetectionModule/Detr_training.ipynb`

- **Safety Classification (DINOv2):**
  `SafetyStatusDinoModule/training_dino_multi_class_classification.ipynb`

- **Activity Classification (ViT):**
    - `ActivityViTModule/VIT_Mobile_crane_training.ipynb`
    - `ActivityViTModule/VIT_Tower_Crane_training.ipynb`

You can open any of these `.ipynb` files in Jupyter Notebook or VS Code to train specific components.


#### Inference and Testing

1. **Download Trained Models**
```bash
python inference/download_models.py
```
This automatically downloads the models and saves them in the `'all_saved_models'` folder (path configured in .env).

2. **Run Inference on Video**
```bash
python inference/inference.py --input_video video_name.mp4 --save_results True --show_results True
```
   - `--input_video:` Path to input video file

   - `--save_results:` Set to True to save annotated output videos

   - `--show_results:` Set to True to display results during inference
  
Output videos (if saved) will appear in the configured Output Dir `output_results`.


4. **Example Demo Video**\
A demonstration video of the inference pipeline in action is available [Demo Results Video](https://youtu.be/xSDrxOv0iaE)


## License
This repository is licensed under the [MIT License](LICENSE).
You are free to use, share, and adapt this work, provided appropriate credit is given.

## Acknowledgment
We acknowledge Kim et al. (2024) for their foundational work on equipment condition detection frameworks, which inspired the structural design of this study.


## Citation
If you use this work or its findings in your research, please cite:

> M. Hussain, *Automated Safety Risk Assessment for Crane Operations Using Cascade Learning*, 2026.  
> Available at: [https://github.com/Mudasir1214/safety-risk-assessment](https://github.com/Mudasir1214/safety-risk-assessment)

