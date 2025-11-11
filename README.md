# Create Your Own Image Classifier - TensorFlow

A deep learning project that demonstrates how to build, train, and deploy your own image classifier using TensorFlow and TensorFlow Hub. This project focuses on classifying flowers from the Oxford Flowers 102 dataset. It includes both model training and a user-friendly command-line interface for predicting new images.

---

## Table of Contents
- [Features](#features)  
- [Installation](#installation)  
- [Dataset](#dataset)  
- [Model Training](#model-training)  
- [Usage](#usage)  
- [Examples](#examples)  
- [Technologies](#technologies)  
- [License](#license)  

---

## Features
- Train a deep neural network using TensorFlow 2 and MobileNetV2 feature extractor from TensorFlow Hub.  
- Apply data augmentation and early stopping to improve model generalization.  
- Save and load trained Keras models in `.h5` or TensorFlow SavedModel format.  
- Command-line tool (`predict.py`) to:
  - Predict the top K flower classes from a given image.
  - Display class probabilities in a simple plot.
  - Map numeric class labels to flower names using a JSON file.  
- Ready-to-use test images included for quick predictions.  

---

## Installation
1. Clone the repository:
\`\`\`bash
git clone https://github.com/elyasawisy/Create-Your-Own-Image-Classifier-TensorFlow.git
cd Create-Your-Own-Image-Classifier-TensorFlow
\`\`\`

2. Create a virtual environment (optional but recommended):
\`\`\`bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\\Scripts\\activate    # Windows
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

---

## Dataset
This project uses the **Oxford Flowers 102** dataset:  
- 102 flower categories  
- Includes pre-split training, validation, and test sets  
- Downloaded automatically using \`tensorflow_datasets\` in the training script  

---

## Model Training
The \`build_model.py\` script trains a deep neural network for flower classification:  
- Uses MobileNetV2 as a feature extractor (pretrained on ImageNet)  
- Adds a dropout and dense output layer for classification  
- Supports early stopping and learning rate reduction for better convergence  

Train the model by running:
\`\`\`bash
python build_model.py
\`\`\`

This will save the trained model as:
- \`models/flower_classifier.h5\` (Keras format)  
- \`models/flower_classifier_tf/\` (TensorFlow SavedModel format)  

---

## Usage
Use the \`predict.py\` script to classify new images via the command line:
\`\`\`bash
python predict.py <image_path> <model_path> [--top_k K] [--category_names label_map.json]
\`\`\`

### Options
- \`--top_k\`: Return the top K most likely classes (default: 5)  
- \`--category_names\`: Path to a JSON file mapping numeric labels to flower names  

---

## Examples
\`\`\`bash
# Predict the top 5 classes for an image
python predict.py ./test_images/wild_pansy.jpg models/flower_classifier.h5

# Return top 3 predictions
python predict.py ./test_images/wild_pansy.jpg models/flower_classifier.h5 --top_k 3

# Map numeric labels to flower names using JSON
python predict.py ./test_images/wild_pansy.jpg models/flower_classifier.h5 --top_k 3 --category_names label_map.json
\`\`\`

---

## Technologies
- Python 3.x  
- TensorFlow 2.x  
- TensorFlow Hub  
- NumPy & Matplotlib  
- PIL (Python Imaging Library)  

---

## License
This project is licensed under the MIT License.

