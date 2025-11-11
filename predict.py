import argparse
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt


with open('label_map.json', 'r') as f:
    class_Names = json.load(f)

def process_image(image):

    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0 
    
    return image.numpy()




def predict(image_path, model, top_k ):
    image = Image.open(image_path)
    image = np.asarray(image)
    
    processed_image = process_image(image)
    
    processed_image = np.expand_dims(processed_image, axis=0)
    
    preds = model.predict(processed_image)
    
    top_k_indices = np.argsort(preds[0])[-top_k:][::-1]
    top_k_probs = preds[0][top_k_indices]
    top_k_classes = [str(i) for i in top_k_indices]
    
    return top_k_probs, top_k_classes


def view_classify(image_path, probs, classes, class_names):
    
    flower_names = [class_names[str(i)] for i in classes]
    
    image = Image.open(image_path)
    
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 6), nrows=1, ncols=2)
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(flower_names[0], fontsize=14)
    
    
    ax2.barh(flower_names, probs, color='darkblue', edgecolor='black')
    ax2.set_xlim(0, 1.0)
    ax2.set_xticks(np.arange(0.0, 1.1, 0.2))  
    ax2.set_xlabel("Class Probability")
    ax2.set_title("Top K Predictions")
    
    ax2.invert_yaxis()    
    
    plt.tight_layout()
    plt.show()




def main():
    parser = argparse.ArgumentParser(description="Predict flower class from an image using a trained model.")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("model_path", help="Path to the trained Keras model (.h5)")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to JSON file mapping labels to flower names")
    
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
    else:
        
        class_names = {str(i): str(i) for i in range(102)}
    
    
    probs, classes = predict(args.image_path, model, top_k=args.top_k)
    
    
    flower_names = [class_names.get(c, c) for c in classes]
    for name, prob in zip(flower_names, probs):
        print(f"{name}: {prob:.4f}")
    
    view_classify(args.image_path, probs, classes, class_names)

if __name__ == "__main__":
    main()