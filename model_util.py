import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model


model = load_model("best_model.h5")

class_names = ["Brown_Swiss", "Gir", "Holstein_Friesian", "Jersey", 
               "Murrah", "Ongole", "Sahiwal", "Tharparkar"]


def predict_breed_from_folder(folder_path, target_size=(224, 224)):
    images = []
    
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            images.append(img_array)

    if not images:
        raise ValueError("No valid images found in the folder!")

    batch = np.vstack(images)
    preds = model.predict(batch)

    avg_pred = np.mean(preds, axis=0)
    top_indices = avg_pred.argsort()[-3:][::-1]
    top_scores = avg_pred[top_indices]
    top_classes = [class_names[i] for i in top_indices]

    return list(zip(top_classes, top_scores))



results = predict_breed_from_folder(r"D:\Desktop\model\sahiwal")
print(results)