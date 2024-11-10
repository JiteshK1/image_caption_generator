from flask import Flask, request, jsonify
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import os
import cv2
from flask_cors import CORS

# Load the model, processor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning", trust_remote_code=True)
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Caption generation parameters
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Function to predict captions for a single image
def predict_caption(image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    pixel_values = processor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()

# Flask app setup
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/after', methods=['POST'])
def after():
    # Save and process the image
    file = request.files['file']
    file_path = 'static/file.jpg'
    file.save(file_path)

    # Image pre-processing (if needed)
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # Generate caption
    caption = predict_caption(file_path)

    return jsonify({'caption': caption})

if __name__ == "__main__":
    app.run(debug=True)
