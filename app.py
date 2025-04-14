from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# Load the YOLOv8 instance segmentation model
model = YOLO("./model/best.pt")  # Use a pretrained model

TEXTURE_IMAGE_PATHS = {
    "black atlantis": "./textures/1.jpg",
    "flor di bosco": "./textures/2.jpg",
    "thunder black": "./textures/3.jpg",
    "wavy mirage": "./textures/4.jpg",
    "grap cascade": "./textures/5.jpg",
    "black lava": "./textures/6.jpg",
    "havana encore": "./textures/11.jpg",
    "verde alpi": "./textures/8.jpg",
    "alps white": "./textures/9.jpg",
    "cala antique": "./textures/10.jpg",
    "black portoro": "./textures/7.jpg",
    "royal green": "./textures/12.jpg",
}

def decode_base64_image(base64_string):
    """Decode a base64 image string into a NumPy array."""
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def encode_image_to_base64(image):
    """Encode a NumPy array image into a base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/api/process-image', methods=['POST'])
def process_image():
    try:
        # Get the uploaded kitchen image file
        kitchen_image_file = request.files.get('kitchenImage')
        texture_image_id = request.form.get('textureImageId')  # Texture image ID
        texture_type = request.form.get('textureType')  # Texture type (e.g., countertop, backsplash, wall)

        # Validate input data
        if not kitchen_image_file or not texture_image_id or not texture_type:
            return jsonify({'error': 'Invalid input data'}), 400

        # Read the uploaded kitchen image file
        kitchen_image = cv2.imdecode(np.frombuffer(kitchen_image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Load the texture image based on the ID
        texture_image_path = TEXTURE_IMAGE_PATHS.get(texture_image_id)
        if not texture_image_path or not os.path.exists(texture_image_path):
            return jsonify({'error': f'Texture image with ID "{texture_image_id}" not found'}), 400

        texture_image = cv2.imread(texture_image_path)
        if texture_image is None:
            return jsonify({'error': 'Failed to load texture image'}), 400

        # Perform instance segmentation on the kitchen image
        results = model(kitchen_image)

        # Initialize the output image (copy of original)
        output_img = kitchen_image.copy()

        # Process each detected instance
        for result in results:
            if result.masks is not None:  # Ensure masks are available
                masks = result.masks.data.cpu().numpy()  # Get masks as NumPy array
                boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes

                for i, mask in enumerate(masks):
                    mask = mask.astype(np.uint8)  # Convert to uint8 (binary mask: 0 or 1)
                    box = boxes[i].astype(int)
                    x_min, y_min, x_max, y_max = box

                    # Crop the region of interest (ROI) from the original image
                    roi = kitchen_image[y_min:y_max, x_min:x_max]

                    # Resize the texture image to match the ROI size
                    material_resized = cv2.resize(texture_image, (x_max - x_min, y_max - y_min))

                    # Apply the texture to the ROI with blending
                    roi_mask = mask[y_min:y_max, x_min:x_max]
                    roi_mask_3ch = np.stack([roi_mask] * 3, axis=2)  # Convert to 3-channel mask

                    # Blend the texture with the original ROI to preserve shadows and lighting
                    alpha = 0.7  # Blending factor (adjust as needed)
                    blended_texture = cv2.addWeighted(material_resized, alpha, roi, 1 - alpha, 0)

                    # Apply the mask to the blended texture
                    masked_texture = blended_texture * roi_mask_3ch
                    masked_original = roi * (1 - roi_mask_3ch)
                    new_roi = masked_texture + masked_original

                    # Place the new ROI back into the output image
                    output_img[y_min:y_max, x_min:x_max] = new_roi

        # Save the processed image to a temporary file
        temp_file = "processed_image.jpg"
        cv2.imwrite(temp_file, output_img)

        # Return the processed image as a file
        return send_file(temp_file, mimetype='image/jpeg')

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)