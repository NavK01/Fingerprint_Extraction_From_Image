import cv2
import numpy as np
from ultralytics import YOLO

# Load the saved model

model_path = ""  #model is private and can be accessed by requesting access
# from the author

model = YOLO(model_path)

def extract_fingerprint(image_path):

    # Predict on the input image
    results = model.predict(
        source=image_path,
        conf=0.25,
        iou=0.7,
        save=True
    )
    # Segmentation results are saved in runs\segment\ 

    # Access the first Results object (for a single image)
    results = results[0]

    # Extract the original image and convert from BGR to RGB
    orig_img = results.orig_img
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # Extract masks and boxes
    masks = results.masks
    boxes = results.boxes

    # Check if masks and boxes exist
    if masks is None or boxes is None:
        print("No objects detected in the image.")
        exit()

    # Loop through detected objects
    for i, (mask, box) in enumerate(zip(masks.data, boxes.xyxy)):
        # Convert mask to numpy array (binary mask: 1 for object, 0 for background)
        mask_np = mask.cpu().numpy()

        # Ensure mask matches the original image dimensions
        if mask_np.shape != orig_img.shape[:2]:
            mask_np = cv2.resize(mask_np, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Shrink the mask by 10 pixels from all sides
        shrink = 10
        kernel = np.ones((2 * shrink + 1, 2 * shrink + 1), np.uint8)
        shrunk_mask_np = cv2.erode(mask_np, kernel, iterations=1)

        # Create a black background image with the same size as the original
        black_bg = np.zeros_like(orig_img)


        # --- Elliptical cropping with shrunken coordinates ---
        # Adjust bounding box coordinates by shrinking 10 pixels from each side
        x1, y1, x2, y2 = map(int, box)
        x1 = max(x1 + shrink, 0)
        y1 = max(y1 + shrink, 0)
        x2 = min(x2 - shrink, orig_img.shape[1])
        y2 = min(y2 - shrink, orig_img.shape[0])
        width = x2 - x1
        height = y2 - y1

        # Define ellipse parameters based on shrunken bounding box
        center_x = x1 + width // 2
        center_y = y1 + height // 2
        major_axis = int(height * 1.0)  # 100% of height for full vertical coverage
        minor_axis = int(width * 0.95)  # 95% of width for wider left and right

        # Create an elliptical mask
        ellipse_mask = np.zeros_like(mask_np)
        cv2.ellipse(
            ellipse_mask,
            center=(center_x, center_y),
            axes=(minor_axis // 2, major_axis // 2),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=1,
            thickness=-1
        )

        # Combine elliptical mask with the shrunken mask to ensure only segmented region is included
        ellipse_mask = ellipse_mask * shrunk_mask_np

        # Create a 3-channel elliptical mask
        ellipse_mask_3ch = np.stack([ellipse_mask] * 3, axis=-1).astype(np.uint8)

        # Apply elliptical mask to original image
        ellipse_masked_img = orig_img * ellipse_mask_3ch
        ellipse_output_img = black_bg + ellipse_masked_img
    
        # Convert RGB to grayscale
        gray_img = cv2.cvtColor(ellipse_output_img, cv2.COLOR_RGB2GRAY)


        # Enhance contrast using histogram equalization
        gray_img = cv2.equalizeHist(gray_img)

        # Apply adaptive thresholding to create a binary (B&W) image
        # Adjust blockSize and C for optimal ridge separation
        block_size = 25  # Size of the pixel neighborhood (adjust based on image resolution)
        C = 15  # Constant subtracted from mean (adjust for contrast)
        binary_img = cv2.adaptiveThreshold(
            gray_img,
            maxValue=255,  # Max pixel value for binary image
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,  # Black ridges, white background
            blockSize=block_size,
            C= C
        )

        # Optional: Apply morphological operations to refine ridges
        kernel = np.ones((3, 3), np.uint8)  # Small kernel for subtle refinement
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=1)  # Close small gaps
        binary_img = cv2.dilate(binary_img, kernel, iterations=1)  # Thicken ridges slightly

        # Save the B&W image
        output_path = "Final_Image.png"
        cv2.imwrite(output_path, binary_img)
        print(f"Saved B&W fingerprint image to {output_path}")

        