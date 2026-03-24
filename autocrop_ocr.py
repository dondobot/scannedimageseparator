import cv2
import os
import sys
import pytesseract
import subprocess

# --- ADJUST THESE THRESHOLDS BASED ON YOUR SCANNER RESOLUTION ---
MIN_AREA = 50000        # Minimum area for a valid object (filters dust)
MAX_LABEL_AREA = 800000 # Max area for the label strip
PADDING = 15            # Pixel buffer to preserve the Polaroid border
# ----------------------------------------------------------------

def process_and_tag(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load '{image_path}'")
        return

    original = image.copy()
    img_height, img_width = original.shape[:2]
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_regions = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_AREA:  
            x, y, w, h = cv2.boundingRect(c)
            bottom_y = y + h
            valid_regions.append({'area': area, 'box': (x, y, w, h), 'bottom_y': bottom_y})

    if not valid_regions:
        print("No distinct items found. Ensure you are using a dark background behind the photos.")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_text = ""
    label_index = -1

    if len(valid_regions) > 1:
        valid_regions.sort(key=lambda item: item['bottom_y'], reverse=True)
        for i, region in enumerate(valid_regions):
            if region['area'] < MAX_LABEL_AREA:
                label_index = i
                break

    if label_index != -1:
        label_region = valid_regions.pop(label_index)
        lx, ly, lw, lh = label_region['box']
        
        lx_pad = max(0, lx - PADDING)
        ly_pad = max(0, ly - PADDING)
        lw_pad = min(img_width - lx_pad, lw + (PADDING * 2))
        lh_pad = min(img_height - ly_pad, lh + (PADDING * 2))
        
        label_roi = original[ly_pad:ly_pad+lh_pad, lx_pad:lx_pad+lw_pad]
        cv2.imwrite(f"{base_name}_label.jpg", label_roi)
        
        print("Analyzing handwritten label with Tesseract OCR...")
        raw_text = pytesseract.image_to_string(label_roi, config='--psm 6')
        label_text = raw_text.strip()
        
        if label_text:
            print(f"Success! Found text: '{label_text}'")
        else:
            print("OCR could not decipher the handwriting.")
    else:
        print("No valid label strip found. Processing everything as photos.")

    photo_files = []
    for idx, region in enumerate(valid_regions, start=1):
        px, py, pw, ph = region['box']
        
        px_pad = max(0, px - PADDING)
        py_pad = max(0, py - PADDING)
        pw_pad = min(img_width - px_pad, pw + (PADDING * 2))
        ph_pad = min(img_height - py_pad, ph + (PADDING * 2))
        
        photo_roi = original[py_pad:py_pad+ph_pad, px_pad:px_pad+pw_pad]
        
        filename = f"{base_name}_photo_{idx}.jpg"
        cv2.imwrite(filename, photo_roi)
        photo_files.append(filename)
        print(f"Saved photo: {filename}")

    if label_text and photo_files:
        print(f"Injecting metadata into {len(photo_files)} photos...")
        cmd = [
            'exiftool', 
            '-overwrite_original', 
            f'-Description={label_text}', 
            f'-Title={label_text}'
        ] + photo_files
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            print("Metadata successfully injected!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to write metadata with ExifTool: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 autocrop_ocr.py <scan_file.jpg>")
    else:
        process_and_tag(sys.argv[1])
