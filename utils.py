import os
import re
import io
import zipfile
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import asyncio

# WinRT Imports (may raise ImportError on non-Windows)
try:
    from winrt.windows.storage import StorageFile, FileAccessMode
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.graphics.imaging import BitmapDecoder
    WINRT_AVAILABLE = True
except ImportError:
    WINRT_AVAILABLE = False

# Pytesseract Import
try:
    import pytesseract
    from pytesseract import Output
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

# Class Definitions
SSD_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

YOLO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Model URL Paths
SSD_PROTOTXT_URL = "https://raw.githubusercontent.com/robmarkcole/object-detection-app/master/model/MobileNetSSD_deploy.prototxt.txt"
SSD_CAFFEMODEL_URL = "https://raw.githubusercontent.com/robmarkcole/object-detection-app/master/model/MobileNetSSD_deploy.caffemodel"
YOLO_ONNX_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"

def download_models(models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    
    ssd_proto = os.path.join(models_dir, "MobileNetSSD_deploy.prototxt")
    ssd_model = os.path.join(models_dir, "MobileNetSSD_deploy.caffemodel")
    yolo_model = os.path.join(models_dir, "yolov5n.onnx")
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Download SSD Prototxt
    if not os.path.exists(ssd_proto):
        print("Downloading MobileNet-SSD prototxt...")
        req = urllib.request.Request(SSD_PROTOTXT_URL, headers=headers)
        with urllib.request.urlopen(req) as response, open(ssd_proto, 'wb') as out_file:
            out_file.write(response.read())
            
    # Download SSD Caffemodel
    if not os.path.exists(ssd_model):
        print("Downloading MobileNet-SSD caffemodel...")
        req = urllib.request.Request(SSD_CAFFEMODEL_URL, headers=headers)
        with urllib.request.urlopen(req) as response, open(ssd_model, 'wb') as out_file:
            out_file.write(response.read())
            
    # Download YOLOv5 ONNX
    if not os.path.exists(yolo_model):
        print("Downloading YOLOv5-Nano ONNX model...")
        req = urllib.request.Request(YOLO_ONNX_URL, headers=headers)
        with urllib.request.urlopen(req) as response, open(yolo_model, 'wb') as out_file:
            out_file.write(response.read())

    return ssd_proto, ssd_model, yolo_model

# Image Pre-processing functions
def grayscale_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def blur_image(img, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def threshold_image(img, thresh_val=127, use_otsu=False):
    gray = grayscale_image(img) if len(img.shape) == 3 else img
    if use_otsu:
        val, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh, int(val)
    else:
        val, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        return thresh, thresh_val

def deskew_image(img):
    gray = grayscale_image(img) if len(img.shape) == 3 else img
    # Threshold to binary inverted
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Grab coordinates of all non-zero pixels
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return img, 0.0
        
    angle = cv2.minAreaRect(coords)[-1]
    
    # minAreaRect returns angle in range [-90, 0)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # Rotate the image
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated, angle

def get_histogram_plot(gray_img, thresh_val=None):
    fig, ax = plt.subplots(figsize=(6, 2.2))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.tick_params(colors='#888888', labelsize=8)
    ax.xaxis.label.set_color('#888888')
    ax.yaxis.label.set_color('#888888')
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    ax.spines['top'].set_color('#333333')
    ax.spines['right'].set_color('#333333')
    
    ax.hist(gray_img.ravel(), 256, [0, 256], color='#00f2fe', alpha=0.7)
    ax.set_xlim([0, 256])
    ax.set_title("Pixel Intensity Distribution", color='white', fontsize=10, pad=10)
    
    if thresh_val is not None:
        ax.axvline(x=thresh_val, color='#ff007f', linestyle='--', linewidth=1.5, label=f'Threshold ({thresh_val})')
        ax.legend(facecolor='#0e1117', labelcolor='white', framealpha=0.8, edgecolor='#333333', fontsize=8)
        
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf

# OCR Wrapper Functions
async def run_uwp_ocr(image_path_or_bytes):
    if not WINRT_AVAILABLE:
        return "UWP OCR is only supported on Windows.", []
        
    # If bytes, write to temporary file because WinRT StorageFile requires file path
    temp_file = False
    if isinstance(image_path_or_bytes, bytes):
        import tempfile
        temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp.write(image_path_or_bytes)
        temp.close()
        image_path = temp.name
        temp_file = True
    else:
        image_path = os.path.abspath(image_path_or_bytes)
        
    try:
        file = await StorageFile.get_file_from_path_async(image_path)
        stream = await file.open_async(FileAccessMode.READ)
        decoder = await BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        
        engine = OcrEngine.try_create_from_user_profile_languages()
        if not engine:
            return "OCR Engine could not be initialized.", []
            
        result = await engine.recognize_async(bitmap)
        words_data = []
        for line in result.lines:
            for word in line.words:
                rect = word.bounding_rect
                words_data.append({
                    'text': word.text,
                    'x': rect.x,
                    'y': rect.y,
                    'w': rect.width,
                    'h': rect.height
                })
        return result.text, words_data
    except Exception as e:
        return f"UWP OCR Error: {e}", []
    finally:
        if temp_file and os.path.exists(image_path):
            os.remove(image_path)

def run_tesseract_ocr(img, tesseract_cmd=None, psm=3):
    if not PYTESSERACT_AVAILABLE:
        return "pytesseract package is not installed.", []
        
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
    try:
        custom_config = f'--psm {psm}'
        # Extract dictionary containing coordinates
        data = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
        text = pytesseract.image_to_string(img, config=custom_config)
        
        words_data = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0: # confidence filter
                txt = data['text'][i].strip()
                if txt:
                    words_data.append({
                        'text': txt,
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'w': data['width'][i],
                        'h': data['height'][i]
                    })
        return text, words_data
    except Exception as e:
        return f"Tesseract OCR Error: {e}\nMake sure Tesseract is installed and path is set in sidebar.", []

# Google Lens Highlight Function
def highlight_words(img, words_data, search_term):
    if not search_term or not words_data:
        return img, 0
        
    overlay = img.copy()
    search_term = search_term.lower().strip()
    found_count = 0
    
    for word in words_data:
        if search_term in word['text'].lower():
            x, y, w, h = int(word['x']), int(word['y']), int(word['w']), int(word['h'])
            # Draw overlay box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 242, 254), -1)
            # Draw border line
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 150, 255), 2)
            found_count += 1
            
    if found_count > 0:
        alpha = 0.35
        # Blend the image and overlay
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
    return img, found_count

# Bounding Box drawing helper
def draw_bbox(img, x_min, y_min, x_max, y_max, label, confidence, color=(0, 242, 254), thickness=2):
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    # Background label box
    lbl = f"{label}: {confidence:.1%}"
    (lbl_w, lbl_h), base = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_min_lbl = max(y_min, lbl_h + 10)
    cv2.rectangle(img, (x_min, y_min_lbl - lbl_h - 10), (x_min + lbl_w + 10, y_min_lbl), color, -1)
    # Write text label in white/black
    cv2.putText(img, lbl, (x_min + 5, y_min_lbl - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img

# Deep Learning Object Detection (SSD)
def detect_objects_ssd(img, confidence_threshold=0.8):
    proto_path = "models/MobileNetSSD_deploy.prototxt"
    model_path = "models/MobileNetSSD_deploy.caffemodel"
    
    if not os.path.exists(proto_path) or not os.path.exists(model_path):
        return img, [], "MobileNet-SSD weights not found. Please click Download in Sidebar."
        
    try:
        net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        (h, w) = img.shape[:2]
        
        # Construct 4D blob: scale=0.007843, size=300x300, mean=127.5
        blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        
        output_img = img.copy()
        found_objects = []
        
        # Loop through detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter detections by confidence >= threshold (milestone standard)
            if confidence >= confidence_threshold:
                idx = int(detections[0, 0, i, 1])
                label = SSD_CLASSES[idx]
                
                # Compute coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Clip box to image boundaries
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w - 1, endX)
                endY = min(h - 1, endY)
                
                # Crop area
                crop = img[startY:endY, startX:endX]
                
                found_objects.append({
                    'class': label,
                    'confidence': confidence,
                    'box': (startX, startY, endX - startX, endY - startY),
                    'crop': crop
                })
                
                # Draw bounding box
                output_img = draw_bbox(output_img, startX, startY, endX, endY, label, confidence)
                
        return output_img, found_objects, None
    except Exception as e:
        return img, [], f"SSD Detection Error: {e}"

# Deep Learning Object Detection (YOLO ONNX)
def detect_objects_yolo(img, confidence_threshold=0.8):
    model_path = "models/yolov5n.onnx"
    
    if not os.path.exists(model_path):
        return img, [], "YOLO ONNX weights not found. Please click Download in Sidebar."
        
    try:
        net = cv2.dnn.readNetFromONNX(model_path)
        (h, w) = img.shape[:2]
        
        # YOLOv5 input: size=640x640, scale=1/255, swapRB=True
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        predictions = net.forward()
        
        # Output shape is [1, 25200, 85]
        rows = predictions[0]
        
        boxes = []
        confidences = []
        class_ids = []
        
        # Factor to scale back coordinates
        x_factor = w / 640.0
        y_factor = h / 640.0
        
        for r in rows:
            conf = r[4]
            if conf >= confidence_threshold:
                # Class probabilities
                classes_scores = r[5:]
                class_id = np.argmax(classes_scores)
                class_score = classes_scores[class_id]
                
                # Combine confidence and class score
                combined_score = conf * class_score
                if combined_score >= confidence_threshold:
                    class_ids.append(class_id)
                    confidences.append(float(combined_score))
                    
                    x, y, width, height = r[0], r[1], r[2], r[3]
                    
                    # Convert center-based box coordinates to corner-based
                    left = int((x - width/2) * x_factor)
                    top = int((y - height/2) * y_factor)
                    width = int(width * x_factor)
                    height = int(height * y_factor)
                    
                    boxes.append([left, top, width, height])
                    
        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.45)
        
        output_img = img.copy()
        found_objects = []
        
        # If predictions were found after NMS
        if len(indices) > 0:
            for idx in indices.flatten():
                box = boxes[idx]
                left = max(0, box[0])
                top = max(0, box[1])
                width = min(w - left - 1, box[2])
                height = min(h - top - 1, box[3])
                
                class_id = class_ids[idx]
                label = YOLO_CLASSES[class_id]
                confidence = confidences[idx]
                
                crop = img[top:top+height, left:left+width]
                
                found_objects.append({
                    'class': label,
                    'confidence': confidence,
                    'box': (left, top, width, height),
                    'crop': crop
                })
                
                output_img = draw_bbox(output_img, left, top, left+width, top+height, label, confidence, color=(0, 255, 127))
                
        return output_img, found_objects, None
    except Exception as e:
        return img, [], f"YOLO Detection Error: {e}"

# NLP Document Parsers
def parse_invoice(text):
    total = None
    date = "Not Found"
    email = "Not Found"
    phone = "Not Found"
    
    # Currency values like $120.50, INR 5,000, etc.
    amounts = re.findall(r'(?:\$|USD|Rs\.?|INR|€|£)\s*(\d{1,3}(?:[.,]\d{3})*(?:\.\d{2})?)', text, re.IGNORECASE)
    if not amounts:
        amounts = re.findall(r'\b\d+\.\d{2}\b', text)
        
    if amounts:
        try:
            # Clean commas and parse
            float_amounts = []
            for a in amounts:
                clean_val = a.replace(',', '')
                float_amounts.append(float(clean_val))
            total = max(float_amounts)
        except:
            pass
            
    # Search for date DD-MM-YYYY, YYYY-MM-DD or Month DD, YYYY
    date_match = re.search(r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b', text)
    if date_match:
        date = date_match.group(0)
    else:
        date_match2 = re.search(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', text, re.IGNORECASE)
        if date_match2:
            date = date_match2.group(0)
            
    # Search for email address
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    if email_match:
        email = email_match.group(0)
        
    # Search for phone number
    phone_match = re.search(r'\b(?:\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b', text)
    if phone_match:
        phone = phone_match.group(0)
        
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    vendor = "Unknown Vendor"
    for line in lines:
        # Ignore common titles/metadata
        if not any(term in line.lower() for term in ["invoice", "bill", "tax", "date", "page", "receipt"]):
            vendor = line
            break
            
    return {
        'Vendor / Store': vendor,
        'Invoice Date': date,
        'Total Amount': f"${total:.2f}" if total else "Not Found",
        'Email Address': email,
        'Phone Number': phone
    }

def parse_business_card(text):
    email = "Not Found"
    phone = "Not Found"
    website = "Not Found"
    
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    if email_match:
        email = email_match.group(0)
        
    phone_match = re.search(r'\b(?:\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b', text)
    if phone_match:
        phone = phone_match.group(0)
        
    web_match = re.search(r'\b(?:https?://)?(?:www\.)?[\w\.-]+\.(?:com|org|net|co|in|info|tech|io)\b', text, re.IGNORECASE)
    if web_match:
        website = web_match.group(0)
        
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    name = "Not Found"
    title_company = "Not Found"
    
    if len(lines) > 0:
        name = lines[0]
    if len(lines) > 1:
        title_company = lines[1]
        
    return {
        'Contact Name': name,
        'Title / Company': title_company,
        'Phone Number': phone,
        'Email Address': email,
        'Website URL': website
    }

def parse_id_card(text):
    dob = "Not Found"
    id_num = "Not Found"
    
    # DOB
    dob_match = re.search(r'(?:DOB|Birth|Born|Date of Birth)[:\s]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})', text, re.IGNORECASE)
    if dob_match:
        dob = dob_match.group(1)
        
    # Document number matching patterns
    id_match = re.search(r'(?:ID|No|Number|License)[:\s]*([A-Z0-9-]{6,16})', text, re.IGNORECASE)
    if not id_match:
        # Search for alpha-numeric patterns
        id_match = re.search(r'\b[A-Z][0-9]{7,8}\b|\b[0-9]{4}-[0-9]{4}-[0-9]{4}\b', text)
        
    if id_match:
        id_num = id_match.group(1) if len(id_match.groups()) > 0 else id_match.group(0)
        
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    name = "Not Found"
    
    # Look for Name: or Name keyword
    for line in lines:
        if 'name' in line.lower() and ':' in line:
            name = line.split(':')[-1].strip()
            break
            
    if name == "Not Found" and len(lines) > 0:
        for line in lines:
            if not any(term in line.lower() for term in ["license", "id card", "identity", "government"]):
                name = line
                break
                
    return {
        'Document Holder': name,
        'Date of Birth': dob,
        'Document / License ID': id_num
    }

# ─── NEW ENHANCEMENT UTILITIES ─────────────────────────────────────────────────

def apply_clahe(img, clip_limit=2.0, tile_grid=(8, 8)):
    """CLAHE - Contrast Limited Adaptive Histogram Equalization"""
    gray = grayscale_image(img) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)

def sharpen_image(img):
    """Unsharp mask sharpening kernel"""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def get_color_channels_plot(img):
    """Separate R, G, B channel intensity histograms on dark background"""
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.spines['bottom'].set_color('#21262d')
    ax.spines['left'].set_color('#21262d')
    ax.spines['top'].set_color('#21262d')
    ax.spines['right'].set_color('#21262d')

    if len(img.shape) == 3 and img.shape[2] == 3:
        bgr = img
        colors = [('#f85149', 2), ('#3fb950', 1), ('#58a6ff', 0)]  # R, G, B in BGR order
        labels = ['Red Channel', 'Green Channel', 'Blue Channel']
        for (color, ch_idx), label in zip(colors, labels):
            hist = cv2.calcHist([bgr], [ch_idx], None, [256], [0, 256])
            ax.plot(hist, color=color, alpha=0.85, linewidth=1.5, label=label)
    else:
        gray = img if len(img.shape) == 2 else grayscale_image(img)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        ax.plot(hist, color='#8b949e', linewidth=1.5, label='Grayscale')

    ax.set_xlim([0, 256])
    ax.set_title('RGB Channel Intensity Distribution', color='#e6edf3', fontsize=10, pad=8)
    ax.legend(facecolor='#0d1117', labelcolor='#c9d1d9', framealpha=0.8,
              edgecolor='#21262d', fontsize=8, loc='upper right')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='#0d1117')
    plt.close(fig)
    buf.seek(0)
    return buf

def get_class_distribution_chart(objects):
    """Horizontal bar chart of detected class frequencies"""
    if not objects:
        return None
    from collections import Counter
    counts = Counter([o['class'] for o in objects])
    classes = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(6, max(2, len(classes) * 0.5)))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#8b949e', labelsize=9)
    ax.spines['bottom'].set_color('#21262d')
    ax.spines['left'].set_color('#21262d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    bars = ax.barh(classes, values, color='#1f6feb', alpha=0.85, height=0.55)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                str(val), va='center', color='#e6edf3', fontsize=9, fontweight='bold')
    ax.set_xlabel('Count', color='#8b949e', fontsize=9)
    ax.set_title('Detected Class Distribution', color='#e6edf3', fontsize=10, pad=8)
    ax.set_xlim(0, max(values) + 1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='#0d1117')
    plt.close(fig)
    buf.seek(0)
    return buf

def get_text_analytics(raw_text):
    """Returns word frequency top-10 bar chart + basic stats dict"""
    import re as _re
    from collections import Counter
    words = _re.findall(r'\b[a-zA-Z]{3,}\b', raw_text.lower())
    stopwords = {'the','and','for','that','this','with','are','was','from',
                 'you','have','not','they','but','has','been','will','all',
                 'one','had','our','your','its','can','more','also','into'}
    filtered = [w for w in words if w not in stopwords]
    freq = Counter(filtered)
    top10 = freq.most_common(10)

    stats = {
        'Total Characters': len(raw_text),
        'Total Words': len(words),
        'Unique Words': len(set(filtered)),
        'Lines': len([l for l in raw_text.split('\n') if l.strip()]),
        'Top Word': top10[0][0].title() if top10 else 'N/A',
        'Avg Word Length': f"{(sum(len(w) for w in words) / len(words)):.1f}" if words else '0'
    }

    chart_buf = None
    if top10:
        labels = [w for w, _ in top10[::-1]]
        vals = [c for _, c in top10[::-1]]
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.spines['bottom'].set_color('#21262d')
        ax.spines['left'].set_color('#21262d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        bars = ax.barh(labels, vals, color='#3fb950', alpha=0.8, height=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    str(v), va='center', color='#e6edf3', fontsize=8)
        ax.set_xlabel('Frequency', color='#8b949e', fontsize=8)
        ax.set_title('Top 10 Words', color='#e6edf3', fontsize=10, pad=8)
        ax.set_xlim(0, max(vals) + 2)
        chart_buf = io.BytesIO()
        plt.savefig(chart_buf, format='png', bbox_inches='tight', dpi=150, facecolor='#0d1117')
        plt.close(fig)
        chart_buf.seek(0)

    return stats, chart_buf

# Batch ZIP Generator
def create_batch_zip(processed_items):
    """
    processed_items: list of dicts { 'filename': str, 'img_bytes': bytes, 'text': str, 'metadata': dict }
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        # Create CSV metadata
        csv_content = "Filename,OCR_Text_Snippet,Metadata_Summary\n"
        
        for item in processed_items:
            fname = item['filename']
            # Save annotated image
            img_name = f"images/processed_{fname}"
            zip_file.writestr(img_name, item['img_bytes'])
            
            # Save raw OCR text
            txt_name = f"text/ocr_{os.path.splitext(fname)[0]}.txt"
            zip_file.writestr(txt_name, item['text'])
            
            # CSV line
            snippet = item['text'][:50].replace('\n', ' ').replace(',', ';')
            meta_sum = str(item['metadata']).replace(',', ';')
            csv_content += f"{fname},{snippet},{meta_sum}\n"
            
        zip_file.writestr("metadata.csv", csv_content)
        
    zip_buffer.seek(0)
    return zip_buffer.getvalue()
