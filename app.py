# final : app.py and index.html to run: flask run
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64
from datetime import datetime
import random
import copy
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    """
    Preprocess the uploaded image using OpenCV
    Returns: processed image and original image
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image file")
    
    original = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding for both dark and light line drawings
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Also try Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Combine both methods
    combined = cv2.bitwise_or(adaptive_thresh, edges)
    
    # Morphological closing to fix gaps in outlines
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return closed, original

def detect_cut_parts(processed_img, original_img, min_area=1000):
    """
    Detect individual cut parts using contour detection
    Returns: list of cut parts with their properties
    """
    # Find contours
    contours, _ = cv2.findContours(
        processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    cut_parts = []
    valid_contours = []
    
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Filter out tiny contours
        if area < min_area:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Create mask for this contour
        mask = np.zeros(processed_img.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Crop the part from original image
        cropped = original_img[y:y+h, x:x+w].copy()
        cropped_mask = mask[y:y+h, x:x+w]
        
        # Apply mask to cropped image (make background white)
        cropped_with_mask = cropped.copy()
        cropped_with_mask[cropped_mask == 0] = [255, 255, 255]
        
        part_info = {
            'id': idx + 1,
            'label': f'Part {idx + 1}',
            'width': w,
            'height': h,
            'area': int(area),
            'x': int(x),
            'y': int(y),
            'cropped_image': cropped_with_mask,
            'mask': cropped_mask,
            'contour': contour,
            'approved': True  # NEW: Default to approved
        }
        
        cut_parts.append(part_info)
        valid_contours.append(contour)
    
    return cut_parts, valid_contours

def save_cut_parts(cut_parts, base_filename):
    """
    Save individual cut parts as separate images
    Returns: list of saved filenames
    """
    saved_files = []
    
    for part in cut_parts:
        filename = f"{base_filename}_part_{part['id']}.png"
        filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        cv2.imwrite(filepath, part['cropped_image'])
        saved_files.append(filename)
    
    return saved_files

def create_visualization(original_img, contours):
    """
    Create visualization with detected contours
    """
    vis_img = original_img.copy()
    
    # Draw all contours with different colors
    for idx, contour in enumerate(contours):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.drawContours(vis_img, [contour], -1, color, 2)
        
        # Add label
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(vis_img, f'Part {idx + 1}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return vis_img

class GeneticNesting:
    """
    Genetic Algorithm for optimal nesting of cut parts on fabric
    """
    def __init__(self, parts, fabric_width, fabric_height, scale_factor):
        self.parts = parts
        self.fabric_width = fabric_width
        self.fabric_height = fabric_height
        self.scale_factor = scale_factor
        
        # Convert parts to real dimensions
        self.real_parts = []
        for part in parts:
            # For drawn shapes, use real_width/real_height, for images use pixel dimensions * scale
            if 'real_width' in part and 'real_height' in part:
                real_width = part['real_width']
                real_height = part['real_height']
            else:
                real_width = part['width'] * scale_factor
                real_height = part['height'] * scale_factor
            
            real_part = {
                'id': part['id'],
                'label': part['label'],
                'width': real_width,
                'height': real_height,
                'area': real_width * real_height,
                'original_width': part['width'],
                'original_height': part['height'],
                'real_width': real_width,
                'real_height': real_height
            }
            self.real_parts.append(real_part)
        
        # Sort parts by area (largest first)
        self.real_parts.sort(key=lambda x: x['area'], reverse=True)
        
    def can_place(self, part, x, y, rotation, placed_parts):
        """Check if part can be placed at (x, y) with given rotation"""
        if rotation in [90, 270]:
            w, h = part['height'], part['width']
        else:
            w, h = part['width'], part['height']
        
        # Check fabric bounds
        if x + w > self.fabric_width or y + h > self.fabric_height:
            return False
        
        # Check overlap with placed parts
        for placed in placed_parts:
            px, py, pw, ph = placed['x'], placed['y'], placed['width'], placed['height']
            
            # Rectangle overlap check
            if not (x + w <= px or px + pw <= x or y + h <= py or py + ph <= y):
                return False
        
        return True
    
    def bottom_left_placement(self, chromosome):
        """
        Place parts using bottom-left heuristic with rotations from chromosome
        chromosome: list of rotations [0, 90, 180, 270] for each part
        """
        placed_parts = []
        
        for i, part in enumerate(self.real_parts):
            rotation = chromosome[i]
            
            if rotation in [90, 270]:
                w, h = part['height'], part['width']
            else:
                w, h = part['width'], part['height']
            
            placed = False
            
            # Try to place from bottom-left, scanning upward and rightward
            for y_pos in range(0, int(self.fabric_height), 5):
                for x_pos in range(0, int(self.fabric_width), 5):
                    if self.can_place(part, x_pos, y_pos, rotation, placed_parts):
                        placed_parts.append({
                            'id': part['id'],
                            'label': part['label'],
                            'x': x_pos,
                            'y': y_pos,
                            'width': w,
                            'height': h,
                            'rotation': rotation,
                            'original_width': part['original_width'],
                            'original_height': part['original_height'],
                            'real_width': part['real_width'],
                            'real_height': part['real_height']
                        })
                        placed = True
                        break
                if placed:
                    break
            
            if not placed:
                # Part couldn't be placed
                return None
        
        return placed_parts
    
    def calculate_fitness(self, placed_parts):
        """Calculate fitness based on utilization and compactness"""
        if placed_parts is None:
            return 0
        
        # Calculate bounding box of all parts
        if len(placed_parts) == 0:
            return 0
        
        max_x = max(p['x'] + p['width'] for p in placed_parts)
        max_y = max(p['y'] + p['height'] for p in placed_parts)
        
        # Total area used by parts
        parts_area = sum(p['width'] * p['height'] for p in placed_parts)
        
        # Fabric area actually used (bounding box)
        used_fabric_area = max_x * max_y
        
        # Fitness: maximize part density, minimize fabric usage
        if used_fabric_area > 0:
            utilization = parts_area / used_fabric_area
            fabric_efficiency = 1.0 - (used_fabric_area / (self.fabric_width * self.fabric_height))
            
            fitness = utilization * 0.7 + fabric_efficiency * 0.3
            return fitness
        
        return 0
    
    def optimize(self, population_size=50, generations=100):
        """Run genetic algorithm"""
        num_parts = len(self.real_parts)
        rotations = [0, 90, 180, 270]
        
        # Initialize population (random rotations for each part)
        population = []
        for _ in range(population_size):
            chromosome = [random.choice(rotations) for _ in range(num_parts)]
            population.append(chromosome)
        
        best_solution = None
        best_fitness = 0
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            placements = []
            
            for chromosome in population:
                placed = self.bottom_left_placement(chromosome)
                fitness = self.calculate_fitness(placed)
                fitness_scores.append(fitness)
                placements.append(placed)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = (chromosome, placed)
            
            # Selection (tournament)
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                idx1, idx2 = random.sample(range(population_size), 2)
                winner = population[idx1] if fitness_scores[idx1] > fitness_scores[idx2] else population[idx2]
                new_population.append(winner[:])
            
            # Crossover and mutation
            for i in range(0, population_size - 1, 2):
                if random.random() < 0.7:  # Crossover probability
                    point = random.randint(1, num_parts - 1)
                    new_population[i][:point], new_population[i+1][:point] = \
                        new_population[i+1][:point], new_population[i][:point]
                
                # Mutation
                if random.random() < 0.2:  # Mutation probability
                    idx = random.randint(0, num_parts - 1)
                    new_population[i][idx] = random.choice(rotations)
                
                if random.random() < 0.2:
                    idx = random.randint(0, num_parts - 1)
                    new_population[i+1][idx] = random.choice(rotations)
            
            population = new_population
        
        return best_solution

def rotate_image_and_mask(image, mask, angle):
    """Rotate image and its mask by given angle"""
    if angle == 0:
        return image, mask
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new dimensions
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    
    # Rotate image and mask
    rotated_img = cv2.warpAffine(image, matrix, (new_w, new_h), 
                                  borderMode=cv2.BORDER_CONSTANT, 
                                  borderValue=(255, 255, 255))
    rotated_mask = cv2.warpAffine(mask, matrix, (new_w, new_h), 
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=0)
    
    return rotated_img, rotated_mask

def create_nesting_visualization(
        placed_parts,
        fabric_width,
        fabric_height,
        scale_factor,
        parts_images,
        show_part_id=True,
        show_selvage=True,
        outline_only=False
    ):
    """High-clarity nesting visualization with sharp contours and optional IDs."""

    # Higher DPI = crisp visualization
    viz_scale = 4
    canvas_width = int(fabric_width * viz_scale)
    canvas_height = int(fabric_height * viz_scale)

    # Fabric canvas
    canvas = np.full((canvas_height, canvas_width, 3), 245, np.uint8)

    # ────────────────────────────────────────────────
    # Draw 10cm grid
    # ────────────────────────────────────────────────
    grid_spacing = int(10 * viz_scale)
    for x in range(0, canvas_width, grid_spacing):
        cv2.line(canvas, (x, 0), (x, canvas_height), (210, 210, 210), 1)

    for y in range(0, canvas_height, grid_spacing):
        cv2.line(canvas, (0, y), (canvas_width, y), (210, 210, 210), 1)

    # Optional selvage line (fabric edge)
    if show_selvage:
        cv2.line(canvas, (0, 0), (0, canvas_height), (50, 50, 50), 3)

    # Color palette
    colors = [
        (255, 120, 120), (120, 255, 120), (120, 120, 255),
        (255, 255, 120), (255, 120, 255), (120, 255, 255),
        (200, 160, 120), (160, 200, 120), (120, 160, 200)
    ]

    # ────────────────────────────────────────────────
    # Draw each part
    # ────────────────────────────────────────────────    
    for i, part in enumerate(placed_parts):
        px = int(part['x'] * viz_scale)
        py = int(part['y'] * viz_scale)
        pw = int(part['width'] * viz_scale)
        ph = int(part['height'] * viz_scale)
        color = colors[i % len(colors)]

        part_data = parts_images.get(part['id'])

        # Case 1: actual image + mask
        if part_data:
            img = part_data['image']
            mask = part_data['mask']

            # Rotate
            rimg, rmask = rotate_image_and_mask(img, mask, part['rotation'])

            # Scale properly
            sx = pw / rimg.shape[1]
            sy = ph / rimg.shape[0]
            s = min(sx, sy)

            rw = int(rimg.shape[1] * s)
            rh = int(rimg.shape[0] * s)

            if rw <= 0 or rh <= 0:
                continue

            simg = cv2.resize(rimg, (rw, rh), interpolation=cv2.INTER_AREA)
            smask = cv2.resize(rmask, (rw, rh), interpolation=cv2.INTER_NEAREST)

            # Center within placement box
            ox = px + (pw - rw) // 2
            oy = py + (ph - rh) // 2

            # Boundaries
            ex = min(ox + rw, canvas_width)
            ey = min(oy + rh, canvas_height)
            cw, ch = ex - ox, ey - oy

            img_crop = simg[:ch, :cw]
            mask_crop = smask[:ch, :cw]

            # Produce tinted shape (unless outline_only)
            if not outline_only:
                overlay = np.full_like(img_crop, color)
                tinted = cv2.addWeighted(img_crop, 0.65, overlay, 0.35, 0)
            else:
                tinted = img_crop.copy()

            # Composite on canvas
            region = canvas[oy:ey, ox:ex]
            mask_bool = mask_crop > 127
            if not outline_only:
                region[mask_bool] = tinted[mask_bool]
            canvas[oy:ey, ox:ex] = region

            # Draw clean outline
            contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_canvas = cnt + [ox, oy]
                cv2.drawContours(canvas, [cnt_canvas], -1, color, 2)

            # Tiny part ID in corner
            if show_part_id:
                cv2.putText(canvas, str(part['id']), (ox + 5, oy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
                cv2.putText(canvas, str(part['id']), (ox + 5, oy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        else:
            # Case 2: rectangular fallback
            if not outline_only:
                cv2.rectangle(canvas, (px, py), (px + pw, py + ph), color, -1)
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (0, 0, 0), 2)

            if show_part_id:
                cv2.putText(canvas, str(part['id']), (px + 5, py + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
                cv2.putText(canvas, str(part['id']), (px + 5, py + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return canvas



def process_drawn_shapes(shapes_data):
    """
    Process shapes drawn on canvas
    shapes_data: list of shapes with paths and dimensions
    Returns: cut_parts similar to image detection
    """
    if not shapes_data or len(shapes_data) == 0:
        return [], []
    
    # Find canvas dimensions from shapes
    all_points = []
    for shape in shapes_data:
        if shape['path']:
            all_points.extend(shape['path'])
    
    if not all_points:
        return [], []
    
    max_x = max(pt['x'] for pt in all_points) if all_points else 0
    max_y = max(pt['y'] for pt in all_points) if all_points else 0
    
    canvas_width = int(max_x) + 100
    canvas_height = int(max_y) + 100
    
    # Create blank canvas
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    cut_parts = []
    valid_contours = []
    
    for idx, shape in enumerate(shapes_data):
        if not shape['path'] or len(shape['path']) < 3:
            continue
            
        # Convert path to numpy contour
        points = np.array([[int(pt['x']), int(pt['y'])] for pt in shape['path']], dtype=np.int32)
        
        # Create mask for this shape
        mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(points)
        
        # Calculate area
        area = cv2.contourArea(points)
        
        # Crop the part
        cropped = canvas[y:y+h, x:x+w].copy()
        cropped_mask = mask[y:y+h, x:x+w]
        
        # Apply mask
        cropped_with_mask = cropped.copy()
        cropped_with_mask[cropped_mask == 0] = [255, 255, 255]
        
        # Draw the shape on cropped image
        shifted_points = points - [x, y]
        cv2.polylines(cropped_with_mask, [shifted_points], True, (0, 0, 0), 2)
        
        # Get real dimensions from shape data
        real_width = shape.get('real_width', w)
        real_height = shape.get('real_height', h)
        
        part_info = {
            'id': idx + 1,
            'label': shape.get('label', f'Part {idx + 1}'),
            'width': w,
            'height': h,
            'area': int(area),
            'x': int(x),
            'y': int(y),
            'cropped_image': cropped_with_mask,
            'mask': cropped_mask,
            'contour': points,
            'real_width': real_width,
            'real_height': real_height,
            'approved': True  # NEW: Default to approved
        }
        
        cut_parts.append(part_info)
        valid_contours.append(points)
    
    return cut_parts, valid_contours

# NEW: Calculate efficiency for manual nesting
def calculate_nesting_efficiency(placed_parts, fabric_width, fabric_height):
    """Calculate efficiency metrics for any nesting arrangement"""
    if not placed_parts or len(placed_parts) == 0:
        return {
            'utilization': 0,
            'fabric_usage': 0,
            'waste_percentage': 100,
            'total_parts_area': 0,
            'used_fabric_area': 0
        }
    
    # Calculate total parts area
    total_parts_area = sum(p['width'] * p['height'] for p in placed_parts)
    
    # Calculate bounding box
    max_x = max(p['x'] + p['width'] for p in placed_parts)
    max_y = max(p['y'] + p['height'] for p in placed_parts)
    used_fabric_area = max_x * max_y
    
    # Calculate metrics
    fabric_area = fabric_width * fabric_height
    utilization = (total_parts_area / used_fabric_area) * 100 if used_fabric_area > 0 else 0
    fabric_usage = (used_fabric_area / fabric_area) * 100 if fabric_area > 0 else 0
    waste = 100 - utilization
    
    return {
        'utilization': round(utilization, 2),
        'fabric_usage': round(fabric_usage, 2),
        'waste_percentage': round(waste, 2),
        'total_parts_area': round(total_parts_area, 2),
        'used_fabric_area': round(used_fabric_area, 2),
        'fabric_width': fabric_width,
        'fabric_height': fabric_height
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only JPG, JPEG, PNG allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{timestamp}_{os.path.splitext(filename)[0]}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}.png")
        
        file.save(upload_path)
        
        # Preprocess image
        processed_img, original_img = preprocess_image(upload_path)
        
        # Detect cut parts
        cut_parts, valid_contours = detect_cut_parts(processed_img, original_img)
        
        if len(cut_parts) == 0:
            return jsonify({'error': 'No cut parts detected. Please upload a clearer image.'}), 400
        
        # Save individual cut parts AND store in session
        saved_files = save_cut_parts(cut_parts, base_name)
        
        # Store parts data for later use (save to disk as numpy arrays)
        parts_dir = os.path.join(app.config['PROCESSED_FOLDER'], f"{base_name}_parts")
        os.makedirs(parts_dir, exist_ok=True)
        
        for part in cut_parts:
            part_file = os.path.join(parts_dir, f"part_{part['id']}.npz")
            np.savez(part_file, 
                    image=part['cropped_image'],
                    mask=part['mask'])
        
        # Create visualization
        vis_img = create_visualization(original_img, valid_contours)
        vis_filename = f"{base_name}_visualization.png"
        vis_path = os.path.join(app.config['PROCESSED_FOLDER'], vis_filename)
        cv2.imwrite(vis_path, vis_img)
        
        # Prepare response data
        parts_data = []
        for part, saved_file in zip(cut_parts, saved_files):
            parts_data.append({
                'id': part['id'],
                'label': part['label'],
                'width': part['width'],
                'height': part['height'],
                'area': part['area'],
                'image_url': f'/processed/{saved_file}',
                'approved': part['approved']  # NEW
            })
        
        return jsonify({
            'success': True,
            'original_image': f'/uploads/{base_name}.png',
            'visualization': f'/processed/{vis_filename}',
            'parts_count': len(cut_parts),
            'parts': parts_data,
            'session_id': base_name
        })
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/process-drawn', methods=['POST'])
@app.route('/process-drawn', methods=['POST'])
def process_drawn():
    """Process shapes drawn on canvas"""
    try:
        data = request.get_json()
        shapes = data.get('shapes', [])
        
        if not shapes or len(shapes) == 0:
            return jsonify({'error': 'No shapes drawn'}), 400
        
        # Process drawn shapes
        cut_parts, valid_contours = process_drawn_shapes(shapes)
        
        if len(cut_parts) == 0:
            return jsonify({'error': 'Could not process drawn shapes'}), 400
        
        # Generate session ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"drawn_{timestamp}"
        
        # Save individual cut parts
        saved_files = save_cut_parts(cut_parts, base_name)
        
        # Store parts data
        parts_dir = os.path.join(app.config['PROCESSED_FOLDER'], f"{base_name}_parts")
        os.makedirs(parts_dir, exist_ok=True)
        
        for part in cut_parts:
            part_file = os.path.join(parts_dir, f"part_{part['id']}.npz")
            np.savez(part_file, 
                    image=part['cropped_image'],
                    mask=part['mask'])
        
        # Create visualization
        vis_filename = ""
        if cut_parts and valid_contours:
            # Create a composite image
            max_x = max(p['x'] + p['width'] for p in cut_parts)
            max_y = max(p['y'] + p['height'] for p in cut_parts)
            
            vis_img = np.ones((max_y + 50, max_x + 50, 3), dtype=np.uint8) * 255
            
            for idx, contour in enumerate(valid_contours):
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.polylines(vis_img, [contour], True, color, 2)
                
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(vis_img, f'Part {idx + 1}', (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            vis_filename = f"{base_name}_visualization.png"
            vis_path = os.path.join(app.config['PROCESSED_FOLDER'], vis_filename)
            cv2.imwrite(vis_path, vis_img)
        
        # Prepare response - NOW MATCHES UPLOAD STRUCTURE
        parts_data = []
        for part, saved_file in zip(cut_parts, saved_files):
            parts_data.append({
                'id': part['id'],
                'label': part['label'],
                'width': part['width'],
                'height': part['height'],
                'area': part['area'],
                'real_width': part.get('real_width', part['width']),
                'real_height': part.get('real_height', part['height']),
                'image_url': f'/processed/{saved_file}',
                'approved': part['approved']  # ADD THIS LINE - matches upload structure
            })
        
        return jsonify({
            'success': True,
            'visualization': f'/processed/{vis_filename}' if vis_filename else '',
            'parts_count': len(cut_parts),
            'parts': parts_data,
            'session_id': base_name
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/optimize', methods=['POST'])
def optimize_nesting():
    try:
        data = request.get_json()
        
        fabric_width = float(data.get('fabric_width'))
        fabric_height = float(data.get('fabric_height'))
        scale_factor = float(data.get('scale_factor'))
        parts = data.get('parts')
        session_id = data.get('session_id')
        
        if not all([fabric_width, fabric_height, scale_factor, parts]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Load part images and masks
        parts_dir = os.path.join(app.config['PROCESSED_FOLDER'], f"{session_id}_parts")
        parts_images = {}
        
        for part in parts:
            part_file = os.path.join(parts_dir, f"part_{part['id']}.npz")
            if os.path.exists(part_file):
                loaded = np.load(part_file)
                parts_images[part['id']] = {
                    'image': loaded['image'],
                    'mask': loaded['mask']
                }
        
        # Run Genetic Algorithm
        ga = GeneticNesting(parts, fabric_width, fabric_height, scale_factor)
        best_solution = ga.optimize(population_size=50, generations=100)
        
        if best_solution is None or best_solution[1] is None:
            return jsonify({'error': 'Could not find valid placement. Try larger fabric or fewer parts.'}), 400
        
        chromosome, placed_parts = best_solution
        
        # ----------- STATISTICS CALCULATION -------------
        
        total_parts_area = sum(p['width'] * p['height'] for p in placed_parts)
        fabric_area = fabric_width * fabric_height
        
        max_x = max(p['x'] + p['width'] for p in placed_parts)
        max_y = max(p['y'] + p['height'] for p in placed_parts)
        used_fabric_area = max_x * max_y
        
        utilization = (total_parts_area / used_fabric_area) * 100 if used_fabric_area > 0 else 0
        fabric_usage = (used_fabric_area / fabric_area) * 100 if fabric_area > 0 else 0
        waste = 100 - utilization

        # ----------- NEW: EFFICIENCY CALCULATION -------------
        # Fabric efficiency = (actual used area of parts / total fabric area)
        efficiency = (total_parts_area / fabric_area) * 100 if fabric_area > 0 else 0
        
        # ----------- VISUALIZATION -------------
        nesting_img = create_nesting_visualization(
            placed_parts, 
            fabric_width, 
            fabric_height, 
            scale_factor, 
            parts_images
        )
        
        nesting_filename = f"{session_id}_nesting.png"
        nesting_path = os.path.join(app.config['PROCESSED_FOLDER'], nesting_filename)
        cv2.imwrite(nesting_path, nesting_img)
        
        # ----------- RESPONSE -------------
        result = {
            'success': True,
            'nesting_image': f'/processed/{nesting_filename}',
            'statistics': {
                'utilization': round(utilization, 2),
                'fabric_usage': round(fabric_usage, 2),
                'efficiency': round(efficiency, 2),  # <-- NEW VALUE ADDED
                'waste_percentage': round(waste, 2),
                'total_parts_area': round(total_parts_area, 2),
                'used_fabric_area': round(used_fabric_area, 2),
                'fabric_width': fabric_width,
                'fabric_height': fabric_height
            },
            'placed_parts': placed_parts
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Optimization error: {str(e)}'}), 500


@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def serve_processed(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)