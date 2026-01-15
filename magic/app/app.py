from flask import Flask, render_template, request, redirect, session, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import pytesseract
import requests
import os
import numpy as np
from PIL import Image
import io
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cards.db'
app.config['SECRET_KEY'] = 'your-secret-key-change-this'  # Change this to a random secret key
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ---------------------------
# Database Models
# ---------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    cards = db.relationship('Card', backref='user', lazy=True, cascade='all, delete-orphan')

class Card(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    card_name = db.Column(db.String(150), nullable=False)
    set_name = db.Column(db.String(150))
    rarity = db.Column(db.String(50))
    price_usd = db.Column(db.String(50))
    image_url = db.Column(db.String(500))
    uploaded_image = db.Column(db.String(200))
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    card_data = db.Column(db.JSON)  # Store full card details as JSON

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Path to tesseract executable - Update this path for your environment!
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------------------
# Improved OCR Functions
# ---------------------------
def preprocess_for_ocr(image):
    """Enhanced preprocessing for better text detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Multiple preprocessing techniques
    processed_images = []
    
    # 1. Simple threshold
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(thresh1)
    
    # 2. Adaptive threshold
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    processed_images.append(thresh2)
    
    # 3. Denoising + threshold
    denoised = cv2.fastNlMeansDenoising(gray)
    _, thresh3 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(thresh3)
    
    # 4. Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, thresh4 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(thresh4)
    
    return processed_images

def extract_text_with_multiple_methods(image):
    """Try multiple OCR methods and return the best result"""
    best_text = ""
    best_confidence = 0
    
    # Different OCR configurations to try
    configs = [
        r'--oem 3 --psm 7',           # Single text line
        r'--oem 3 --psm 8',           # Single word
        r'--oem 3 --psm 6',           # Uniform block of text
        r'--oem 3 --psm 13',          # Raw line
    ]
    
    processed_images = preprocess_for_ocr(image)
    
    for i, processed_img in enumerate(processed_images):
        for config in configs:
            try:
                # Get detailed OCR data
                ocr_data = pytesseract.image_to_data(processed_img, config=config, output_type=pytesseract.Output.DICT)
                
                # Calculate average confidence for good detections
                confidences = [int(conf) for i, conf in enumerate(ocr_data['conf']) 
                             if int(conf) > 0 and ocr_data['text'][i].strip()]
                
                if confidences:
                    avg_confidence = np.mean(confidences)
                    text = ' '.join([ocr_data['text'][i] for i in range(len(ocr_data['text'])) 
                                   if int(ocr_data['conf'][i]) > 0 and ocr_data['text'][i].strip()])
                    
                    if avg_confidence > best_confidence and text.strip():
                        best_confidence = avg_confidence
                        best_text = text.strip()
                        print(f"Method {i+1}, Config {config}: {text} (conf: {avg_confidence:.1f})")
                        
            except Exception as e:
                # print(f"OCR failed for config {config} on method {i+1}: {e}")
                continue
    
    return best_text, best_confidence

def extract_card_name_direct(image_path):
    """Direct card name extraction without complex detection"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        height, width = img.shape[:2]
        
        # Try different regions of the image
        # Focus on the top area where the name is
        regions_to_try = [
            (0, int(height * 0.25), 0, width),           # Top 25%
            (0, int(height * 0.20), 0, width),           # Top 20%
            (0, int(height * 0.30), 0, width),           # Top 30%
            (int(height * 0.05), int(height * 0.25), 0, width),  # Slightly lower
        ]
        
        best_result = ""
        best_confidence = 0
        
        for y1, y2, x1, x2 in regions_to_try:
            region = img[y1:y2, x1:x2]
            text, confidence = extract_text_with_multiple_methods(region)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_result = text
        
        print(f"Best OCR result: '{best_result}' with confidence {best_confidence:.1f}")
        
        if best_result and best_confidence > 30:  # Minimum confidence threshold
            # Clean up the result
            lines = [line.strip() for line in best_result.split('\n') if line.strip()]
            if lines:
                # Return the first substantial line (usually the card name)
                for line in lines:
                    if len(line) > 2:  # Filter out very short lines
                        return line
        return None
        
    except Exception as e:
        print(f"Error in direct extraction: {e}")
        return None

def smart_card_name_cleanup(card_name):
    """Clean up the detected card name"""
    if not card_name:
        return None
    
    # Remove common OCR artifacts
    artifacts = ['@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '=', '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '<', '>', '?', '!']
    for artifact in artifacts:
        card_name = card_name.replace(artifact, '')
    
    # Remove extra whitespace
    card_name = ' '.join(card_name.split())
    
    # Common Magic card prefixes/suffixes to handle
    magic_terms = [
        'planeswalker', 'instant', 'sorcery', 'creature', 'enchantment', 
        'artifact', 'land', 'basic land', 'legendary', 'token', 'the'
    ]
    
    # Split and take only substantial words, but be careful not to remove the whole name
    words = card_name.split()
    filtered_words = []
    
    for word in words:
        # Keep words that are likely part of a card name
        if (len(word) > 1 and 
            not word.lower() in ['and', 'or', 'of', 'a', 'an'] and
            not any(term in word.lower() for term in magic_terms)):
            filtered_words.append(word)
    
    # Fallback to the original cleanup if aggressive filtering removes too much
    if len(filtered_words) < 1 and len(words) >= 1:
        return card_name
        
    if filtered_words:
        return ' '.join(filtered_words)
    
    return card_name

# ---------------------------
# Fetch data from Scryfall
# ---------------------------
def fetch_card_details(card_name):
    """Fetch card details from Scryfall with multiple attempts"""
    if not card_name:
        return None
    
    # Clean the card name first
    clean_name = smart_card_name_cleanup(card_name)
    print(f"Searching for: '{clean_name}'")
    
    attempts = [
        f"https://api.scryfall.com/cards/named?exact={clean_name}",
        f"https://api.scryfall.com/cards/named?fuzzy={clean_name}",
    ]
    
    # If the cleaned name failed to return a result, try the original detected name
    if clean_name != card_name:
        attempts.append(f"https://api.scryfall.com/cards/named?exact={card_name}")
        attempts.append(f"https://api.scryfall.com/cards/named?fuzzy={card_name}")
        
    
    for url in attempts:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Determine TCGPlayer ID and price for the specific printing
                tcgplayer_id = data.get("tcgplayer_id", "N/A")
                prices = data.get("prices", {}) or {}
                price_usd = prices.get("usd", "N/A")
                price_usd_foil = prices.get("usd_foil", "N/A")
                
                # Fetch rulings for Legality/Banned status (optional, but good for detail)
                # Not strictly necessary for the main task, but possible enhancement
                
                # Scryfall provides legality details directly
                legalities = data.get("legalities", {})

                # Robust image selection: normal image_uris, or first face image if double-faced
                image_url = None
                if "image_uris" in data and data.get("image_uris"):
                    image_url = data["image_uris"].get("normal") or data["image_uris"].get("large")
                elif data.get("card_faces") and isinstance(data.get("card_faces"), list):
                    first_face = data["card_faces"][0]
                    if first_face.get("image_uris"):
                        image_url = first_face["image_uris"].get("normal") or first_face["image_uris"].get("large")

                return {
                    "name": data.get("name", "Unknown"),
                    "set": data.get("set_name", "Unknown"),
                    "set_code": data.get("set", "").upper(),
                    "rarity": data.get("rarity", "Unknown"),
                    "color_identity": data.get("color_identity", []),
                    "mana_cost": data.get("mana_cost", ""),
                    "type_line": data.get("type_line", "Unknown"),
                    "printed_text": data.get("oracle_text", "No description available"),
                    "image_url": image_url,
                    "price_usd": price_usd,
                    "price_usd_foil": price_usd_foil,
                    "tcgplayer_id": tcgplayer_id,
                    "legalities": legalities,
                    "artist": data.get("artist", "N/A"),
                    "collector_number": data.get("collector_number", "N/A"),
                    "power": data.get("power", "N/A"),
                    "toughness": data.get("toughness", "N/A"),
                }
        except Exception as e:
            print(f"API attempt failed for URL {url}: {e}")
            continue
    
    return None

# ---------------------------
# Authentication Routes
# ---------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not password:
            return render_template('register.html', error='Username and password are required.')
        
        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match.')
        
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error='Username already exists.')
        
        user = User(username=username, password=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('upload_card'))
        else:
            return render_template('login.html', error='Invalid username or password.')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/collection')
@login_required
def collection():
    """View user's card collection"""
    cards = Card.query.filter_by(user_id=current_user.id).all()
    return render_template('collection.html', cards=cards)

@app.route('/delete-card/<int:card_id>', methods=['POST'])
@login_required
def delete_card(card_id):
    """Delete a card from user's collection"""
    card = Card.query.get(card_id)
    if card and card.user_id == current_user.id:
        db.session.delete(card)
        db.session.commit()
    return redirect(url_for('collection'))

@app.route('/add-card', methods=['POST'])
@login_required
def add_card():
    """Add a card to user's collection from result page or search"""
    from flask import jsonify
    
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or not data.get('card_name'):
            return jsonify({'success': False, 'error': 'Invalid card data'}), 400
        
        # Check if card already exists in user's collection
        existing_card = Card.query.filter_by(
            user_id=current_user.id,
            card_name=data.get('card_name'),
            set_name=data.get('set_name', '')
        ).first()
        
        if existing_card:
            return jsonify({'success': False, 'error': 'Card already in collection'}), 409
        
        # Create new card entry
        card = Card(
            user_id=current_user.id,
            card_name=data.get('card_name'),
            set_name=data.get('set_name', ''),
            rarity=data.get('rarity', ''),
            price_usd=data.get('price_usd', ''),
            image_url=data.get('image_url', ''),
            card_data={
                'collector_number': data.get('collector_number', ''),
                'type_line': data.get('type_line', ''),
                'mana_cost': data.get('mana_cost', '')
            }
        )
        
        db.session.add(card)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Card added to collection'}), 201
    
    except Exception as e:
        db.session.rollback()
        print(f"Error adding card: {str(e)}")
        return jsonify({'success': False, 'error': 'Server error'}), 500

# ---------------------------
# Flask Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
@login_required
def upload_card():
    if request.method == "POST":
        # If user submitted a card name via the search box
        form_name = request.form.get("card_name")
        if form_name:
            card_name = form_name.strip()
            if not card_name:
                return render_template("index.html", error="Please enter a card name.")

            details = fetch_card_details(card_name)
            if not details:
                return render_template("index.html", error=f"No Magic card found for '{card_name}'.")

            return render_template(
                "result.html",
                card_name=card_name,
                details=details,
                image_path=""
            )

        # Otherwise handle file upload
        file = request.files.get("card_image")
        if not file or file.filename == "":
            return render_template("index.html", error="Please select a file.")
        
        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            return render_template("index.html", error="Please upload an image file (PNG, JPG, JPEG, BMP, WEBP).")
        
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        print(f"File saved to: {filepath}")

        # Try direct OCR extraction
        card_name = extract_card_name_direct(filepath)
        print(f"Final extracted name: {card_name}")
        
        if not card_name:
            return render_template("index.html", error="Could not detect card name. Try a clearer image with good contrast.")
        
        details = fetch_card_details(card_name)
            
        if not details:
            return render_template("index.html", error=f"No Magic card found for '{card_name}'. Try a different image or check the card name.")
        
        # Save card to database
        card = Card(
            user_id=current_user.id,
            card_name=details['name'],
            set_name=details['set'],
            rarity=details['rarity'],
            price_usd=str(details['price_usd']),
            image_url=details['image_url'],
            uploaded_image=file.filename,
            card_data=details
        )
        db.session.add(card)
        db.session.commit()
        
        return render_template(
            "result.html",
            card_name=card_name,
            details=details,
            image_path=f"static/uploads/{file.filename}"
        )
    
    return render_template("index.html")

@app.route("/debug", methods=["GET", "POST"])
@login_required
def debug_upload():
    """Debug route to see what OCR is detecting"""
    if request.method == "POST":
        file = request.files.get("card_image")
        if not file or file.filename == "":
            return render_template("debug.html", error="Please select a file.")
        
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        
        # Read and process image
        img = cv2.imread(filepath)
        height, width = img.shape[:2]
        
        # Try different regions
        regions = [
            ("Top 20%", 0, int(height * 0.20), 0, width),
            ("Top 25%", 0, int(height * 0.25), 0, width),
            ("Top 30%", 0, int(height * 0.30), 0, width),
            ("Middle", int(height * 0.35), int(height * 0.65), 0, width),
        ]
        
        results = []
        for region_name, y1, y2, x1, x2 in regions:
            region = img[y1:y2, x1:x2]
            text, confidence = extract_text_with_multiple_methods(region)
            results.append({
                'region': region_name,
                'text': text,
                'confidence': f"{confidence:.1f}%"
            })
        
        return render_template(
            "debug.html",
            results=results,
            image_path=f"static/uploads/{file.filename}"
        )
    
    return render_template("debug.html")

if __name__ == "__main__":
    # Ensure you have a 'static' directory and a 'static/uploads' directory
    # Also ensure you create a 'static/css' folder for the new style.css file
    os.makedirs(os.path.join(os.path.dirname(__file__), "static/css"), exist_ok=True)
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    app.run(debug=True, host='0.0.0.0', port=5000)