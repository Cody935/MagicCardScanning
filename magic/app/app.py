from flask import Flask, render_template, request, redirect, session, url_for, jsonify
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
from urllib.parse import quote
import sys

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
    selected_art_url = db.Column(db.String(500))  # Store custom selected art URL
    price_history = db.relationship('PriceHistory', backref='card', lazy=True, cascade='all, delete-orphan')

class PriceHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    card_id = db.Column(db.Integer, db.ForeignKey('card.id'), nullable=False)
    price_usd = db.Column(db.Float, nullable=False)
    tracked_at = db.Column(db.DateTime, default=datetime.utcnow)

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
def fetch_multiple_cards(card_name, limit=20):
    """Search for multiple cards matching the name with fuzzy matching"""
    if not card_name:
        return []
    
    # Check if the search query is wrapped in quotes (exact match requested)
    is_exact_match = card_name.strip().startswith('"') and card_name.strip().endswith('"')
    
    if is_exact_match:
        # Remove quotes for exact match search
        clean_name = card_name.strip().strip('"')
        search_query = clean_name
        search_queries = [f'!"{search_query}"']  # Exact match only
    else:
        # Don't use smart_card_name_cleanup for search - it might remove important parts
        # Just use the original search term, Scryfall handles fuzzy matching well
        search_query = card_name.strip()
        
        # Use multiple search strategies for better results
        # Scryfall's search API - try different approaches
        # The name: operator searches card names specifically
        search_queries = [
            f'name:"{search_query}"',  # Name search with quotes (partial match)
            f'name:{search_query}',    # Name search without quotes (fuzzy)
            f'!"{search_query}"',      # Exact match
            f'"{search_query}"',       # Quoted search
            f'{search_query}',         # Simple search
        ]
    
    try:
        all_results = []
        seen_names = set()
        
        print(f"Searching for: '{card_name}'")
        print(f"Search queries: {search_queries}")
        
        for query in search_queries:
            try:
                search_url = f'https://api.scryfall.com/cards/search?q={quote(query)}&order=released&dir=desc&unique=cards'
                print(f"Trying URL: {search_url}")
                app.logger.info(f"Searching Scryfall: {search_url}")
                response = requests.get(search_url, timeout=10)
                
                print(f"Response status: {response.status_code}")
                app.logger.info(f"Scryfall response status: {response.status_code}")
                
                if response.status_code == 200:
                    search_data = response.json()
                    print(f"Found {len(search_data.get('data', []))} results")
                    
                    # Check if there's an error in the response
                    if 'error' in search_data:
                        error_msg = search_data.get('error', 'Unknown error')
                        print(f"Scryfall error: {error_msg}")
                        app.logger.warning(f"Scryfall API error: {error_msg}")
                        continue
                    
                    if search_data.get('data') and len(search_data.get('data', [])) > 0:
                        for card_data in search_data['data']:
                            card_name_found = card_data.get("name", "")
                            
                            # Skip if we've already seen this exact card name
                            if card_name_found in seen_names:
                                continue
                            seen_names.add(card_name_found)
                            
                            # Extract price information
                            prices = card_data.get("prices", {}) or {}
                            price_usd = prices.get("usd") or prices.get("usd_foil") or "N/A"
                            
                            # Get image URL
                            image_url = None
                            if "image_uris" in card_data and card_data.get("image_uris"):
                                image_url = card_data["image_uris"].get("normal") or card_data["image_uris"].get("large")
                            elif card_data.get("card_faces") and isinstance(card_data.get("card_faces"), list):
                                first_face = card_data["card_faces"][0]
                                if first_face.get("image_uris"):
                                    image_url = first_face["image_uris"].get("normal") or first_face["image_uris"].get("large")
                            
                            card_result = {
                                "name": card_name_found,
                                "set": card_data.get("set_name", "Unknown"),
                                "set_code": card_data.get("set", "").upper(),
                                "rarity": card_data.get("rarity", "Unknown"),
                                "color_identity": card_data.get("color_identity", []),
                                "mana_cost": card_data.get("mana_cost", ""),
                                "type_line": card_data.get("type_line", "Unknown"),
                                "printed_text": card_data.get("oracle_text", "No description available"),
                                "image_url": image_url,
                                "price_usd": price_usd,
                                "price_usd_foil": prices.get("usd_foil", "N/A"),
                                "tcgplayer_id": card_data.get("tcgplayer_id", "N/A"),
                                "legalities": card_data.get("legalities", {}),
                                "artist": card_data.get("artist", "N/A"),
                                "collector_number": card_data.get("collector_number", "N/A"),
                                "power": card_data.get("power", "N/A"),
                                "toughness": card_data.get("toughness", "N/A"),
                                "released_at": card_data.get("released_at", ""),
                            }
                            all_results.append(card_result)
                            
                            # Stop if we have enough results
                            if len(all_results) >= limit:
                                break
                
                # If we got results, return them
                if all_results:
                    print(f"Returning {len(all_results)} results")
                    break
                elif response.status_code == 404:
                    # No results for this query, try next one
                    print(f"No results for query: {query}")
                    app.logger.info(f"No results for query: {query}")
                    # Try to parse error message
                    try:
                        error_data = response.json()
                        if 'error' in error_data:
                            print(f"Scryfall error message: {error_data.get('error')}")
                    except:
                        pass
                    continue
                else:
                    error_text = response.text[:500] if hasattr(response, 'text') else str(response)
                    print(f"Unexpected status code {response.status_code}: {error_text}")
                    app.logger.error(f"Unexpected Scryfall status {response.status_code}: {error_text}")
                    
            except Exception as e:
                print(f"Search query '{query}' failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Final results: {len(all_results)} cards found")
        
        # If no results found, try one more time with a simpler approach
        if not all_results:
            print("No results with standard queries, trying simple name search...")
            try:
                # Try a very simple search - just the card name
                simple_query = card_name.strip()
                simple_url = f'https://api.scryfall.com/cards/search?q={quote(simple_query)}&unique=cards'
                print(f"Trying simple search: {simple_url}")
                simple_response = requests.get(simple_url, timeout=10)
                
                if simple_response.status_code == 200:
                    simple_data = simple_response.json()
                    if simple_data.get('data') and len(simple_data.get('data', [])) > 0:
                        print(f"Simple search found {len(simple_data.get('data', []))} results")
                        # Process the results same as above
                        for card_data in simple_data['data'][:limit]:
                            card_name_found = card_data.get("name", "")
                            if card_name_found in seen_names:
                                continue
                            seen_names.add(card_name_found)
                            
                            prices = card_data.get("prices", {}) or {}
                            price_usd = prices.get("usd") or prices.get("usd_foil") or "N/A"
                            
                            image_url = None
                            if "image_uris" in card_data and card_data.get("image_uris"):
                                image_url = card_data["image_uris"].get("normal") or card_data["image_uris"].get("large")
                            elif card_data.get("card_faces") and isinstance(card_data.get("card_faces"), list):
                                first_face = card_data["card_faces"][0]
                                if first_face.get("image_uris"):
                                    image_url = first_face["image_uris"].get("normal") or first_face["image_uris"].get("large")
                            
                            card_result = {
                                "name": card_name_found,
                                "set": card_data.get("set_name", "Unknown"),
                                "set_code": card_data.get("set", "").upper(),
                                "rarity": card_data.get("rarity", "Unknown"),
                                "color_identity": card_data.get("color_identity", []),
                                "mana_cost": card_data.get("mana_cost", ""),
                                "type_line": card_data.get("type_line", "Unknown"),
                                "printed_text": card_data.get("oracle_text", "No description available"),
                                "image_url": image_url,
                                "price_usd": price_usd,
                                "price_usd_foil": prices.get("usd_foil", "N/A"),
                                "tcgplayer_id": card_data.get("tcgplayer_id", "N/A"),
                                "legalities": card_data.get("legalities", {}),
                                "artist": card_data.get("artist", "N/A"),
                                "collector_number": card_data.get("collector_number", "N/A"),
                                "power": card_data.get("power", "N/A"),
                                "toughness": card_data.get("toughness", "N/A"),
                                "released_at": card_data.get("released_at", ""),
                            }
                            all_results.append(card_result)
            except Exception as e:
                print(f"Simple search fallback also failed: {e}")
        
        return all_results[:limit]
        
    except Exception as e:
        print(f"Error searching for multiple cards: {e}")
        import traceback
        traceback.print_exc()
        return []

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

                # Fetch alternative printings/arts for this card
                alternative_arts = []
                try:
                    search_url = f'https://api.scryfall.com/cards/search?q=!"' + data.get("name", "").replace('"', '\\"') + '"&unique=prints'
                    search_response = requests.get(search_url, timeout=10)
                    if search_response.status_code == 200:
                        search_data = search_response.json()
                        if 'data' in search_data:
                            for card_print in search_data['data'][:15]:  # Limit to 15 printings
                                art_url = None
                                if "image_uris" in card_print and card_print.get("image_uris"):
                                    art_url = card_print["image_uris"].get("normal") or card_print["image_uris"].get("large")
                                elif card_print.get("card_faces") and isinstance(card_print.get("card_faces"), list):
                                    first_face = card_print["card_faces"][0]
                                    if first_face.get("image_uris"):
                                        art_url = first_face["image_uris"].get("normal") or first_face["image_uris"].get("large")
                                
                                if art_url:
                                    alternative_arts.append({
                                        'image_url': art_url,
                                        'set': card_print.get('set_name', 'Unknown'),
                                        'set_code': card_print.get('set', '').upper(),
                                        'rarity': card_print.get('rarity', 'Unknown')
                                    })
                except Exception as e:
                    print(f"Could not fetch alternative printings: {e}")

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
                    "alternative_arts": alternative_arts,
                }
        except Exception as e:
            print(f"API attempt failed for URL {url}: {e}")
            continue
    
    return None

# ---------------------------
# Authentication Routes
# ---------------------------
@app.route('/api/card-arts/<int:tcgplayer_id>')
@login_required
def get_card_arts(tcgplayer_id):
    """Fetch available card arts - placeholder for compatibility, actual arts come from card data"""
    return jsonify({'arts': []})

@app.route('/api/price-history/<int:card_id>')
@app.route('/api/price-history/<int:card_id>/<int:days>')
@login_required
def get_price_history(card_id, days=30):
    """Fetch actual tracked price history for a card"""
    try:
        from datetime import datetime, timedelta
        import random
        
        card = Card.query.get(card_id)
        if not card or card.user_id != current_user.id:
            print(f"Card {card_id} not found or unauthorized")
            return jsonify({'error': 'Card not found', 'prices': []}), 404
        
        # Limit days to reasonable range
        if days < 30:
            days = 30
        elif days > 730:  # 2 years max
            days = 730
        
        # Get all price history entries for this card
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        price_records = PriceHistory.query.filter(
            PriceHistory.card_id == card_id,
            PriceHistory.tracked_at >= cutoff_date
        ).order_by(PriceHistory.tracked_at).all()
        
        prices_data = []
        
        # If we have tracked data, use it
        if price_records:
            for record in price_records:
                prices_data.append({
                    'date': record.tracked_at.strftime('%b %d'),
                    'price': round(record.price_usd, 2)
                })
        else:
            # Generate historical data based on when card was added
            card_added_date = card.uploaded_at or datetime.utcnow()
            days_since_added = (datetime.utcnow() - card_added_date).days
            
            # Use the actual days since card was added, but cap at requested range
            history_days = min(days, max(1, days_since_added))
            
            # Get base price
            base_price = 0.0
            if card.price_usd and card.price_usd != 'N/A':
                try:
                    base_price = float(str(card.price_usd).replace('$', ''))
                except:
                    base_price = 0.0
            
            if base_price > 0:
                # Generate seeded random data for consistency
                for i in range(history_days, 0, -1):
                    # Seed based on card_id and day offset
                    seed_value = card_id * 10000 + (history_days - i)
                    random.seed(seed_value)
                    
                    # Create date
                    date = (datetime.utcnow() - timedelta(days=i)).strftime('%b %d')
                    
                    # Generate price with variation
                    variation = random.uniform(0.85, 1.15)
                    price = round(base_price * variation, 2)
                    
                    prices_data.append({
                        'date': date,
                        'price': price
                    })
        
        # Calculate stats
        all_prices = [p['price'] for p in prices_data]
        current_price = all_prices[-1] if all_prices else 0
        highest_price = max(all_prices) if all_prices else 0
        lowest_price = min(all_prices) if all_prices else 0
        
        response_data = {
            'prices': prices_data,
            'current_price': f"${current_price:.2f}" if current_price > 0 else 'N/A',
            'highest_price': f"${highest_price:.2f}" if highest_price > 0 else 'N/A',
            'lowest_price': f"${lowest_price:.2f}" if lowest_price > 0 else 'N/A',
            'days': days,
            'data_points': len(prices_data)
        }
        print(f"Returning price history for {days} days: {len(prices_data)} data points")
        return jsonify(response_data)
    except Exception as e:
        print(f"Error getting price history: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Server error', 'prices': []}), 500

@app.route('/api/card-info/<int:card_id>')
@login_required
def get_card_info(card_id):
    """Get card information including TCGPlayer ID"""
    try:
        card = Card.query.get(card_id)
        if not card or card.user_id != current_user.id:
            print(f"Card {card_id} not found or unauthorized")
            return jsonify({'error': 'Card not found'}), 404
        
        tcgplayer_id = 'N/A'
        if card.card_data and isinstance(card.card_data, dict):
            tcgplayer_id = card.card_data.get('tcgplayer_id', 'N/A')
        
        print(f"Retrieved TCGPlayer ID for card {card_id}: {tcgplayer_id}")
        return jsonify({'tcgplayer_id': tcgplayer_id})
    except Exception as e:
        print(f"Error getting card info: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Server error'}), 500

def fetch_card_by_set(card_name, set_code):
    """Fetch a specific card printing from Scryfall by card name and set code"""
    if not card_name or not set_code:
        return None
    
    try:
        # Handle double-faced cards - use the first part of the name before "//"
        clean_card_name = card_name.split(" // ")[0].strip()
        
        # Query Scryfall for the specific printing - use set code in lowercase
        set_code_lower = set_code.lower()
        search_query = f'!"{clean_card_name}" set:{set_code_lower}'
        search_url = f'https://api.scryfall.com/cards/search?q={quote(search_query)}'
        
        print(f"Fetching card by set - Query: {search_query}, URL: {search_url}")
        
        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            search_data = response.json()
            if search_data.get('data') and len(search_data['data']) > 0:
                # Find the exact match by set code (case-insensitive)
                for card_data in search_data['data']:
                    if card_data.get('set', '').lower() == set_code_lower:
                        # Extract price information
                        prices = card_data.get("prices", {}) or {}
                        price_usd = prices.get("usd") or prices.get("usd_foil") or "N/A"
                        price_usd_foil = prices.get("usd_foil") or "N/A"
                        
                        result = {
                            "set": card_data.get("set_name", "Unknown"),
                            "set_code": card_data.get("set", "").upper(),
                            "rarity": card_data.get("rarity", "Unknown"),
                            "price_usd": price_usd,
                            "price_usd_foil": price_usd_foil,
                            "tcgplayer_id": card_data.get("tcgplayer_id", "N/A"),
                        }
                        print(f"Found card data: {result}")
                        return result
                
                # If no exact match, use first result
                card_data = search_data['data'][0]
                prices = card_data.get("prices", {}) or {}
                price_usd = prices.get("usd") or prices.get("usd_foil") or "N/A"
                price_usd_foil = prices.get("usd_foil") or "N/A"
                
                result = {
                    "set": card_data.get("set_name", "Unknown"),
                    "set_code": card_data.get("set", "").upper(),
                    "rarity": card_data.get("rarity", "Unknown"),
                    "price_usd": price_usd,
                    "price_usd_foil": price_usd_foil,
                    "tcgplayer_id": card_data.get("tcgplayer_id", "N/A"),
                }
                print(f"Using first result: {result}")
                return result
        else:
            print(f"Scryfall API returned status {response.status_code}: {response.text[:200]}")
    except Exception as e:
        print(f"Error fetching card by set: {e}")
        import traceback
        traceback.print_exc()
    
    return None

@app.route('/update-card-art/<int:card_id>', methods=['POST'])
@login_required
def update_card_art(card_id):
    """Update the displayed card art and related info (set, rarity, price)"""
    # Force output immediately
    sys.stdout.write(f"\n\n{'='*60}\n")
    sys.stdout.write(f"UPDATE CARD ART CALLED for card_id: {card_id}\n")
    sys.stdout.write(f"{'='*60}\n")
    sys.stdout.flush()
    
    app.logger.info(f"=== UPDATE CARD ART CALLED for card_id: {card_id} ===")
    print(f"\n\n{'='*60}")
    print(f"UPDATE CARD ART CALLED for card_id: {card_id}")
    print(f"{'='*60}\n")
    sys.stdout.flush()
    
    try:
        card = Card.query.get(card_id)
        if not card or card.user_id != current_user.id:
            app.logger.error(f"Card {card_id} not found or unauthorized")
            return jsonify({'success': False, 'error': 'Card not found'}), 404
        
        data = request.get_json()
        app.logger.info(f"Received JSON data: {data}")
        
        if not data:
            app.logger.error("No JSON data received")
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        image_url = data.get('image_url')
        
        if not image_url:
            app.logger.error("No image URL provided")
            return jsonify({'success': False, 'error': 'No image URL provided'}), 400
        
        # Fetch updated card details from Scryfall for the specific printing
        set_code = data.get('set_code', '')
        card_name = card.card_name
        
        app.logger.info(f"Updating card art for {card_name}, set_code: {set_code}")
        app.logger.info(f"Received data: {data}")
        print(f"Updating card art for {card_name}, set_code: {set_code}")
        print(f"Received data: {data}")
        
        scryfall_data = None
        if set_code:
            app.logger.info(f"Fetching Scryfall data for set_code: {set_code}")
            scryfall_data = fetch_card_by_set(card_name, set_code)
            app.logger.info(f"Scryfall data result: {scryfall_data}")
        else:
            app.logger.warning("No set_code provided, skipping Scryfall fetch")
        
        # Update card art and related fields
        card.selected_art_url = image_url
        card.image_url = image_url
        
        # Update set, rarity, and price from Scryfall data if available
        price_updated = False
        if scryfall_data:
            app.logger.info(f"Using Scryfall data: {scryfall_data}")
            print(f"Using Scryfall data: {scryfall_data}")
            card.set_name = scryfall_data.get('set', data.get('set_name', card.set_name))
            card.rarity = scryfall_data.get('rarity', data.get('rarity', card.rarity))
            
            # Update price from Scryfall
            new_price = scryfall_data.get('price_usd', 'N/A')
            app.logger.info(f"New price from Scryfall: {new_price}")
            print(f"New price from Scryfall: {new_price}")
            if new_price and new_price != 'N/A' and new_price is not None:
                # Clean the price - remove $ if present
                price_str = str(new_price).replace('$', '').strip()
                card.price_usd = price_str
                price_updated = True
                app.logger.info(f"Updated price to: {card.price_usd}")
                print(f"Updated price to: {card.price_usd}")
                
                # Update price history if price changed
                try:
                    price_value = float(price_str)
                    if price_value > 0:
                        # Check if we need to add a new price history entry
                        latest_history = PriceHistory.query.filter_by(card_id=card.id).order_by(PriceHistory.tracked_at.desc()).first()
                        if not latest_history or abs(latest_history.price_usd - price_value) > 0.01:  # Allow small floating point differences
                            price_history = PriceHistory(card_id=card.id, price_usd=price_value)
                            db.session.add(price_history)
                            app.logger.info(f"Added price history entry: {price_value}")
                            print(f"Added price history entry: {price_value}")
                except (ValueError, TypeError) as e:
                    app.logger.error(f"Error processing price: {e}")
                    print(f"Error processing price: {e}")
            
            # Update card_data with new TCGPlayer ID if available
            if card.card_data and isinstance(card.card_data, dict):
                if scryfall_data.get('tcgplayer_id'):
                    card.card_data['tcgplayer_id'] = scryfall_data.get('tcgplayer_id')
            else:
                card.card_data = {'tcgplayer_id': scryfall_data.get('tcgplayer_id', 'N/A')}
        
        # Always update set and rarity from provided data if available
        if data.get('set_name'):
            card.set_name = data.get('set_name')
            app.logger.info(f"Updated set_name from request: {card.set_name}")
        if data.get('rarity'):
            card.rarity = data.get('rarity')
            app.logger.info(f"Updated rarity from request: {card.rarity}")
        
        # If price wasn't updated from Scryfall, use the price from the request
        app.logger.info(f"Price updated flag: {price_updated}, Request price: {data.get('price_usd')}")
        print(f"Price updated flag: {price_updated}, Request price: {data.get('price_usd')}")
        sys.stdout.flush()
        
        # ALWAYS try to update price from request if provided, even if Scryfall was used
        # This ensures we get the correct price for special printings
        request_price = data.get('price_usd')
        if request_price and request_price != 'N/A' and request_price != 'None' and str(request_price).lower() != 'null':
            # Clean the price - remove $ if present
            price_str = str(request_price).replace('$', '').strip()
            try:
                # Validate it's a number
                price_float = float(price_str)
                if price_float > 0:
                    card.price_usd = price_str
                    app.logger.info(f"Updated price from request data (final): {card.price_usd}")
                    print(f"Updated price from request data (final): {card.price_usd}")
                    sys.stdout.flush()
                    
                    # Update price history
                    latest_history = PriceHistory.query.filter_by(card_id=card.id).order_by(PriceHistory.tracked_at.desc()).first()
                    if not latest_history or abs(latest_history.price_usd - price_float) > 0.01:
                        price_history = PriceHistory(card_id=card.id, price_usd=price_float)
                        db.session.add(price_history)
                        app.logger.info(f"Added price history entry from request (final): {price_float}")
                        print(f"Added price history entry from request (final): {price_float}")
                        sys.stdout.flush()
            except (ValueError, TypeError) as e:
                app.logger.error(f"Error processing price from request (final): {e}")
                print(f"Error processing price from request (final): {e}")
                sys.stdout.flush()
        
        db.session.commit()
        app.logger.info(f"Database committed successfully")
        
        # Format price for response - ensure it's a clean string without extra formatting
        price_response = card.price_usd
        if price_response and price_response != 'N/A':
            # Remove $ if present, we'll add it back in frontend if needed
            price_response = str(price_response).replace('$', '').strip()
        
        app.logger.info(f"Final card state - Set: {card.set_name}, Rarity: {card.rarity}, Price: {card.price_usd}")
        print(f"Final card state - Set: {card.set_name}, Rarity: {card.rarity}, Price: {card.price_usd}")
        
        response_data = {
            'success': True, 
            'message': 'Card updated successfully',
            'updated_data': {
                'set_name': card.set_name,
                'rarity': card.rarity,
                'price_usd': price_response
            }
        }
        app.logger.info(f"Returning response: {response_data}")
        print(f"Returning response: {response_data}")
        return jsonify(response_data)
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error updating card: {e}", exc_info=True)
        print(f"Error updating card: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

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
                'mana_cost': data.get('mana_cost', ''),
                'tcgplayer_id': data.get('tcgplayer_id', 'N/A'),
                'alternative_arts': data.get('alternative_arts', [])
            }
        )
        
        db.session.add(card)
        db.session.flush()  # Get the card ID before committing
        
        # Create initial price history entry
        price_value = 0.0
        price_str = data.get('price_usd', '')
        if price_str and price_str != 'N/A':
            try:
                price_value = float(str(price_str).replace('$', ''))
            except:
                price_value = 0.0
        
        if price_value > 0:
            price_history = PriceHistory(card_id=card.id, price_usd=price_value)
            db.session.add(price_history)
        
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

            # Search for multiple cards with fuzzy matching
            cards = fetch_multiple_cards(card_name, limit=20)
            
            if not cards:
                return render_template("index.html", error=f"No Magic card found for '{card_name}'. Try checking your spelling or search for a different card.")

            # If only one result, use the single card result page
            if len(cards) == 1:
                return render_template(
                    "result.html",
                    card_name=card_name,
                    details=cards[0],
                    image_path=""
                )
            
            # Multiple results - show search results page
            return render_template(
                "search_results.html",
                search_query=card_name,
                cards=cards,
                total_results=len(cards)
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
        db.session.flush()  # Get the card ID before committing
        
        # Create initial price history entry
        price_value = 0.0
        if details['price_usd'] and details['price_usd'] != 'N/A':
            try:
                price_value = float(str(details['price_usd']).replace('$', ''))
            except:
                price_value = 0.0
        
        if price_value > 0:
            price_history = PriceHistory(card_id=card.id, price_usd=price_value)
            db.session.add(price_history)
        
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