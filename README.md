# 🃏 Magic Card Scanning

A comprehensive web application for scanning, cataloging, and managing your Magic: The Gathering card collection with real-time price tracking and OCR-powered card detection.

**Live Site:** [https://magiccardscanning.onrender.com/login](https://magiccardscanning.onrender.com/login)

---

## ✨ Features

### 📸 Smart OCR Card Detection
- Upload card images and automatically detect card names using Tesseract OCR
- Intelligent preprocessing with CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Multi-region fallback detection for reliable name extraction
- Balanced speed and accuracy optimized for Magic card fonts

### 🎨 Card Collection Management
- Build and organize your personal Magic card collection
- **2-Column grid layout** for efficient browsing
- View detailed card information (set, rarity, type, price)
- Delete cards from collection with one click
- Responsive design works seamlessly on mobile and desktop

### 💳 Card Art Selection
- **Click on any card image** to browse available art variants
- Select from multiple printings and art styles
- Custom art selection persists in your collection
- Browse up to 10 different art variations per card

### 💰 Price Tracking
- Real-time card pricing from Scryfall API
- Historical price data tracked over time
- **📈 Price Graph button** - View TCGPlayer market trends
- Automatic price updates for your collection

### 👤 User Authentication
- Secure registration and login system
- Personal collection linked to user accounts
- Session management with Flask-Login
- Password-protected access to features

---

## 🛠️ Tech Stack

### Backend
- **Flask** - Web framework
- **SQLAlchemy** - ORM for database management
- **Tesseract OCR** - Optical character recognition
- **OpenCV** - Image preprocessing and analysis
- **Scryfall API** - Magic card database integration
- **Pytesseract** - Python wrapper for Tesseract

### Frontend
- **HTML5 / CSS3** - Responsive interface
- **Bootstrap** - UI components
- **JavaScript** - Interactive features
- **Jinja2** - Template engine

### Database
- **SQLite** - Local database storage
- **SQLAlchemy ORM** - Object-relational mapping

### Deployment
- **Render.com** - Cloud hosting platform

---

## 📋 Requirements

### System Dependencies
- **Tesseract OCR** (Windows path: `C:\Program Files\Tesseract-OCR\tesseract.exe`)
- **Python 3.8+**
- **OpenCV** (cv2)
- **Pytesseract**

### Python Dependencies
```
Flask==3.1.2
Flask-SQLAlchemy==3.1.1
Flask-Login==0.6.3
pytesseract
opencv-python
numpy
requests
Pillow
```

See [requirements.txt](requirements.txt) for complete dependency list.

---

## 🚀 Getting Started

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MagicCardScanning
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR** (Windows)
   - Download installer: [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki)
   - Install to: `C:\Program Files\Tesseract-OCR\`
   - This path is already configured in `app.py`

5. **Set up database**
   ```bash
   cd magic/app
   python
   >>> from app import app, db
   >>> with app.app_context():
   ...     db.create_all()
   >>> exit()
   ```

### Running Locally

```bash
cd magic/app
python app.py
```

Visit: `http://localhost:5000`

---

## 📱 How to Use

### 1. **Create an Account**
   - Click "Register" on the login page
   - Enter email and password
   - Your account is ready to use!

### 2. **Upload a Card Image**
   - Go to the upload section
   - Take or select a photo of a Magic card
   - The OCR will automatically detect the card name
   - Card details are fetched from Scryfall API
   - Card is added to your collection

### 3. **Browse Your Collection**
   - View all cards in your collection
   - Cards displayed in 2-column grid
   - See card details: set, rarity, type, price
   - Price is updated automatically

### 4. **Customize Card Art**
   - Click on any card image
   - Browse available art variants
   - Select your preferred art
   - Selection saves automatically

### 5. **Track Prices**
   - Click **"📈 Price Graph"** button on any card
   - Opens TCGPlayer market data in new tab
   - See price trends and historical data

### 6. **Delete Cards**
   - Click **"Delete"** button on any card
   - Card is removed from collection
   - Changes save immediately

---

## 🏗️ Project Structure

```
MagicCardScanning/
├── README.md                 # Original README
├── FEATURES_ADDED.md         # New features documentation
├── requirements.txt          # Python dependencies
├── magic/
│   └── app/
│       ├── app.py           # Main Flask application
│       ├── __init__.py       # Package initialization
│       ├── requirements.txt  # App-specific dependencies
│       ├── static/
│       │   ├── css/
│       │   │   ├── style.css          # Main styles
│       │   │   └── result_style.css   # Result page styles
│       │   └── uploads/      # Uploaded card images
│       └── templates/
│           ├── index.html            # Home page
│           ├── login.html            # Login page
│           ├── register.html         # Registration page
│           ├── collection.html       # Collection view
│           ├── result.html           # OCR result page
│           ├── search_results.html   # Search results
│           ├── debug.html            # Debug/testing page
│           └── index_old.html        # Legacy template
├── instance/                # Flask instance folder
└── .git/                   # Git repository
```

---

## 🧠 OCR System

### How Card Detection Works

1. **Image Upload** - User uploads card image
2. **Preprocessing** - CLAHE enhancement + OTSU threshold
3. **Region Detection** - Primary region (top 25%), with secondary fallbacks
4. **Multi-Config OCR** - Tesseract tries 3 PSM modes:
   - PSM 6: Uniform text blocks
   - PSM 7: Single text lines
   - PSM 8: Single words
5. **Confidence Thresholding**:
   - ≥60% confidence: Accept immediately (primary region)
   - <50% confidence: Try secondary regions
   - ≥30% confidence: Final minimum threshold
6. **Text Cleanup** - Remove OCR artifacts and Magic card type terms
7. **API Lookup** - Query Scryfall with detected name

### Optimization Features
- **Single CLAHE preprocessing** - Fast and effective
- **Early exit at 70% confidence** - Balances speed/accuracy
- **Image resizing** - Scales down images >800px wide
- **Region fallbacks** - Finds card names reliably
- **Confidence-based acceptance** - Smart thresholds

---

## 🔗 API Integration

### Scryfall API
- Fuzzy name matching for detected cards
- Card details: type, rarity, set, image URL
- Price information from multiple sources
- Fallback strategies for unmatched cards

### TCGPlayer Integration
- Real-time price data
- Historical price tracking
- Direct links to product pages
- Market trend visualization

---

## 🗄️ Database Schema

### User Model
```python
- id: Integer (Primary Key)
- username: String (Unique)
- email: String (Unique)
- password_hash: String
- cards: Relationship (One-to-Many)
```

### Card Model
```python
- id: Integer (Primary Key)
- user_id: Integer (Foreign Key)
- scryfall_id: String
- name: String
- set: String
- rarity: String
- image_url: String
- selected_art_url: String (Custom art selection)
- price_usd: Float
- added_at: DateTime
- price_history: Relationship (One-to-Many)
```

### PriceHistory Model
```python
- id: Integer (Primary Key)
- card_id: Integer (Foreign Key)
- price_usd: Float
- tracked_at: DateTime
```

---

## 📊 Recent Improvements

### OCR Optimization
- Replaced complex multi-method preprocessing with proven CLAHE approach
- Reduced OCR configs from 8 to 3 most effective modes
- Implemented confidence-based early exits
- Balanced speed and accuracy for reliable card detection

### Collection Features
- 2-column grid layout for better card browsing
- Direct TCGPlayer price graph integration
- Custom card art selection and persistence
- Responsive design for all screen sizes

---

## 🔍 Debug Mode

For OCR testing and debugging:
1. Navigate to `/debug_upload` (if running locally)
2. Upload a card image
3. View OCR results for different regions
4. See confidence scores for each detection method

---

## 🚨 Troubleshooting

### Tesseract Not Found
- **Error**: `pytesseract.pytesseract.TesseractNotFoundError`
- **Solution**: Install Tesseract or update path in `app.py` line 64

### OCR Results Inaccurate
- Try different card angles or lighting
- Ensure card image is clear and not blurry
- Use the debug mode to test regions

### Database Errors
- Delete `instance/` folder to reset database
- Run `db.create_all()` again

### Image Upload Issues
- Check `magic/app/static/uploads/` permissions
- Ensure folder exists and is writable

---

## 📝 Environment Variables

Create a `.env` file (optional for advanced config):
```
FLASK_ENV=development
FLASK_DEBUG=True
DATABASE_URL=sqlite:///instance/app.db
UPLOAD_FOLDER=magic/app/static/uploads
```

---

## 🤝 Contributing

To contribute improvements:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👥 Author

Created as a personal Magic: The Gathering collection management system.

---

## 📞 Support

For issues or questions:
- Check the debug page for OCR testing
- Review logs in console output
- Verify all dependencies are installed
- Ensure Tesseract OCR is properly installed

---

## 🎯 Future Enhancements

- [ ] Batch card upload
- [ ] Card value analytics dashboard
- [ ] Export collection to CSV/PDF
- [ ] Mobile app integration
- [ ] Advanced filtering and sorting
- [ ] Deck building integration
- [ ] Wishlist features
- [ ] Card condition grading

---

**Enjoy managing your Magic collection! 🃏✨**
