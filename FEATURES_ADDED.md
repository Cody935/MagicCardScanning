# New Collection Features - Magic Scanner

## Overview
Three amazing new features have been added to enhance your Magic card collection management!

---

## Feature 1: 2-Column Grid Layout 📊

### What Changed
- Cards in your collection now display **2 per row** instead of a single column
- This gives you a better horizontal view of your collection at a glance
- More efficient use of screen space on larger monitors
- Responsive design: reverts to 1 column on mobile devices (< 768px width)

### Where to See It
- Go to **My Collection** page
- All your cards now display in a neat 2-column grid

### Code Changes
- Updated `.collection-grid` CSS from `grid-template-columns: repeat(auto-fill, minmax(250px, 1fr))` to `grid-template-columns: repeat(2, 1fr)`

---

## Feature 2: Price Graph Button 📈

### What It Does
- Click the **📈 Price Graph** button directly across from the Delete button
- Automatically opens the card's **TCGPlayer price history page** in a new tab
- Shows real-time price trends and market data for that specific card

### How to Use It
1. Look at any card in your collection
2. Under the 4 details (Set, Rarity, Price, Added)
3. Click the green **"📈 Price Graph"** button on the right
4. Your default browser will open the TCGPlayer product page

### UI Details
- **Color**: Green accent (rgba(34, 193, 138, ...))
- **Position**: Right side of the card actions area
- **Hover Effect**: Interactive green highlight

### Technical Implementation
- New route: `/api/card-info/<card_id>` - fetches TCGPlayer ID
- Opens: `https://www.tcgplayer.com/product/{tcgplayer_id}`
- Secure: Only works for cards in user's collection

---

## Feature 3: Custom Card Art Selection 🎨

### What It Does
- **Click on any card image** to open the art selection modal
- Browse **up to 10 different card printings/arts** from TCGPlayer
- Select a specific art variant to display in your collection
- Changes persist - your selection is saved to the database

### How to Use It

1. **Open Art Selection Modal**
   - Click on any card image in your collection
   - The art selection modal will pop up

2. **Browse Available Arts**
   - See all available printings for that card
   - Cards are displayed in a grid format
   - Hover over any art to preview it

3. **Select Your Preferred Art**
   - Click on the art you want to select
   - A green checkmark (✓) appears on selected art
   - The **"Select This Art"** button becomes enabled

4. **Confirm Your Selection**
   - Click **"Select This Art"** button
   - Your collection card image updates immediately
   - Selection is saved to the database

### UI Details
- **Modal**: Beautiful dark theme with glass-morphism effect
- **Art Grid**: Responsive grid showing multiple variants
- **Selection Indicator**: Green checkmark on selected art
- **Color Scheme**: Green/Teal accents for selection (matches your app theme)

### Data Persistence
- Selected art URL is stored in the `Card` model field: `selected_art_url`
- The app prioritizes showing custom selected art over default art
- Fallback chain: `selected_art_url` → `image_url` → "No Image Available"

### Technical Implementation

**Database Changes:**
- New field in `Card` model: `selected_art_url = db.Column(db.String(500))`

**New API Endpoints:**
- `GET /api/card-arts/<tcgplayer_id>` - Fetches available arts from TCGPlayer
- `GET /api/card-info/<card_id>` - Gets card's TCGPlayer ID
- `POST /update-card-art/<card_id>` - Updates selected art in database

**Frontend:**
- New modal interface with art selection grid
- JavaScript functions:
  - `openArtSelectionModal(cardId, cardName, cardData)`
  - `confirmArtSelection()`
  - `closeArtSelectionModal()`
  - `openPriceGraph(cardId, cardName)`

---

## Summary of Changes

### Files Modified:
1. **magic/app/app.py**
   - Added `selected_art_url` field to Card model
   - Added 3 new API routes
   - Added `update_card_art()` function

2. **magic/app/templates/collection.html**
   - Updated grid layout to 2 columns
   - Added Price Graph button (📈)
   - Added modal for art selection
   - Added JavaScript event handlers
   - Updated card display to show selected art prioritization
   - Added new CSS styles for buttons and modal

### Database Migration Note:
Run this in your Flask shell to add the new column:
```python
from magic.app.app import app, db, Card
with app.app_context():
    db.session.execute('ALTER TABLE card ADD COLUMN selected_art_url VARCHAR(500)')
    db.session.commit()
```

Or simply delete `cards.db` and restart - it will recreate with the new schema.

---

## Features at a Glance

| Feature | Button | Color | Action |
|---------|--------|-------|--------|
| **Price Graph** | 📈 Price Graph | Green | Opens TCGPlayer price page |
| **Delete** | Delete | Red | Removes card from collection |
| **Select Art** | Click card image | N/A | Opens art selection modal |

---

## Visual Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                    📚 My Collection                             │
├─────────────────────────────────────────────────────────────────┤
│ [Card 1]                    │ [Card 2]                          │
│ ┌─────────────────────────┐ │ ┌─────────────────────────┐      │
│ │                         │ │ │                         │      │
│ │   Click to Select       │ │ │   Click to Select       │      │
│ │        Art              │ │ │        Art              │      │
│ │                         │ │ │                         │      │
│ └─────────────────────────┘ │ └─────────────────────────┘      │
│ Set: Dominaria              │ Set: M21                         │
│ Rarity: Rare                │ Rarity: Uncommon                 │
│ Price: $5.99                │ Price: $0.50                     │
│ Added: Jan 18, 2026         │ Added: Jan 18, 2026              │
│ ┌──────────────┐ ┌─────────┐ │ ┌──────────────┐ ┌─────────────┐│
│ │ 📈 Price G.. │ │ Delete  │ │ │ 📈 Price G.. │ │ Delete      ││
│ └──────────────┘ └─────────┘ │ └──────────────┘ └─────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Testing the Features

### Test 1: Verify 2-Column Grid
- ✅ Add multiple cards to your collection
- ✅ Check that cards display in 2 columns on desktop
- ✅ Check that cards display in 1 column on mobile/tablet

### Test 2: Test Price Graph Button
- ✅ Click "📈 Price Graph" on any card
- ✅ Should open TCGPlayer product page in new tab
- ✅ Verify TCGPlayer ID is correctly fetched from card data

### Test 3: Test Art Selection
- ✅ Click on a card image
- ✅ Modal should pop up with art options
- ✅ Select different art variants
- ✅ Click "Select This Art" button
- ✅ Verify card image updates immediately
- ✅ Refresh page - verify selection persists
- ✅ Verify database stores `selected_art_url`

---

## Troubleshooting

### Art Selection Modal Shows "No additional card arts available"
- The card may not have a TCGPlayer ID
- Some older or custom cards may not have variants available
- This is expected behavior - not all cards have multiple printings

### Price Graph Opens Wrong Page
- Ensure the card's `tcgplayer_id` was correctly captured when added
- Try re-adding the card using a clearer image

### Images Not Loading in Art Modal
- Check your internet connection (fetching from TCGPlayer API)
- Try again in a few moments
- Some images may be temporarily unavailable on TCGPlayer

### Database Errors
- If you see database column errors, delete `cards.db` and restart
- The app will recreate the database with the new schema

---

## Enjoy Your Enhanced Collection! 🎉

These features make managing and showcasing your Magic card collection even more awesome. Happy collecting!
