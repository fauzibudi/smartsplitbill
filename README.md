# Smart Split Bill 🧾

A Smart Split Bill application that automatically extracts bill details and helps split expenses among friends using Streamlit and advanced computer vision.

## Features ✨

- **AI Receipt Processing**: Uses Donut AI model to extract text from receipt images
- **Smart Item Detection**: Automatically identifies menu items, prices, and quantities
- **Flexible Bill Splitting**: Split bills equally or proportionally based on items consumed
- **Multi-format Support**: Works with JPG, PNG, and JPEG receipt images
- **Export Results**: Download split results as JSON for easy sharing
- **Real-time Processing**: Instant receipt analysis and bill calculation

## Tech Stack 🛠️

- **Frontend**: Streamlit
- **AI/ML**: Transformers, PyTorch, Donut model
- **Image Processing**: PIL/Pillow
- **Text Processing**: SentencePiece
- **Data Processing**: Pandas, NumPy

## Installation 🚀

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/fauzibudi/smartsplitbill.git
   cd smart-split-bill
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the application**
   - Open your browser to `http://localhost:8501`

## Usage 📖

### Step 1: Load AI Models
- Click "Load AI Models" button on startup
- Wait for models to download and initialize (first time only)

### Step 2: Upload Receipt
- Upload a clear receipt image (JPG/PNG format)
- Ensure the receipt is well-lit and readable

### Step 3: Review Extracted Data
- Check the extracted receipt information
- Verify items, prices, and quantities are correct

### Step 4: Split the Bill
- Enter names of people splitting the bill (comma-separated)
- Assign each item to the person who ordered it
- Choose split method: Equal or Proportional
- Review final amounts for each person

### Step 5: Export Results
- Download the split results as JSON file
- Share with friends for easy payment tracking

## Supported Receipt Formats 📄

The AI model supports various receipt formats including:
- Restaurant bills
- Cafe receipts
- Bar tabs
- Retail receipts

## Model Information 🤖

- **Model**: naver-clova-ix/donut-base-finetuned-cord-v2
- **Purpose**: Document understanding and receipt parsing
- **Accuracy**: High accuracy on structured receipts
- **Languages**: Optimized for English receipts

## Troubleshooting 🔧

### Common Issues

**Models won't load**
- Check internet connection
- Ensure sufficient disk space (~2GB for models)
- Verify all dependencies are installed

**Receipt not processing**
- Ensure image is clear and well-lit
- Check image format (JPG/PNG supported)
- Try rotating the image if text appears sideways

**Incorrect item detection**
- Manual review and correction is available
- Ensure receipt is not too blurry or damaged

### Error Messages

- **"Models not loaded properly"**: Re-run the application and reload models
- **"Failed to parse receipt data"**: Try with a clearer receipt image
- **"Receipt processing failed"**: Check image format and quality

## Development 🏗️

### Project Structure
```
smart-split-bill/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── [receipt images]      # Sample receipt images
```

### Reasons for Choosing donut-base-finetuned-cord-v2 then donut-base

- The donut-base-finetuned-cord-v2 model is specifically fine-tuned on the CORD (Consolidated Receipt Dataset), which consists of receipt images from various sources (e.g., restaurants, retail) with structured annotations for items, quantities, prices, subtotals, taxes, and totals. The donut-base model is a general-purpose pre-trained model for document understanding, not fine-tuned for any specific task like receipt parsing. It lacks the specialized training on receipt-specific data, making it less accurate and reliable for extracting structured information from receipts. The fine-tuning on CORD ensures that cord-v2 can handle the specific layout, terminology, and structure of receipts (e.g., item lists, totals) with high precision, producing consistent JSON outputs like {"menu": [...], "sub_total": {...}, "total": {...}}.

### Shortcomings

- Dependency on a single model without a backup.
- Lack of image quality validation.
- Slow processing due to reliance on torch CPU.
- Limited handling of currency and number formats.
- Lack of support for multilingual receipts.
- Potential performance issues with heavy loads or large images.

### General Recommendations

- Add Alternative Models: Integrate other models if the current model struggles to understand the image.
- Improve Parsing: Enhance the parsing function to handle different receipt formats and languages.
- Log Errors: Use file-based logging for debugging without disrupting the user experience.

## License 📄

This project is open source and available under the [MIT License](LICENSE).
