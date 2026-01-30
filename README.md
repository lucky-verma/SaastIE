# SaastIE

An OCR-free transformer-based document understanding system that performs document classification, information extraction, and visual question answering using the Donut model architecture.

## Features

- **Document Classification**: Classify documents into categories using a fine-tuned model on RVL-CDIP dataset
- **Information Extraction**: Parse and extract structured data from documents (e.g., receipts, invoices)
- **Visual Question Answering (VQA)**: Answer natural language questions about document content

## Tech Stack

- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **ML Framework**: PyTorch, Hugging Face Transformers
- **Model**: Donut (Document Understanding Transformer)

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster inference)

### Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:lucky-verma/SaastIE.git
   cd SaastIE
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the FastAPI backend server:
   ```bash
   cd backend
   uvicorn main:app --reload --port 6969
   ```

2. In a separate terminal, start the Streamlit frontend:
   ```bash
   cd frontend
   streamlit run home.py
   ```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/classifier` | POST | Classify uploaded document image |
| `/parser` | POST | Extract structured information from document |
| `/vqa` | POST | Answer questions about document content |

## Repository Structure

```
SaastIE/
├── backend/
│   ├── main.py              # FastAPI application and API endpoints
│   ├── functions.py         # ML inference functions (classifier, parser, VQA)
│   ├── config.py            # Configuration settings
│   ├── Dockerfile           # Backend container definition
│   └── requirements.txt     # Backend-specific dependencies
├── frontend/
│   ├── home.py              # Streamlit main page
│   ├── pages/               # Streamlit multi-page app pages
│   │   ├── Classification.py
│   │   ├── Information_Extraction.py
│   │   └── Visual_Question_Answering.py
│   ├── src/                 # Static assets
│   ├── Dockerfile           # Frontend container definition
│   └── docker-compose.yml   # Docker Compose configuration
├── data/                    # Data directory for documents
├── requirements.txt         # Main project dependencies
└── rough.ipynb              # Development notebook
```

## Models Used

- `naver-clova-ix/donut-base-finetuned-rvlcdip` - Document classification
- `naver-clova-ix/donut-base-finetuned-cord-v2` - Receipt/document parsing
- `naver-clova-ix/donut-base-finetuned-docvqa` - Document VQA

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Authors

- [@lucky-verma](https://www.github.com/lucky-verma)
- [@tambi](https://www.github.com/tambi)
- [@hetankdi](https://www.github.com/hetankdi)
