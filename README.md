# Class Chat Bot

A Streamlit-based RAG (Retrieval-Augmented Generation) application for querying lecture content using vector databases and LLMs.

## Features

- Multiple vector database support (ChromaDB, FAISS)
- Document upload and text input capabilities
- Real-time query processing with LLM
- Efficient content extraction and summarization
- User-friendly Streamlit interface

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running locally
- OpenAI API key (for GPT-3.5-turbo)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd class_chat_bot2
```

2. Create and activate a virtual environment:
```bash
# For Ubuntu/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

5. Install and start Ollama:
```bash
# For Ubuntu/Linux
curl https://ollama.ai/install.sh | sh
ollama serve

# For Windows
# Download and install from https://ollama.ai/download
```

6. Pull required models:
```bash
ollama pull llama3.2
```

## Usage

1. Start the application:
```bash
streamlit run src/app.py
```

2. Access the application in your browser at `http://localhost:8501`

3. Select operation mode:
   - Create New Database: Upload documents or input text
   - Query Existing Database: Select a database and ask questions

## Project Structure

```
class_chat_bot2/
├── src/
│   ├── app.py                 # Main Streamlit application
│   ├── utils/
│   │   ├── db_utils.py       # Vector database utilities
│   │   ├── llm_utils.py      # LLM model utilities
│   │   └── embeddings.py     # Embedding model utilities
│   └── config/
│       ├── config.py         # Configuration settings
│       └── models_config.yaml # Model configurations
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## Configuration

- Vector databases: ChromaDB, FAISS
- Embedding model: nomic-embed-text
- LLM models: llama3.2 (extraction), gpt-3.5-turbo (reasoning)
- Maximum concurrent users: 5
- Document processing: Single document processing for optimal performance

## Troubleshooting

1. If Ollama connection fails:
   - Ensure Ollama is running (`ollama serve`)
   - Check if the models are properly pulled
   - Verify OLLAMA_BASE_URL in config

2. If OpenAI API calls fail:
   - Verify your API key in .env file
   - Check internet connectivity
   - Ensure you have API credits available

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines] 