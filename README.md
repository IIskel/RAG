# Building an Agentic RAG Pipeline

## 📋 Table of Contents
- [Overview](#overview)
- [Project Goals](#project-goals)
- [Architecture](#architecture)
- [Key Components](#key-components)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Pipeline Workflow](#pipeline-workflow)
- [Configuration & Parameters](#configuration--parameters)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Performance Considerations](#performance-considerations)
- [Future Enhancements](#future-enhancements)

## Overview

This project implements an **Agentic Retrieval-Augmented Generation (RAG) Pipeline** that intelligently combines document retrieval with large language models to answer queries. The pipeline uses an agent controller to decide whether to retrieve context from documents or answer queries directly, making it a hybrid approach to question-answering.

### Purpose
The RAG pipeline is designed to:
- Process and index PDF documents for fast retrieval
- Generate contextually relevant answers using retrieved document content
- Route queries intelligently using an agent controller
- Utilize local, lightweight language models for inference

## Project Goals

1. **Intelligent Query Routing**: Use an agent controller to determine whether a query requires document search or direct answering
2. **Efficient Document Indexing**: Split and embed documents for rapid similarity-based retrieval
3. **Context-Aware Generation**: Generate answers using both retrieved context and the local LLM
4. **Modularity**: Build reusable components that can be extended or customized
5. **Local Inference**: Run entirely with local models (no API calls required)

## Architecture

```
User Query
    ↓
Agent Controller (Keyword Analysis)
    ├─→ Route: "search" → Vector DB Retrieval → Context Assembly
    │   ↓
    └─→ Route: "direct" → Direct Query
    ↓
LLM (google/flan-t5-base)
    ↓
Generated Response
```

### Component Flow

1. **PDF Loading** → Documents are loaded and parsed
2. **Text Splitting** → Large documents are split into manageable chunks
3. **Embedding** → Text chunks are converted to vector representations
4. **Vector Storage** → Embeddings are stored in Chroma DB for fast retrieval
5. **Query Processing** → Agent controller determines routing
6. **Context Retrieval** → Relevant chunks are retrieved for the query
7. **Prompt Construction** → Retrieved context is formatted with the query
8. **Generation** → LLM generates a response based on context
9. **Output** → Answer is returned to the user

## Key Components

### 1. **PDF Document Loader**
- **Library**: `langchain_community.document_loaders.PyPDFLoader`
- **Function**: Loads PDF files from specified directories
- **Output**: List of document pages as `langchain.schema.Document` objects

### 2. **Text Splitter**
- **Type**: `RecursiveCharacterTextSplitter`
- **Parameters**:
  - `chunk_size`: 500 characters
  - `chunk_overlap`: 80 characters
- **Purpose**: Breaks large documents into overlapping chunks to preserve context

### 3. **Embedding Model**
- **Model**: `all-MiniLM-L6-v2` (from Hugging Face)
- **Framework**: `HuggingFaceEmbeddings`
- **Dimension**: 384-dimensional vectors
- **Purpose**: Converts text into numerical representations for similarity matching

### 4. **Vector Database**
- **Backend**: `Chroma`
- **Collection Name**: `rag_store`
- **Purpose**: Stores and retrieves document embeddings efficiently
- **Retrieval**: Top-k similarity search (k=3 by default)

### 5. **Language Model**
- **Model**: `google/flan-t5-base`
- **Framework**: `transformers.pipeline` (text-generation)
- **Max Tokens**: 150
- **Purpose**: Generates contextual answers based on input prompts

### 6. **Agent Controller**
- **Type**: Rule-based routing logic
- **Trigger Keywords**: "pdf", "document", "data", "summarize", "information", "find", "book"
- **Logic**: Determines whether to use RAG (search) or direct LLM response
- **Purpose**: Intelligently routes queries to the appropriate handler

## Prerequisites

### System Requirements
- Python 3.8 or higher
- At least 4GB RAM (8GB recommended for smooth operation)
- ~2GB disk space for model downloads

### Optional: Google Colab
- Google account for Colab access
- Google Drive for file uploads

## Installation

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd RAG
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
Create a `requirements.txt` file with the following packages:

```
langchain>=0.1.0
langchain-community>=0.0.1
langchain-chroma>=0.1.0
transformers>=4.30.0
sentence-transformers>=2.2.0
pypdf>=3.15.0
chromadb>=0.3.21
```

Then install:
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```python
import langchain
import transformers
import chromadb
print("All dependencies installed successfully!")
```

### First-Time Model Downloads
On first run, the embedding model (`all-MiniLM-L6-v2`) and LLM (`google/flan-t5-base`) will be automatically downloaded (~500MB total).

## Project Structure

```
RAG/
├── Building_An_agentic_RAG_pipeline.ipynb  # Main notebook with full pipeline
├── README.md                               # This file
└── requirements.txt                        # Python dependencies (to be created)
```

### Notebook Sections

| Section | Purpose |
|---------|---------|
| 1. Setup & Dependencies | Install and import required libraries |
| 2. Import & PDF Loading | Load PDF documents from folders |
| 3. Document Splitting | Split documents into chunks |
| 4. Embeddings & Vector DB | Create embeddings and store in Chroma |
| 5. LLM Loading | Initialize the local language model |
| 6. Agent Controller | Define routing logic for queries |
| 7. RAG Pipeline | Implement end-to-end answer generation |

## Usage

### Running the Notebook

#### In Google Colab:
1. Open the `.ipynb` file in Google Colab
2. Run cells sequentially (Ctrl+Enter or Cmd+Enter)
3. Upload a PDF when prompted (Cell 1)
4. Follow the pipeline through each section

#### Locally:
```bash
jupyter notebook Building_An_agentic_RAG_pipeline.ipynb
```

### Basic Query Flow

```python
# Query that triggers document search (RAG)
response = rag_answer("Give me a 5-point summary from the PDF")

# Query that triggers direct LLM response
response = rag_answer("What is machine learning?")
```

## Pipeline Workflow

### Workflow for Document-Specific Query
```
1. User: "Give me a 5-point summary from the PDF"
   ↓
2. Agent Controller: Detects "PDF" keyword → Route = "search"
   ↓
3. Vector DB: Retrieve top 3 relevant chunks
   ↓
4. Prompt Assembly: "Use this context: [retrieved text] Answer: [query]"
   ↓
5. LLM: Generate answer using context
   ↓
6. Return: Contextual answer based on document
```

### Workflow for General Knowledge Query
```
1. User: "What is machine learning?"
   ↓
2. Agent Controller: No trigger keywords → Route = "direct"
   ↓
3. Prompt Assembly: "What is machine learning?"
   ↓
4. LLM: Generate answer from its training knowledge
   ↓
5. Return: General knowledge answer
```

## Configuration & Parameters

### Text Splitting Configuration
Modify in Cell 3:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,          # Increase for larger context windows
    chunk_overlap=80         # Increase for more context overlap
)
```

**Impact**:
- Smaller chunks: Faster retrieval, less context
- Larger chunks: Slower retrieval, more context
- Higher overlap: Better context continuity, more redundant data

### Retriever Configuration
Modify in Cell 4:
```python
retriever = db.as_retriever(search_kwargs={"k": 3})  # k = number of docs to retrieve
```

**Impact**:
- k=1: Fastest, minimal context
- k=5: Slower, richer context
- k=10: Very slow, redundant information

### LLM Parameters
Modify in Cell 5:
```python
llm = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_new_tokens=150  # Maximum length of generated response
)
```

**Impact**:
- max_new_tokens: Controls response length (50-200 recommended)

### Agent Controller Keywords
Modify in Cell 6:
```python
if any(word in q for word in ["pdf", "document", "data", "summarize", "information", "find", "book"]):
    return "search"
```

Add or remove keywords to adjust routing behavior.

## Examples

### Example 1: Summarization Query
```python
query = "Summarize the key concepts from the document"
response = rag_answer(query)
# Output: 🕵️ Agent decided to SEARCH document for: '...'
#         [Generated summary based on document content]
```

### Example 2: General Knowledge Query
```python
query = "Explain neural networks"
response = rag_answer(query)
# Output: 🤖 Agent decided to answer DIRECTLY: '...'
#         [Generated explanation from LLM training]
```

### Example 3: Data Extraction Query
```python
query = "Find all data preprocessing techniques mentioned in the PDF"
response = rag_answer(query)
# Output: 🕵️ Agent decided to SEARCH document for: '...'
#         [Extracted information from document]
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'langchain'"
**Solution**: Reinstall dependencies
```bash
pip install --upgrade langchain langchain-community langchain-chroma
```

### Issue: "Model download takes too long"
**Solution**: Pre-download models
```python
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")  # Pre-downloads embedding model
```

### Issue: "Out of memory error"
**Solution**:
- Reduce `chunk_size` to 250-300
- Reduce `max_new_tokens` to 100
- Use `k=2` for retriever instead of k=3

### Issue: "Chroma database is empty"
**Solution**: Verify PDF was uploaded and chunks were created
```python
print(f"Chunks Created: {len(chunks)}")
print(f"First chunk: {chunks[0].page_content[:100]}")
```

### Issue: "Agent controller not routing correctly"
**Solution**: Check query keywords match defined triggers
```python
print(agent_controller("your query here"))  # Returns "search" or "direct"
```

## Performance Considerations

### Speed Optimization
| Action | Impact |
|--------|--------|
| Reduce chunk_size | ↑ Retrieval speed, ↓ Context quality |
| Lower k in retriever | ↑ Generation speed, ↓ Context richness |
| Use smaller LLM | ↑ Inference speed, ↓ Answer quality |

### Memory Optimization
| Component | Memory Usage |
|-----------|-------------|
| Embedding model | ~120 MB |
| LLM (flan-t5-base) | ~800 MB |
| Chroma DB (1000 chunks) | ~200 MB |
| **Total** | **~1.1 GB** |

### Retrieval Metrics
- Average retrieval time for 1000 chunks: ~50-100ms
- Document indexing time: Varies with PDF size
- Answer generation time: 2-5 seconds per query

## Future Enhancements

### Planned Features
- [ ] Support for additional document formats (DOCX, TXT, CSV)
- [ ] Multi-model retrieval (semantic + keyword hybrid)
- [ ] Query expansion for improved retrieval
- [ ] Chat history management for context awareness
- [ ] Fine-tuned models for domain-specific tasks
- [ ] Response ranking based on relevance scores
- [ ] Batch query processing
- [ ] Custom vector database persistence

### Potential Model Upgrades
- **Embedding Models**: `all-mpnet-base-v2` (slower but more powerful)
- **LLMs**: `google/flan-t5-large` (better quality, more resources)
- **Specialized Models**: Domain-specific fine-tuned models

### Architecture Extensions
- Real-time chat interface (Streamlit/Gradio)
- REST API for production deployment
- Multiple PDF collection management
- User feedback loop for continuous improvement