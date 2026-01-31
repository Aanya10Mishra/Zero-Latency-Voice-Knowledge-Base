# 🎙️ Zero-Latency Voice Knowledge Base

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![Groq](https://img.shields.io/badge/LLM-Groq-orange.svg)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A voice-powered RAG (Retrieval-Augmented Generation) system with sub-800ms Time-To-First-Byte (TTFB) for real-time technical assistance.**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-endpoints) • [Contributing](#-contributing)

</div>

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| 🎤 **Voice Input** | Real-time speech recognition using Google Speech API |
| 🔊 **Voice Output** | Natural text-to-speech responses with pyttsx3 |
| ⚡ **Low Latency** | Sub-800ms TTFB with filler responses while processing |
| 🔍 **Hybrid Search** | Combines vector similarity + BM25 keyword search |
| 🎯 **Cross-Encoder Reranking** | Improves search accuracy with neural reranking |
| 🧠 **LLM Generation** | Powered by Groq's ultra-fast Llama 3.1 |
| 💬 **Context-Aware** | Maintains conversation history for follow-up questions |
| 🎨 **Modern Web UI** | Beautiful, responsive interface with real-time metrics |
| 🛑 **Interruptible** | Say "stop" or "quit" anytime to interrupt |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ZERO-LATENCY VOICE RAG                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  Voice  │───▶│    Query     │───▶│   Hybrid    │───▶│   Cross-  │  │
│  │  Input  │    │   Rewriter   │    │   Search    │    │   Encoder   │  │
│  │  (ASR)  │    │   (Groq)     │    │ (Vec+BM25)  │    │   Rerank    │  │
│  └─────────┘    └──────────────┘    └─────────────┘    └─────────────┘  │
│       │                                                       │         │
│       │         ┌──────────────────────────────────────────────         │
│       │         │                                                       │
│       │         ▼                                                       │
│       │    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│       │    │     LLM     │───▶│    Voice    │───▶│   Voice    │        │
│       │    │  Generator  │    │  Optimizer  │    │   Output    │        │
│       │    │   (Groq)    │    │             │    │   (TTS)     │        │
│       │    └─────────────┘    └─────────────┘    └─────────────┘        │
│       │                                                   │             │
│       └───────────────────────────────────────────────────┘             │
│                          (Conversation Loop)                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **ASR** | Google Speech Recognition | Convert voice to text |
| **Query Rewriter** | Groq Llama 3.1 | Context-aware query expansion |
| **Vector Search** | ChromaDB + Sentence Transformers | Semantic similarity search |
| **BM25 Search** | rank-bm25 | Keyword-based search |
| **Reranker** | Cross-Encoder (ms-marco) | Neural relevance scoring |
| **LLM Generator** | Groq Llama 3.1 | Response generation |
| **Voice Optimizer** | Custom | Optimize text for speech |
| **TTS** | pyttsx3 | Convert text to speech |

---

## 📁 Project Structure

```
Zero Latency-Voice Knowledge Base/
│
├── 📄 main.py                    # FastAPI server & endpoints
├── 📄 config.py                  # Configuration settings
├── 📄 run_indexer.py             # Document indexing script
├── 📄 voice_chat_optimized.py    # Voice chat interface
├── 📄 test_debug.py              # Testing utilities
├── 📄 .env                       # Environment variables
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # This file
│
├── 📁 data/
│   └── 📄 technical_manual.pdf   # Your PDF document
│
├── 📁 indexing/
│   ├── 📄 __init__.py
│   ├── 📄 chunking.py            # Voice-optimized text chunking
│   └── 📄 document_processor.py  # PDF processing
│
├── 📁 pipeline/
│   ├── 📄 __init__.py
│   ├── 📄 asr.py                 # Speech recognition
│   ├── 📄 hybrid_search.py       # Vector + BM25 search
│   ├── 📄 query_rewriter.py      # Query expansion
│   ├── 📄 reranker.py            # Cross-encoder reranking
│   ├── 📄 llm_generator.py       # LLM response generation
│   ├── 📄 voice_optimizer.py     # Text optimization for TTS
│   └── 📄 tts.py                 # Text-to-speech
│
├── 📁 utils/
│   ├── 📄 __init__.py
│   └── 📄 conversation_memory.py # Conversation history
│
├── 📁 static/
│   ├── 📁 css/
│   │   └── 📄 style.css          # UI styles
│   └── 📁 js/
│       └── 📄 app.js             # Frontend JavaScript
│
└── 📁 templates/
    └── 📄 index.html             # Web UI template
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- Windows 10/11 (also works on macOS/Linux)
- Microphone (for voice input)
- Speakers (for voice output)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/zero-latency-voice-rag.git
cd zero-latency-voice-rag
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Additional Dependencies for Voice

```bash
pip install SpeechRecognition pyaudio

# If pyaudio fails on Windows:
pip install pipwin
pipwin install pyaudio
```

### Step 5: Set Up Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your-groq-api-key-here
```

Get your free Groq API key at: https://console.groq.com/keys

### Step 6: Add Your Document

Place your PDF in the `data/` folder:

```bash
mkdir data
copy "C:\path\to\your\manual.pdf" "data\technical_manual.pdf"
```

### Step 7: Index the Document

```bash
python run_indexer.py
```

You should see:
```
Starting document indexing...
Reading: ./data/technical_manual.pdf
Found 150 pages
  Batch 1/30 done (100/3000 chunks)
  ...
✅ Successfully indexed 3000 chunks!
```

---

## 💻 Usage

### Option 1: Web UI (Recommended)

Start the server:

```bash
python main.py
```

Open your browser and go to:

```
http://localhost:8000
```

**Features:**
- 💬 Type or speak your questions
- 🎤 Click the microphone for voice input
- 🔊 Toggle voice output on/off
- 📊 View real-time performance metrics
- 📚 See source documents

### Option 2: Voice Chat (Terminal)

```bash
python voice_chat_optimized.py
```

Choose your mode:
- **Option 1:** Simple Voice Chat (more reliable)
- **Option 2:** Advanced Interruptible Chat

**Voice Commands:**
| Command | Action |
|---------|--------|
| *Your question* | Ask anything about the manual |
| "stop" / "cancel" | Interrupt current response |
| "quit" / "exit" | End the session |

### Option 3: API Only

```bash
# Start the server
python main.py

# Query via curl
curl -X POST "http://localhost:8000/query?query=What%20is%20this%20about&conversation_id=test1"
```

---

## 🔌 API Endpoints

### `GET /`
Serves the web UI.

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### `POST /query`
Main query endpoint.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | The user's question |
| `conversation_id` | string | Session ID for context |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/query?query=How%20do%20I%20create%20a%20table&conversation_id=session1"
```

**Example Response:**
```json
{
  "original_query": "How do I create a table",
  "rewritten_query": "How to create a table in the database",
  "response": "To create a table, you can use the CREATE TABLE command followed by the table name and column definitions...",
  "sources": [
    {"page": "45", "score": -0.85},
    {"page": "46", "score": -0.72},
    {"page": "12", "score": -0.65}
  ]
}
```

---

## ⚡ Performance

### Latency Breakdown

| Stage | Time | Optimization |
|-------|------|--------------|
| Speech Recognition | ~500ms | Google Speech API |
| Query Rewriting | ~200ms | Groq (fast inference) |
| Hybrid Search | ~50ms | ChromaDB + BM25 |
| Reranking | ~150ms | Cross-Encoder |
| LLM Generation | ~300ms | Groq Llama 3.1 |
| Text-to-Speech | ~100ms | pyttsx3 (offline) |
| **Total** | **~1300ms** | |
| **TTFB (with filler)** | **~50ms** | Filler response |

### TTFB Optimization

The system achieves near-zero TTFB by:
1. **Immediate filler response** - "Let me look that up..."
2. **Parallel processing** - RAG runs while filler plays
3. **Streaming output** - Response streams as it generates

---

## 🛠️ Configuration

Edit `config.py` to customize:

```python
@dataclass
class Config:
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # Search Settings
    VECTOR_TOP_K: int = 10
    RERANK_TOP_K: int = 5
    
    # Chunking Settings
    MAX_CHUNK_WORDS: int = 150
    OVERLAP_WORDS: int = 20
    
    # LLM Settings
    LLM_MODEL: str = "llama-3.1-8b-instant"
    MAX_TOKENS: int = 300
    TEMPERATURE: float = 0.7
    
    # TTS Settings
    TTS_RATE: int = 160
```

---

## 🧪 Testing

### Test All Components

```bash
python test_debug.py
```

Expected output:
```
1. Testing config...
   GROQ_API_KEY set: True
   Key starts with: gsk_xxxxx...

2. Testing HybridSearch...
   ✓ HybridSearch initialized

3. Testing QueryRewriter...
   ✓ QueryRewriter initialized

4. Testing search...
   ✓ Search returned 10 results

5. Testing LLM...
   ✓ LLM initialized

✅ All components working!
```

### Test Voice Output

```bash
python test_voice.py
```

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'indexing'` | Run with `python -m indexing.document_processor` or use `run_indexer.py` |
| `chroma-hnswlib build error` | Install Visual Studio Build Tools or use `pip install chromadb==0.4.15` |
| `pyaudio installation failed` | Use `pipwin install pyaudio` on Windows |
| `GROQ_API_KEY not found` | Check your `.env` file exists and has the correct key |
| `Search returned 0 results` | Run `python run_indexer.py` to index your PDF |
| `localhost:8000 not reachable` | Make sure `python main.py` is running |

### Still Having Issues?

1. Check the terminal for error messages
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Verify your PDF is in the `data/` folder
4. Make sure the Groq API key is valid

---

## 📊 Tech Stack

| Category | Technology |
|----------|------------|
| **Backend** | FastAPI, Python 3.10+ |
| **LLM** | Groq (Llama 3.1 8B Instant) |
| **Vector Database** | ChromaDB |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Reranking** | Cross-Encoder (ms-marco-MiniLM-L-6-v2) |
| **Speech Recognition** | Google Speech API |
| **Text-to-Speech** | pyttsx3 |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Styling** | Custom CSS with animations |

---

## 🗺️ Roadmap

- [x] Core RAG pipeline
- [x] Hybrid search (Vector + BM25)
- [x] Cross-encoder reranking
- [x] Voice input/output
- [x] Web UI
- [x] Conversation memory
- [ ] Multi-document support
- [ ] WebSocket streaming
- [ ] Docker deployment
- [ ] Custom voice selection
- [ ] Multi-language support

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Groq](https://groq.com/) for ultra-fast LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework

---

<div align="center">

## UI of the Application

<img width="1846" height="905" alt="Screenshot 2026-01-31 153829" src="https://github.com/user-attachments/assets/521ff719-10ba-4662-91c7-a264a73bdce3" />


##  Features Added

<img width="467" height="583" alt="Screenshot 2026-01-31 153900" src="https://github.com/user-attachments/assets/a5d863cc-4e72-4841-b881-e154db539337" />

⭐ Star this repo if you find it useful!

</div>
