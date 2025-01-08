# ChatBot RAG

ChatBot RAG (Retrieval-Augmented Generation) is a PDF-based chatbot application that leverages advanced embedding models and a Qdrant vector database to provide context-aware responses to user queries. It uses the LangChain Community and Hugging Face tools to process, store, and retrieve document data efficiently.

## Features

- **PDF Upload and Analysis**: Upload PDF documents to analyze and generate vector embeddings.
- **Vector Database Storage**: Store document embeddings in a Qdrant vector database for efficient retrieval.
- **Interactive Chat**: Engage with the chatbot to retrieve document-specific information in natural language.
- **Multi-Model Support**: Modular design allows using different embedding and language models.

---

## Installation

### Prerequisites

1. Install Python 3.8 or higher.
2. Install and run a Qdrant server. For local testing, use Docker:
   ```bash
   docker run -p 6333:6333 -d qdrant/qdrant
   ```

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yusufbek-murodov/chatbot-rag.git
   cd chatbot-rag
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

---

## How It Works

### Application Workflow

1. **Upload PDF**: Users upload a PDF document via the web interface.
2. **Preview PDF**: The uploaded document is displayed for verification.
3. **Create Embeddings**:
    - The application processes the PDF, splitting it into chunks.
    - Each chunk is converted into a vector embedding using Hugging Face models.
    - The embeddings are stored in a Qdrant collection for retrieval.
4. **Interactive Chat**:
    - Users type queries in the chat interface.
    - The chatbot retrieves relevant context from the Qdrant database.
    - The response is generated based on the retrieved context.

---

## Code Overview

### 1. **app.py**

This is the main entry point for the Streamlit application. It handles:

- **PDF Upload**: Allows users to upload a PDF and displays a preview.
- **Embedding Creation**: Generates embeddings for the uploaded document and stores them in Qdrant.
- **Interactive Chat Interface**: Provides a chat interface for users to interact with the bot.

### 2. **chatbot.py**

Defines the `ChatbotClass` for managing chatbot interactions. Key functionalities:

- **Prompt Template**: Customizes the chatbot's response behavior.
- **Qdrant Integration**: Manages context retrieval from the Qdrant database.
- **Response Generation**: Uses LangChain RetrievalQA to generate answers based on retrieved context.

### 3. **vectors.py**

Defines the `Embeddings` class for managing document embeddings. Key functionalities:

- **PDF Processing**: Uses LangChain tools to load and split PDF documents.
- **Embedding Generation**: Converts document chunks into vector embeddings using Hugging Face models.
- **Database Storage**: Stores embeddings in the Qdrant vector database.

---

## Usage

### Uploading a PDF
1. Open the application in your browser.
2. Upload a PDF document via the provided interface.
3. Confirm the file details and preview.

### Creating Embeddings
1. Select the **Create Embeddings** checkbox.
2. Wait for the process to complete.
3. Success message confirms embeddings are stored in Qdrant.

### Chat Interaction
1. Use the chat input box to type queries.
2. View responses generated based on the PDF content.

---

## Example

### Upload and Chat

1. Upload a file named `example.pdf`.
2. Generate embeddings.
3. Ask: "What are the key points in this document?"

**Expected Response**:
Detailed answers based on the document content.

---

## Dependencies

- Python 3.8+
- Streamlit
- LangChain Community
- Hugging Face Transformers
- Qdrant Client

---

## Future Enhancements

1. Support for multiple document formats (e.g., Word, TXT).
2. Integration with cloud-based Qdrant instances.
3. Enhanced UI for better user interaction.
4. Asynchronous processing for large document uploads.

---

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

## Acknowledgments

- [LangChain Community](https://github.com/langchain-community) for providing robust tools.
- [Hugging Face](https://huggingface.co) for state-of-the-art embedding models.
- [Qdrant](https://qdrant.tech) for the vector database solution.

---

For more details, visit [GitHub Repository](https://github.com/yusufbek-murodov/chatbot-rag).

