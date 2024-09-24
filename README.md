# Document-based-chatbot
---

### Overview

This project demonstrates the creation of a chatbot system that integrates various tools and techniques to process, analyze, and respond to user queries based on PDF documents. It is designed for semantic question-answering using advanced language models.

---

### Architecture

The chatbot system is structured with the following components:

1. **PdfReader**: Extracts text from PDF documents.
2. **CharacterTextSplitter**: Breaks down extracted text into manageable chunks.
3. **OpenAIEmbeddings**: Converts these chunks into embeddings for semantic understanding.
4. **Vector Database (FAISS)**: Stores and retrieves embeddings efficiently.
5. **LLMs (Language Models)**: Handles question answering using embeddings.

Process:
- The system extracts text, splits it into chunks, generates embeddings, and stores them in FAISS.
- User queries are converted into embeddings and searched against stored embeddings to retrieve relevant text.
- Language models generate context-aware responses.

---

### Tools and Techniques

- **Text Extraction**: PyPDF2 library is used to extract content from PDFs.
- **Text Splitting**: LangChain’s `CharacterTextSplitter` divides the extracted text into smaller chunks for easier processing.
- **Embeddings**: LangChain's `OpenAIEmbeddings` converts chunks into numerical representations (embeddings).
- **Vector Storage**: FAISS is used to store these embeddings for fast semantic search.
- **Question-Answering**:
  - OpenAI’s ChatOpenAI
  - Mistral 7b model
  - Other models experimented with include HuggingFace Instruct SentenceTransformer (`sentence-transformers/all-MiniLM-L6-v2`).

**Why these techniques?**
- **Performance**: Selected LLMs offer high performance for natural language understanding and generation.
- **Ease of Implementation**: Libraries like PyPDF2, FAISS, and LangChain make the implementation straightforward.
- **Customizability**: The system is flexible for further customization, such as prompt changes and model fine-tuning.

---

### Demo

A working demonstration was presented, showcasing the full pipeline from PDF text extraction to question-answering using the chatbot interface.


![Image 1](![image](https://github.com/user-attachments/assets/b1d4ca2a-3400-4859-9921-e4b5d85d5b7d)
)
![Image 2 - Chatbot](![image](https://github.com/user-attachments/assets/d6d00a60-2956-43ff-975c-d1d6017326d0)
)

---

### Conclusion

This project offers a foundation for building intelligent, responsive chatbots using state-of-the-art natural language models and vector databases. It provides a flexible system that can be expanded and improved for more complex tasks, such as automated reasoning and report generation.

---
