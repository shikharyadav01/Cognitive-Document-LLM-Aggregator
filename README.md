# Cognitive-Document-LLM-Aggregator
The Cognitive Document LLM Aggregator is a Python application that allows you to chat with multiple documents. You can ask questions about the DOCs using natural language, and the application will provide relevant responses based on the content of the documents. This app utilizes a language model to generate accurate answers to your queries. Please note that the app will only respond to questions related to the loaded PDFs.

How It Works
Cognitive Document LLM Aggregator Diagram

<img width="1046" height="583" alt="Screenshot 2026-05-20 151804" src="https://github.com/user-attachments/assets/6055465b-e365-4f66-8994-c1c64a1db575" />


The application follows these steps to provide responses to your questions:

PDF Loading: The app reads multiple PDF documents and extracts their text content.

Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.

Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.

Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.

Dependencies and Installation
To install the Cognitive Document LLM Aggregator, please follow these steps:

Clone the repository to your local machine.

Install the required dependencies.

Obtain an API key from OpenAI

