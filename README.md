# Multiple-PDF-QA-Chatbot
The Multiple PDF Q&amp;A Chatbot allows users to upload multiple PDF files for content analysis. Using Langchain, it extracts text and processes queries with advanced language models. Each response includes file source information and page numbers, facilitating efficient information retrieval and analysis of large document sets.

## Features

- Upload multiple PDF files for content extraction.
- Ask questions related to the uploaded documents.
- Contextual responses utilizing OpenAI's language model.
- Detailed answer sourcing, indicating which document and page the information is derived from.

## Requirements

- Python 3.7 or higher
- Gradio
- OpenAI SDK
- LangChain Community
- LangChain Chroma
- Dotenv

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/isilbekci/Multiple-PDF-QA-Chatbot.git
   cd Multiple-PDF-QA-Chatbot

2. Install the required packages:

   ```bash
   pip install -r requirements.txt

3. Set up your environment variables. Create a `.env` file in the root directory and add your OpenAI API key:

   ```plaintext
   OPENAI_API_KEY=your_api_key_here
   
## Usage

To run the chatbot interface, execute the following command:

```bash
python multiple_pdf_qa_chatbot.py
