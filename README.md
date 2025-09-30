# AI-Powered Legal Adviser: A Study on Low-Resource Languages

### Table of Contents
* [Introduction](#introduction)
* [Project Overview](#project-overview)
* [Aim and Objectives](#aim-and-objectives)
* [Analysis Approach](#analysis-approach) 
* [Features](#features)
* [System Requirements](#system-requirements)
* [Technologies Used](#technologies-used)
* [References](#references)
* [Contact Information](#contact-information)

### Introduction:  
<div align="justify">This repository contains the code for an LLM-powered Conversational AI system designed to resolve constitutional and legal queries, initially tailored for Nepal, but also adaptable for other low-resource languages. </div>

### Project Overview:  
<div align="justify">
The goal of this project is to develop an easy-to-use platform that empowers public to quickly and accurately obtain legal information and clarification through AI-powered responses. The system leverages a Large Language Model (LLM) integrated with a backend server and a frontend interface, working together to deliver accurate, accessible, and multilingual legal assistance.  </div><br>

**System Components:**

**Backend:**
 <div align="justify">The server processes incoming user queries using Natural Language Processing (NLP) and a RAG-based transformer architecture. It interprets legal texts and retrieves contextually relevant responses from a knowledge base. The LLM is fine-tuned on Nepali constitutional and legal data, ensuring legally accurate and reliable answers.</div><br>

**Frontend:**
<div align="justify">The user interface offers a seamless and interactive experience, supporting English, Nepali, and Romanized Nepali languages. It is designed with a focus on accessibility and simplicity, making legal information usable even for individuals with no technical or legal background.</div><br>

<div align="justify">This platform acts as a virtual legal assistant, helping bridge the gap between complex legal language and the everyday understanding of citizens. By focusing on low-resource languages, it aims to democratize access to justice, enhance legal literacy, and provide support to individuals who may not have easy access to legal professionals or services.
 </div>


### Aim and Objectives: 
**Aim:**
<div align="justify"> 
The primary aim of this research is to develop an LLM-based conversational AI system capable of interpreting and responding to legal queries in multiple languages (English, Nepali, Romanized Nepali). The system will provide simplified answers to constitutional and legal queries based on the Nepali Constitution and related legal documents. </div><br>

**Objectives:**
<div align="justify"> 

- LLM Pipeline Development: LLM Pipeline Development: Build a robust LLM pipeline leveraging a RAG-based transformer architecture, fine-tuned for legal texts in low-resource languages. <br>
- Data Collection Challenges: Identify challenges in collecting, cleaning, and processing legal data for low-resource languages (font issues, text extraction, mojibake, etc.). <br>
- User-Friendly Conversational AI: Develop a multilingual conversational AI that addresses legal language complexity and promotes inclusive access to justice. <br>
- Performance Analysis: Compare the performance of leading LLM models ( GPT, BERT-Base, Gemini) in interpreting legal queries in a low-resource language domain.

</div>

### Analysis Approach:    
<div align="justify">To tackle this problem effectively, I have established a structured analysis approach. This research adopts a Mixed-Method Approach for both system development and evaluation:</div><br>

**1. Qualitative Approach:**
- Legal Language Analysis: Review Nepali legal texts, including the Nepali Constitution and laws, to understand the legal terminology, structure, and context.
- Expert Consultation: Conduct interviews with legal professionals to identify accessibility challenges and user needs. 

**2. Quantitative Approach:**
- Performance Evaluation: Measure system performance through metrics like cosine similarity, accuracy, response time, F1 score, and user satisfaction.
- User Testing: Collect user feedback through surveys and evaluations to refine the system’s capabilities.

**3. Data Collection and Preprocessing:**
- Sources: Collect legal texts from official sources, including the Nepali Constitution, laws, and public legal databases.
- Preprocessing: Use Python and OCR for text extraction, followed by tokenization, lemmatization, and data cleaning, along with novel techniques such as cross-language translation, advanced prompt engineering, and chunk description generation to enhance corpus quality for training.

**4. Model Selection and Training:**
- Pre-trained LLMs: Use models like GPT, BERT, and Gemini as base architectures and fine-tune them using the RAG-based transformer approach.
- Vectorization: Use models like Google's text-embedding-004 for semantic understanding and store vectors in Pinecone for fast retrieval.
 </div>

**5.Testing and Validation:**
- Evaluation Metrics: Use cosine similarity, F1 score, BLEU score, and average response time to measure the system’s accuracy and performance. The code for evaluation metrics has been provided in .ipynb file format.
- User Surveys: Conduct satisfaction surveys to gather qualitative feedback on clarity, usefulness, and overall user experience.
- Feedback Integration: Incorporate user and expert feedback to iteratively improve response accuracy and conversational quality.

### Features:

- Backend powered by Python 3.12 with RAG-based AI server.
- Frontend built with modern Node.js (v22).
- Simple setup with virtual environment for backend.
- Interactive web UI running on localhost.

### System Requirements: 

**Running the Backend:**
**Requirements:**
- Python 3.12
- venv module to create virtual environments
- Flask (and other backend dependencies as defined in requirements.txt)

**Steps:**

**Open Command Prompt, then navigate to the backend directories**

**1. Create Virtual Environment:**
```bash
python -m venv legaladviser
```

**2. Activate Virtual Environment (For Windows):**
```
.\legaladviser\Scripts\activate
```

**3. Install Dependencies:**
```
pip install -r requirements.txt
```
**4. Run the Server:**
```
python ./rag_server.py
```
**The server will run on port 8000 by default.**

**Running the Frontend:**
**Requirements:**
- Node.js v22 or higher
- npm for package management

**Steps:**

**Open Command Prompt, then navigate to the frontend directories**

**1.	Install Dependencies:** 
```
npm install
```
**2.	Start the Server:**
```
npm run dev
```

**The frontend UI will be available at http://localhost:3000.**

**Running the Application**

**Step 1: Start Backend**

Ensure the backend server is running on port 8000 by following the instructions in the Backend Setup section.

**Step 2: Start Frontend**

Ensure the frontend server is running on port 3000 by following the instructions in the Frontend Setup section.

**Step 3: Access the Application**

Once both the backend and frontend servers are running, you can access the application by opening a browser and visiting http://localhost:3000. The UI will allow you to enter queries and receive responses from the AI system.

**Project Structure**
```
/project-root
│
├── backend/
│   ├── rag-server.py               # Python backend server file
│   ├── requirements.txt            # Backend dependencies
│   ├── .env                        # Environment variables for backend
│   
│
├── frontend/
│   ├── /public                     # Static files (images, icons, etc.)
│   ├── /src                        # Source code for React app
│   ├── package.json                # Frontend dependencies and scripts
│   └── .env                        # Environment variables for frontend
│
└── README.md                       # Project documentation
```

### Technologies Used:
- Backend: Python 3.12, Flask, Node.js v22
- LLMs & NLP: GPT, BERT, Gemini, Hugging Face, Sentence Transformers
- Frontend: React.js, HTML5, CSS3, JavaScript
- Data Storage and Authentication: Server-Sent Events (SSE), Firebase
- Vector Database: Pinecone
- Dev Tools: Jupyter, VS Code, PyCharm, Git, GitHub
- Environments & Dependencies: venv, pip, npm 

### References:
- Python Documentation  
- Flask Documentation  
- LLM, OpenAI Documentation   
- Hugging Face Transformers Documentation
- Vector Database Documentation  
- Node.js & React.js Documentation  
- Jupyter Notebook, VS Code, PyCharm Official Docs 
- Kaggle 
- Stack Overflow  
- Research papers and tutorials on Gen AI, LLM, RAG, Conversational AI, and Low-Resource Language Processing

### Contact Information:
Created by https://github.com/Erkhanal - feel free to contact!