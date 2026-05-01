Project Overview:

This project is an AI-powered Mental Health Support System designed to assist users by analyzing text inputs, detecting emotional states, and identifying potential psychological risk levels. It provides intelligent conversational support using a chatbot integrated with deep learning models.
The system aims to support early detection of emotional distress and provide a safe, responsive AI assistant for mental health awareness and guidance.

Objectives:
Detect user emotions from text using deep learning models
Identify potential mental health risk levels
Provide empathetic AI chatbot responses
Store chat sessions for mood tracking and analysis
Offer a scalable backend API for integration with frontend applications

System Architecture:
Frontend (future integration): Chat interface for user interaction
Backend: FastAPI REST API
AI Models:
chatbot Model (mistralai/Mistral-7B-Instruct-v0.2)
Emotion Detection Model (stacked or hirarichal Bi-LSTM + Attention layer model)
Risk Detection Model (Bi-GRU / Attention-based model)
NLP preprocessing pipeline
Database using Supabase:  Stores chat sessions and user interaction history
Model Storage: Git LFS for large .keras files

How to Run the Project:
git clone https://github.com/JumanaRayes/Ai_Mental_Health_project.git
cd Ai_Mental_Health_project

Create Virtual Environment:
python -m venv venv
source venv/Scripts/activate   # Windows

Install Dependencies:
pip install -r requirements.txt

Run Backend Server:
uvicorn app.main:app --reload   -> changed to ##uvicorn backend.app.main:app --reload

expected output For Example: 

{
  "emotion": "sad",
  "risk_level": "moderate",
  "response": "I'm really sorry you're feeling this way. I'm here to support you..."
}


GloVe Embeddings Setup:
This project uses pre-trained GloVe word embeddings to improve NLP understanding and semantic representation of text.

Download GloVe:

GloVe is not installed via pip. It must be downloaded manually from Stanford:
https://nlp.stanford.edu/projects/glove/
Download:

glove.6B.zip (recommended) Then extract:  glove.6B.300d.txt 

Project Placement
After downloading, place the file in:
AIModels/gloveEmbed/glove.6B.300d.txt
