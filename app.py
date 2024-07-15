import gradio as gr
from huggingface_hub import InferenceClient
from typing import List, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Placeholder for the app's state
class MyApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_pdf("THEDIA1.pdf")
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        """Extracts text from a PDF file and stores it in the app's documents."""
        doc = fitz.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        """Builds a vector database using the content of the PDF."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = model.encode([doc["content"] for doc in self.documents])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant documents found."]

app = MyApp()

def respond(
    message: str,
    history: List[Tuple[str, str]],
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    system_message = "You are a knowledgeable DBT coach. You always talk about one options at at a time. you add greetings and you ask questions like real counsellor. Remember you are helpful and a good listener. You are concise and never ask multiple questions, or give long response. You response like a human counsellor accurately and correctly. consider the users as your client. and practice verbal cues only where needed. Remember you must be respectful and consider that the user may not be in a situation to deal with a wordy chatbot.  You Use DBT book to guide users through DBT exercises and provide helpful information. When needed only then you ask one follow up question at a time to guide the user to ask appropiate question. You avoid giving suggestion if any dangerous act is mentioned by the user and refer to call someone or emergency."
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    # RAG - Retrieve relevant documents
    retrieved_docs = app.search_documents(message)
    context = "\n".join(retrieved_docs)
    messages.append({"role": "system", "content": "Relevant documents: " + context})

    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=100,
        stream=True,
        temperature=0.98,
        top_p=0.7,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

demo = gr.Blocks()

with demo:
    gr.Markdown(
        "‼️Disclaimer: This chatbot is based on a DBT exercise book that is publicly available. and just to test RAG implementation.‼️"
    )
    
    chatbot = gr.ChatInterface(
        respond,
        examples=[
            ["I feel overwhelmed with work."],
            ["Can you guide me through a quick meditation?"],
            ["How do I stop worrying about things I can't control?"],
            ["What are some DBT skills for managing anxiety?"],
            ["Can you explain mindfulness in DBT?"],
            ["I am interested in DBT excercises"],
            ["I feel restless. Please help me."],
            ["I have destructive thoughts coming to my mind repetatively."]
        ],
        title='Dialectical Behaviour Therapy Assistant👩‍⚕️🧘‍♀️'
    )

if __name__ == "__main__":
    demo.launch()