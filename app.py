import gradio as gr
from huggingface_hub import InferenceClient
from typing import List, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss


client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

class MyApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_pdf("Education_Tutor_for_engineering_students.pdf")
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
        # Generate embeddings for all document contents
        self.embeddings = model.encode([doc["content"] for doc in self.documents])
        # Create a FAISS index
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        # Add the embeddings to the index
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Generate an embedding for the query
        query_embedding = model.encode([query])
        # Perform a search in the FAISS index
        D, I = self.index.search(np.array(query_embedding), k)
        # Retrieve the top-k documents
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant documents found."]

app = MyApp()
   

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    system_message = "You are a educational tutor for engineering students. You support learning journey in engineering, helping with complex concepts, exam preparation tips or guide through steps to make project.Discuss what's on your mind and ask me for a quick guidance in studies."
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value = "You are a educational tutor for engineering students. You support learning journey in engineering, helping with complex concepts, exam preparation tips or guide through steps to make project.Discuss what's on your mind and ask me for a quick guidance in studies."),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],

    examples = [ 
        ["I feel struggled with engineering concepts"],
        ["Can you guide me through a quick steps to make a python project"],
        ["can you describe basic components of electrical circuits and their functions"],
        ["What are the principles of mechanics"],
        ["Which programming language is used by most of the companies and how can I learn it?"],
        ["What are different queries that are commonly used in SQL?"],
        ["How can I do surveying and mapping in civil engineering?"],
        ["How many research papers are published on Artificial Intelligence?"],
        ["How can I crack a entry-level exam of JEE?"],
        ["Can you please give the roadmap of C++ for beginner who don't know anything about coding"],
        ["How can I build an application using RAG for LLM chatbot?"],
        ["Display step by step guide to install active directory on windows server?"],
        ["Define DHCP, DNS, TCP/IP"],
        ["What are the different topics and from where I have to study for passing Comp TIA A+ exam"],
        ["Disply all the information about Loops in all programming languages"],
        ["what I need to configure and test VPN on the windows server?"],
        ["Display all the information about the basics of Python Language"],
        ["Display different certificates do I need to get a entry-level job in IT with no experience"],
        ["How can I handle my coursework with other commitments ?"]
    ],
    title = '📝🖋️Educational Tutor for Engineering Students📝🖋️'
)


if __name__ == "__main__":
    demo.launch()
