
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

# Instantiate the app
app = MyApp()

# Define the respond function for the chatbot
def respond(
    message: str,
    history: List[Tuple[str, str]],
    system_message: str = "You are a knowledgeable consultant for solar panel installation and maintenance. You provide concise and accurate advice on sustainability practices, and you encourage environmentally friendly habits. Remember to greet the user warmly, ask relevant questions, and offer supportive and insightful responses.",
    max_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
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
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

# Create the Gradio interface
demo = gr.Blocks()

with demo:
    gr.Markdown("🌍 **Solar Panel Installation and MaintenanceAdvisorS**")
    gr.Markdown(
        "‼️Disclaimer: This chatbot is based on publicly available sustainability guidelines and practices. "
        "We are not certified sustainability experts, and the use of this chatbot is at your own responsibility.‼️"
    )
    
    chatbot = gr.ChatInterface(
        respond,
        examples=[
            ["What are the benefits of installing solar panels?"],
            ["How do I choose the right solar panel for my home?"],
            ["What is the best way to maintain solar panels?"],
            ["Can you explain the installation process for solar panels?"],
            ["How can I maximize the efficiency of my solar panels?"],
            ["What are the common issues with solar panels and how to fix them?"],
            ["How do I clean my solar panels?"],
            ["What factors affect the performance of solar panels?"]

        ],
        title='Solar Panel Installation and MaintenanceAdvisorS 🌞'
    )

if __name__ == "__main__":
    demo.launch()
