import streamlit as st
import tiktoken
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Semantic FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

h1 {
    color: #1565C0;
    text-align: center;
}

.stButton>button {
    width:100%;
    background-color:#1565C0;
    color:white;
    border-radius:8px;
    height:45px;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #

st.title("🤖 Semantic FAQ Chatbot")

st.write("Ask any AI-related question using natural language.")

# ---------------- LOAD MODEL ---------------- #

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- TOKENIZER ---------------- #

encoding = tiktoken.get_encoding("cl100k_base")

# ---------------- FAQ QUESTIONS ---------------- #

faq_questions = [

"What is Artificial Intelligence?",

"What is Machine Learning?",

"What is Deep Learning?",

"What is Natural Language Processing?",

"What is Tokenization?",

"What are Embeddings?",

"What is Semantic Similarity?",

"What is a Transformer?",

"What is ChatGPT?",

"What is Generative AI?",

"What is Fine Tuning?",

"What is RAG?",

"What is Hallucination in AI?",

"What is Prompt Engineering?",

"What is Data Science?"

]

# ---------------- FAQ ANSWERS ---------------- #

faq_answers = [

"Artificial Intelligence is the simulation of human intelligence by machines capable of learning, reasoning and decision making.",

"Machine Learning is a subset of Artificial Intelligence that allows systems to learn automatically from data.",

"Deep Learning is a branch of Machine Learning that uses neural networks with multiple hidden layers.",

"Natural Language Processing is the field of AI that enables computers to understand and process human language.",

"Tokenization is the process of splitting text into smaller units called tokens.",

"Embeddings are numerical vector representations of text that capture semantic meaning.",

"Semantic Similarity measures how closely two texts are related in meaning.",

"A Transformer is a deep learning architecture that uses self-attention mechanisms for language understanding.",

"ChatGPT is a conversational AI model developed by OpenAI.",

"Generative AI creates new content such as text, images, audio and code.",

"Fine Tuning means training a pre-trained model on your own dataset.",

"RAG stands for Retrieval Augmented Generation. It combines retrieval with generation to improve responses.",

"Hallucination is when an AI model generates incorrect or fabricated information.",

"Prompt Engineering is the art of designing effective prompts for AI models.",

"Data Science combines statistics, programming and machine learning to extract insights from data."

]

# ---------------- CREATE EMBEDDINGS ---------------- #

faq_embeddings = model.encode(faq_questions)

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("📘 About")

st.sidebar.info("""
Semantic FAQ Chatbot

Built using:

• Python

• Streamlit

• Sentence Transformers

• Cosine Similarity

• NLP Embeddings
""")

# ---------------- USER INPUT ---------------- #

query = st.text_input("💬 Enter your question")

# ---------------- BUTTON ---------------- #

if st.button("🔍 Get Answer"):

    if query.strip() == "":
        st.warning("Please enter a question.")

    else:

        tokens = encoding.encode(query)

        query_embedding = model.encode(query)

        scores = cosine_similarity(
            query_embedding.reshape(1, -1),
            faq_embeddings
        )

        best_match = np.argmax(scores)

        similarity_score = scores[0][best_match]

        st.markdown("---")

        st.subheader("📌 Best Matching Question")

        st.info(faq_questions[best_match])

        st.subheader("💡 Answer")

        if similarity_score >= 0.50:

            st.success(faq_answers[best_match])

        else:

            st.error("Sorry! I could not find a relevant answer.")

        st.subheader("📊 Similarity Score")

        st.progress(float(similarity_score))

        st.write(f"**{similarity_score*100:.2f}% Match**")

        st.write("**Token Count:**", len(tokens))

        if similarity_score > 0.80:

            st.success("✅ Highly Similar")

        elif similarity_score > 0.60:

            st.info("👍 Strongly Related")

        elif similarity_score > 0.40:

            st.warning("⚠ Moderately Related")

        else:

            st.error("❌ Mostly Unrelated")