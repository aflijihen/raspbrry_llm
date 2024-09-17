import os
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pushbullet import Pushbullet
import logging


class RecommendationEngine:

    def __init__(self, openai_api_key, pushbullet_api_key, gps_lat=None, gps_long=None):
        self.docs_dir = "./handbook/"
        self.persist_dir = "./handbook_faiss/"
        self.embedding = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
        self.llm = ChatOpenAI(api_key=openai_api_key, temperature=0.6)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.API_KEY = pushbullet_api_key
        self.pb = Pushbullet(self.API_KEY)
        self.gps_lat = gps_lat
        self.gps_long = gps_long
        self.__load_or_build_faiss()
        self.__initialize_qa_chain()

    def __load_or_build_faiss(self):
        if os.path.exists(self.persist_dir):
            logging.info(f"Loading FAISS index from {self.persist_dir}")
            self.vectorstore = FAISS.load_local(self.persist_dir, self.embedding, allow_dangerous_deserialization=True)
        else:
            self.__build_faiss_index()

    def __initialize_qa_chain(self):
        general_system_template = r"""
        You are spirulina AI controller agent.
        You will receive input in JSON format with parameters such as "temperature", "conductivity", "brightness", etc.
        You will generate a JSON output with recommended actions and status analysis.
        Format example:
        {
            "commands": [COMMAND_OBJ1, COMMAND_OBJ2, ...],
            "recommendations": [
                {
                    "parameter": "temperature",
                    "status": "low/high",
                    "action": "increase/decrease value"
                },
                ...
            ]
        }
        ----
        {context}
        ----
        """
        general_user_template = "Question:```{question}```"

        messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
        ]
        qa_prompt = ChatPromptTemplate.from_messages(messages)

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            memory=self.memory,
            retriever=self.vectorstore.as_retriever(),
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )

    def __build_faiss_index(self):
        loader = DirectoryLoader(
            self.docs_dir,
            loader_cls=PyPDFLoader,
            recursive=True,
            silent_errors=True,
            show_progress=True,
            glob="**/*.pdf"
        )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
        frags = text_splitter.split_documents(docs)
        logging.info(f"Populating vector store with {len(docs)} docs in {len(frags)} fragments")
        self.vectorstore = FAISS.from_documents(frags, self.embedding)
        logging.info(f"Persisting vector store to: {self.persist_dir}")
        self.vectorstore.save_local(self.persist_dir)
        logging.info(f"Saved FAISS index to {self.persist_dir}")

    def generate_recommendation(self, data):
        result = self.qa_chain.invoke({"question": data})
        generated_response = result["answer"]
        logging.info("Recommendations: " + generated_response)
        return generated_response

    def notify(self, title, notes):
        self.pb.push_note(title, notes)





# import os
# import logging
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from PyPDF2 import PdfReader  # Pour lire le fichier PDF
# from pushbullet import Pushbullet

# class RecommendationEngine:

#     def __init__(self, pushbullet_api_key, gps_lat, gps_long):
#         self.gps_lat = gps_lat
#         self.gps_long = gps_long
#         self.API_KEY = pushbullet_api_key
#         self.pb = Pushbullet(self.API_KEY)

#         # Charger TinyGPT
#         self.model_name = "gpt2"  # Vous pouvez choisir un autre modèle GPT léger
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = GPT2LMHeadModel.from_pretrained(self.model_name).to(self.device)
#         self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
    
#     def load_document(self, file_path):
#         """
#         Charge un document PDF et retourne son texte sous forme de chaîne.
#         """
#         reader = PdfReader(file_path)
#         text = ""
#         for page in reader.pages:
#             text += page.extract_text()
#         logging.info(f"Texte du document chargé ({len(text)} caractères)")
#         return text
    
#     def generate_recommendation(self, data, document_text):
#         # Préparer les données des capteurs et le texte du document sous forme de texte
#         input_text = f"Données des capteurs : {data}\n\nDocument : {document_text}\n\nQue recommandez-vous ?"
        
#         # Tokeniser et générer du texte
#         inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
#         outputs = self.model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)

#         # Convertir les résultats générés en texte
#         generated_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         logging.info("Recommandations générées: " + generated_response)
#         return generated_response

#     def notify(self, title, notes):
#         self.pb.push_note(title, notes)

