from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from langchain_ollama import ChatOllama


class ChatbotClass:
    def __init__(self, model_name, device, llm_model, temperature, encode_kwargs, qdrant_url, connection_name):
        self.model_name = model_name
        self.device = device
        self.llm_model = llm_model
        self.temperature = temperature
        self.encode_kwargs = encode_kwargs
        self.qdrant_url = qdrant_url
        self.connection_name = connection_name

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs
        )
        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=self.temperature
        )
        self.prompt_template = '''
        You are professional analyser.
        If you don't know the answer, just say that you don't know.
        Answers should be helpful and well explained with details'''

        self.client = QdrantClient(
            url=self.qdrant_url,
            prefer_grpc=False,
        )

        self.db = QdrantClient(
            client=self.client,
            embeddings=self.embeddings,
            connection_name=self.connection_name
        )

        self.retriever = self.db.as_retriever(
            search_kwargs={"k": 1}
        )

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=['context', 'question']
        )

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_document=False
        )

    def get_response(self, query):
        try:
            answer = self.qa.run(query)
        except:
            answer = 'Error'
        return answer
