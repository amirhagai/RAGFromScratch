from load_env_keys import load_env_keys, offline_transformers
load_env_keys()
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from operator import itemgetter


def load_web_doc(web_addresses="https://lilianweng.github.io/posts/2023-06-23-agent/"):
    loader = WebBaseLoader(
        web_paths=(web_addresses,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    return docs

def split_docs(docs, chunk_size=500, chunk_overlap=300):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    return text_splitter, splits

def embed(splits, k=5):
    vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=OpenAIEmbeddings(),
                                        collection_name="example_collection",
                                        persist_directory="./chroma_langchain_db",)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return vectorstore, retriever



def load_prompt_and_model(prompt_identifier="rlm/rag-prompt", model_name="gpt-3.5-turbo"):
    
    prompt = hub.pull(prompt_identifier)
    #print(dict(prompt)['messages'][0].prompt.template)
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    # llm = HuggingFacePipeline.from_model_id(
    # model_id="google/flan-t5-large", #"Qwen/Qwen2.5-32B-Instruct",
    # task="text-generation",
    # device=-1, 
    # pipeline_kwargs={
    #     "max_new_tokens": 1000,
    #     "top_k": 1,
    #     "temperature": 1e-3,
    # },  
    # )
    offline_transformers()
    return prompt, llm


# # Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_chat_prompt(custom_instruction="Answer the question based only on the following context:"):
    template = f"""{custom_instruction}
    {{context}}

    Question: {{question}}
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt


def get_multy_queries_generator(str_num_queries):
    # Multi Query: Different Perspectives
    template = f"""You are an AI language model assistant. Your task is to generate {str_num_queries}
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {{question}}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)
    generate_queries = (
        prompt_perspectives 
        | ChatOpenAI(temperature=0) 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )
    return generate_queries


def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


def get_multy_query_rag_retrival_chain(retriever):
    # Retrieve
    generate_queries =  get_multy_queries_generator("five")
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    return retrieval_chain

def multy_query_rag(retriver, prompt, llm, question):
    retrieval_chain = get_multy_query_rag_retrival_chain(retriever)
    rag_chain = (
        {"context": retrieval_chain, "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )
    result = rag_chain.invoke({"question":question})
    return result

def define_base_chain(retriever, prompt, llm):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

docs = load_web_doc()
text_splitter, splits = split_docs(docs=docs)
vectorstore, retriever = embed(splits=splits)
prompt, llm = load_prompt_and_model()
rag_chain = define_base_chain(retriever=retriever, prompt=prompt, llm=llm)

res = rag_chain.invoke("What is Task Decomposition?")
# Question
print(rag_chain.invoke("What is Task Decomposition?"))

res = multy_query_rag(retriever, prompt, llm, "What is Task Decomposition?")
print(res)