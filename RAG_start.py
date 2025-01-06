from load_env_keys import load_env_keys, offline_transformers
load_env_keys()
offline_transformers()
import torch
import os
from transformers import set_seed
set_seed(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import re
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema.document import Document
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
import transformers
from get_llama import get_llama
from get_embedding_model import EmbedModel
from chromadb import Client
from chromadb.config import Settings
import chromadb


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

def embed(splits, k=5, delete_old=False, collection_name="example_collection", add_files_to_existing_db=True):


    # if delete_old:
    client_settings = Settings(persist_directory="./chroma_langchain_db")
    client = chromadb.PersistentClient(path="./chroma_langchain_db", settings=client_settings)

        # Check if the collection exists
        # existing_collections = [col.name for col in client.list_collections()]
        # if collection_name in existing_collections:
        #     if delete_old:
        #         # Delete the existing collection
        #         client.delete_collection(collection_name)
    vectorstore_raw_obj = Chroma(collection_name=collection_name,
                                        persist_directory="./chroma_langchain_db",
                                        client=client, 
                                        client_settings=client_settings,
                                        embedding_function=EmbedModel())
    if vectorstore_raw_obj._collection.count() == 0:
        vectorstore = Chroma.from_documents(documents=splits, 
                                            embedding=EmbedModel(), #OpenAIEmbeddings(),
                                            collection_name=collection_name,
                                            persist_directory="./chroma_langchain_db",
                                            client=client, 
                                            client_settings=client_settings)
    else:
        vectorstore = vectorstore_raw_obj

    if add_files_to_existing_db:
        # Retrieve existing document IDs from the collection
        existing_docs = vectorstore ._collection.get(include=["documents"])
        existing_content = set(existing_docs["documents"])
        # Example list of files with unique IDs
        files_to_add = splits

        # Identify files that are not in the collection
        files_missing = [file for file in files_to_add if file.page_content not in existing_content]
        for file in files_missing:
            vectorstore.add_texts(
                texts=[file.page_content],
                metadatas=[file.metadata]
            )


    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return vectorstore, retriever



def load_prompt_and_model(prompt_identifier="rlm/rag-prompt", model_name="gpt-3.5-turbo", max_new_tokns=256):
    
    prompt = hub.pull(prompt_identifier)
    #print(dict(prompt)['messages'][0].prompt.template)
    # llm = ChatOpenAI(model_name=model_name, temperature=0)
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

    llm = get_llama(max_new_tokens=max_new_tokns)
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


def get_multy_queries_generator(str_num_queries, llm, question="What is Task Decomposition?"):
    # Multi Query: Different Perspectives
    template = f"""You are an AI language model assistant. Your task is to generate {str_num_queries}
    different versions of the given user question to retrieve relevant documents from a vector 
    database. please number the questions (e.g 1., 2., ...) By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    please only provide the questions and nothing else
    Provide these alternative questions separated by newlines. Original question: {{question}}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)
    # llm = ChatOpenAI(temperature=0)
    res = llm(prompt_perspectives.invoke(question).messages[0].content, pipeline_kwargs={'return_full_text' : False})
    result = re.sub(r'^\s*$\n', '', res, flags=re.MULTILINE)
    simmilar_queries = (lambda x: x.split("\n"))(result)
    return lambda x : simmilar_queries


def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

def use_pipe_directly_example(llm, msg):
    return llm.pipeline(
                    [msg],
                    **{'return_tensors' : True},
                )

def get_multy_query_rag_retrival_chain(retriever, llm, how_many_questions, question):
    # Retrieve
    generate_queries =  get_multy_queries_generator(how_many_questions, llm, question)
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    return retrieval_chain, generate_queries

def multy_query_rag(retriever, prompt, llm, question, how_many_questions="five"):
    retrieval_chain, generate_queries  = get_multy_query_rag_retrival_chain(retriever, llm, how_many_questions, question)
    retrive_prompt_chain = (
        {"context": retrieval_chain, "question": itemgetter("question")} 
        | prompt
    )
    retrive_prompt = retrive_prompt_chain.invoke({"question":question})
    return retrieval_chain, generate_queries, retrive_prompt_chain, retrive_prompt 


def get_retrive_prompt_chain(retriever, prompt):
    retrive_prompt_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
    )
    return retrive_prompt_chain

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (f'document - {loads(doc).page_content}', f'relevance according to Reciprocal rank fusion  - {score}')
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

def rag_fusion(retriever, prompt, llm, question="What is Task Decomposition?", how_many_questions="five"):
    _, generate_queries  = get_multy_query_rag_retrival_chain(retriever, llm, how_many_questions, question)
    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
    # docs = retrieval_chain_rag_fusion.invoke({"question": question})
    retrive_prompt_chain = (
        {"context": retrieval_chain_rag_fusion, "question": itemgetter("question")} 
        | prompt
    )
    retrive_prompt = retrive_prompt_chain.invoke({"question":question})
    return retrive_prompt

def generate_query_decomposition(llm, question):
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n you should only output 3 such queries, with no additional explanations. so your answer should only contain 3 lines, line for each query.
    queries should be numberd 1, 2, 3. please stop generate text after the thierd query. 
    Output (3 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    queries_decomposition_prompt = (prompt_decomposition).invoke({"question":question})
    queries_decomposition_prompt = [p.to_string() for p in [queries_decomposition_prompt]][0]
    questions = llm.invoke(queries_decomposition_prompt.replace('Human', 'user'), pipeline_kwargs={'return_full_text' : False}).split("\n")

    def remove_numeric_prefix(text):
        # Define the regular expression pattern for an integer followed by '. ' at the start of the string
        pattern = r'^\d+\. '
        # Use re.sub() to replace the pattern with an empty string
        return re.sub(pattern, '', text)

    questions = list(set([remove_numeric_prefix(q).strip() for q in questions if (q != ' ' and q != '\n')]))
    return questions




def format_qa_pair(question, answer):
    """Format Q and A pair"""    
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()


def rag_query_translation_decomposition(retriever, llm, question="What is Task Decomposition?"):
    questions = generate_query_decomposition(llm, question)
    questions = questions + [question]

    template = """Here is the question you need to answer:

    \n --- \n {question} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question: 

    \n --- \n {context} \n --- \n

    Use the above context and any background question + answer pairs to answer the question: \n {question}
    please answer as shortly as possible and include the final answer only with no additional information
    """

    decomposition_prompt = ChatPromptTemplate.from_template(template)

    q_a_pairs = ""
    for q in questions:
        
        decompose_chain = (
        {"context": itemgetter("question") | retriever, 
        "question": itemgetter("question"),
        "q_a_pairs": itemgetter("q_a_pairs")} 
        | decomposition_prompt)

        q_with_info = decompose_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
        q_with_info = [p.to_string() for p in [q_with_info]][0]
        answer = llm.invoke(q_with_info.replace('Human', 'user'), pipeline_kwargs={'return_full_text' : False})
        q_a_pair = format_qa_pair(q,answer)
        q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
    return answer



def llama_rag(return_full_text):
    docs = load_web_doc()
    text_splitter, splits = split_docs(docs=docs)
    vectorstore, retriever = embed(splits=splits)
    prompt, llm = load_prompt_and_model()
    retrive_prompt_chain = get_retrive_prompt_chain(retriever, prompt)
    retrive_prompt = retrive_prompt_chain.invoke("What is Task Decomposition?")
    respones = llm.invoke(retrive_prompt.messages[0].content, pipeline_kwargs={'return_full_text' : return_full_text})
    return docs, text_splitter, splits, vectorstore, retriever, prompt, llm, retrive_prompt_chain, retrive_prompt, respones

def llama_multy_query_rag():
    docs = load_web_doc()
    text_splitter, splits = split_docs(docs=docs)
    vectorstore, retriever = embed(splits=splits, delete_old=True)
    prompt, llm = load_prompt_and_model(max_new_tokns=512)
    retrieval_chain, generate_queries, retrive_prompt_chain, retrive_prompt = multy_query_rag(retriever, prompt, llm, question="What is Task Decomposition?", how_many_questions="five")

    print("############################################", end="\n\n\n")
    result = llm.invoke(retrive_prompt.messages[0].content, pipeline_kwargs={'return_full_text' : False})
    print(result)

def llama_RAG_fusion():
    docs = load_web_doc()
    text_splitter, splits = split_docs(docs=docs)
    vectorstore, retriever = embed(splits=splits, delete_old=True)
    prompt, llm = load_prompt_and_model(max_new_tokns=512)
    retrive_prompt = rag_fusion(retriever, prompt, llm, question="What is Task Decomposition?", how_many_questions="five")
    msg = [p.to_string() for p in [retrive_prompt]][0]
    result = llm.invoke(msg.replace('Human', 'user'), pipeline_kwargs={'return_full_text' : False})
    print(result)

def llama_RAG_query_translation_decomposition():
    docs = load_web_doc()
    text_splitter, splits = split_docs(docs=docs)
    vectorstore, retriever = embed(splits=splits, delete_old=True)
    prompt, llm = load_prompt_and_model(max_new_tokns=512)
    answer = rag_query_translation_decomposition(retriever, llm, question="What is Task Decomposition?")
    print(answer)

if __name__ == '__main__':
    llama_RAG_query_translation_decomposition()
    llama_RAG_fusion()
    # print("##########")
    llama_multy_query_rag()
    # docs, text_splitter, splits, vectorstore, retriever, prompt, llm, retrive_prompt_chain, retrive_prompt, respones = llama_rag(return_full_text=False)
    # print(respones)