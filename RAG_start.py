from load_env_keys import load_env_keys
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

def split_docs(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    return text_splitter, splits

def embed(splits):
    vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    return vectorstore, retriever



def load_prompt_and_model(prompt_identifier="rlm/rag-prompt", model_name="gpt-3.5-turbo"):
    
    prompt = hub.pull(prompt_identifier)
    #print(dict(prompt)['messages'][0].prompt.template)
    # llm = ChatOpenAI(model_name=model_name, temperature=0)
    llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    device=0, 
    pipeline_kwargs={
        "max_new_tokens": 100,
        "top_k": 1,
        "temperature": 0,
    },
)
    return prompt, llm


# # Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def define_base_chain(retriever, prompt, llm):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        # | StrOutputParser()
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