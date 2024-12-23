from sentence_transformers import SentenceTransformer

# This model supports two prompts: "s2p_query" and "s2s_query" for sentence-to-passage and sentence-to-sentence tasks, respectively.
# They are defined in `config_sentence_transformers.json`
query_prompt_name = "s2p_query"
queries = [
    "What are some ways to reduce stress?",
    "What are the benefits of drinking green tea?",
]
# docs do not need any prompts
docs = [
    "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.",
    "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.",
]
class EmbedModel:

    def __init__(self, model_name="dunzhang/stella_en_1.5B_v5", model_cache_path="/RAG/models--dunzhang--stella_en_1.5B_v5"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True, cache_folder=model_cache_path).cuda()
        self.query_prompt_name_1 = "s2p_query"
        self.query_prompt_name_2 = "s2s_query"

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self.model.encode(documents, convert_to_tensor=True).tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode(query, prompt_name=self.query_prompt_name_1, convert_to_tensor=True).tolist()
    
    def embed(self, docs, query):
        if query == 0:
            return self.model.encode(docs)
        elif query == 1:
            return self.model.encode(docs, prompt_name=self.query_prompt_name_1)
        elif query == 2:
            return self.model.encode(docs, prompt_name=self.query_prompt_name_2)
        
    def get_similarities(self, query_embeddings, doc_embeddings):
        return self.model.similarity(query_embeddings, doc_embeddings)



if __name__ == '__main__':
    a = EmbedModel()
    docs_emb = a.embed(docs=queries, query=1)
    q_emb = a.embed(docs=docs, query=0)
    q_emb = a.embed_query(docs)

    print(a.get_similarities(q_emb, docs_emb))