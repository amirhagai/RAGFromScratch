from dotenv import load_dotenv
import os

def load_env_keys():
    load_dotenv()

    # Access environment variables
    os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2', 'false')  # Default to 'false' if not set
    os.environ['LANGCHAIN_ENDPOINT'] = os.getenv('LANGCHAIN_ENDPOINT', 'https://default.endpoint')
    os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.environ['COHERE_API_KEY'] = os.getenv('COHERE_API_KEY')
    os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
    os.environ['USER_AGENT'] = os.getenv('USER_AGENT', 'myagent')


def offline_transformers():
    os.environ['TRANSFORMERS_OFFLINE'] = os.getenv('TRANSFORMERS_OFFLINE')