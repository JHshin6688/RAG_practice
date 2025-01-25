import os
from openai import OpenAI

from dotenv import load_dotenv

#from langchain.document_loaders import PyMuPDFLoader
#from langchain.embeddings import OpenAIEmbeddings
#from langchain_community.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import Chroma
#from langchain.chat_models import ChatOpenAI
#from langchain.retrievers import WikipediaRetriever

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI

from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#import warnings
#warnings.filterwarnings("ignore", category= DeprecationWarning)

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"), 
)

def ask_chatgpt1(question):
    chatbot_role = "You are a helpful assistant"

    response = client.chat.completions.create(
        messages=[
            {"role" : "system", "content" : chatbot_role},
            {"role": "user", "content": question}
        ],
        model="gpt-4o"
    )

    return response.choices[0].message.content

#question = "세계에서 가장 높은 산은 무엇인가요?"
#response = ask_chatgpt1(question)
#print(response)

def ask_chatgpt2(sys_role, question):
    response = client.chat.completions.create(
        messages = [
            {"role" : "system", "content" : sys_role},
            {"role" : "user", "content" : question}
        ],
        model="gpt-4o"
    )
    return response.choices[0].message.content

#sys_role = "당신은 아름답고 감동적인 시를 창작하는데 영감을 주는 시적인 천재입니다.\
#당신의 시는 감정의 깊이, 자연의 아름다움, 인간 경험의 복잡성을 탐구하는 작품이며, 당신의 시를 읽는 이들의 마음을 움직입니다."

#question = "생성형 AI라는 주제로 시를 지어줘. 운율에 맞춰서 작성해줘."
#response = ask_chatgpt2(sys_role, question)
#print(response)

chat = ChatOpenAI(model_name = "gpt-3.5-turbo")
#chat = ChatOpenAI(model_name = "gpt-3.5-turbo", streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

#result = chat([HumanMessage(content= "안녕하세요!")])  ->  deprecated
#result = chat.invoke([HumanMessage(content= "안녕하세요!")])
#result2 = chat.invoke("안녕하세요!")
#print(result2.content)

#sys_role = "당신은 대한민국 국민입니다"
#question = "독도는 어느나라 땅이야?"
#result = chat.invoke([HumanMessage(content = question), SystemMessage(content = sys_role)])
#print(result.content)

wiki_retriever = WikipediaRetriever(lang = 'ko')
wiki_chat = RetrievalQA.from_llm(
    llm = chat,
    retriever = wiki_retriever,
    return_source_documents = False
)

question = "지미카터 대통령의 출신 대학은?"
result = wiki_chat.invoke(question)
print(result["result"])
