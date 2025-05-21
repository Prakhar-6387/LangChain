from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate,PromptTemplate,MessagesPlaceholder)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
import streamlit as st



base_url = "http://localhost:11434"
model_name = "llama3.2:1b"

from dotenv import load_dotenv

load_dotenv('/home/prakhar-tiwari/Desktop/myProjects/AI-Learn/LangChain/lanchain-smith-all/.env')

llm = ChatOllama(
  base_url = base_url,
    model = model_name,
    temperature = 0.8,
    num_predict = 256,
    # other params ...
)

def get_session_history(session_id):
  return SQLChatMessageHistory(session_id, "sqlite:///chat_history1.db");


user_id = st.text_input("Enter the userId", "prakhar")

# setting title
st.title(" Hello Welcom to chat application..")


# if chat_history not in session

if "chat_history" not in st.session_state:
  st.session_state.chat_history=[]


# if someone start the new converstaion
if st.button(" Press this to start new conversation.."):
  st.session_state.chat_history=[]
  histroy = get_session_history(user_id)
  histroy.clear()


# if someone on page so we have to load its conversation
for message in st.session_state.chat_history:
  with st.chat_message(message['role']):
    st.markdown(message['content'])


system = SystemMessagePromptTemplate.from_template("you are helful assitant")
human = HumanMessagePromptTemplate.from_template("{input}")

prompt1 = ChatPromptTemplate.from_messages([
    system,
    MessagesPlaceholder(variable_name="history"),
    human
])

chain = prompt1  | llm | StrOutputParser()


runnable_with_history= RunnableWithMessageHistory(chain , get_session_history, input_message_key = 'input', history_messages_key = 'history')

def chat_with_llm(session_id, input):
  for output in runnable_with_history.stream({'input': input}, config = {'configurable' : {'session_id': session_id}}):
    yield output


prompt = st.chat_input("whats is up ?")
# st.write(prompt)

if prompt:
  st.session_state.chat_history.append({'role': "user" , "content": prompt})

  with st.chat_message("user"):
    st.markdown(prompt)

  with st.chat_message("assistant"):
    response = st.write_stream(chat_with_llm(user_id, prompt))

  st.session_state.chat_history.append({'role': "assistant" , "content": response})