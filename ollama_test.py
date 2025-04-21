import os    
from langchain_community.chat_models import ChatOllama

def get_chat_model(
    model_name: str,
    temperature: float,
    ):
    chat_model = ChatOllama(
        temperature=temperature,
        base_url="someurl" 
        model=model_name,
        )

    return chat_model

def get_chat_response(
    messages: list,
    model_name: str = "mistral:7b",
    temperature: float = 0.2,
) -> str:
    """Get a chat model response, given a list of input messages (i.e., a chat history)."""

    chat_model = get_chat_model(
        model_name=model_name, temperature=temperature
    )

    return str(chat_model.invoke(input=messages).content)

messages = [
    {"role": "user", "content": "Hello! Spreek je ook nederlands?"},
]

# Call the function
response = get_chat_response(messages=messages)
print(response)
