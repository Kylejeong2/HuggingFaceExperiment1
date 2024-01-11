from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import requests
import os


load_dotenv(find_dotenv())

# img2text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(
        url
    )[0]['generated text']

    print(text)
    return text

#llm

def generate_story(scenario):
    template = """
    You are a story teller;
    You can generate a short story based on a simple narrative, the story should be no more than 20 words; 

    CONTEXT: {scenario}
    STORY:

    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(llm=OpenAI(
        model_name="gpt-3.5-turbo", temperature=1
    ), prompt=prompt, verbose=True)

    story = story_llm.predict(scenario=scenario)

    print(story)
    return story

scenario = img2text("license.jpg")
story = generate_story(scenario)

#text to speech

def text2speech(message):
    API_URL = ""
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payloads) 
    with open('audio.flac', 'wb') as file:
        file.write(response.content)
