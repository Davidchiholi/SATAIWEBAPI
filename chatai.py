import os
import json
from openai import AzureOpenAI
from datetime import datetime
from zoneinfo import ZoneInfo

f = open("aikey.txt", "r")
aikey = f.read()

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint = ("https://satmappopenai.openai.azure.com/"), 
    api_key=(aikey),  
    api_version="2024-05-01-preview"
)

    
deployment_name='satgpt4o' #This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment. 
    
# Send a completion call to generate an answer
print('Sending a test completion job')

f = open("Json.txt", "r")
filecontent = f.read()
#print(filecontent)

result_dict = json.loads(filecontent)

print(len(result_dict["matched_list"]))

matchlist = result_dict["matched_list"]

joint1_length = result_dict["joint_distiance1"]

joint1_avg_speed = result_dict["joint1_avg_speed"]

question = f"The user and trainer sport's movement matching percentage trend list : {matchlist}. Question is you need to describe this action matching trend. Length in meter between left hip and left knee is {joint1_length}. Describe this length compare to normal asian human. Average speed on Left knee is {joint1_avg_speed} meter/second. Describe this speed."

print(question)


response = client.chat.completions.create(
    model="satgpt4o", # model = "deployment_name".
    messages=[
        {"role": "system", "content": "Use Chinese to reply. The user input JSON format data for analysis."},
        {"role": "user", "content": question}
    ]
)

print(response.choices[0].message.content)