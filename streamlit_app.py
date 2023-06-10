from PIL import Image
import streamlit as st
from ultralytics import YOLO
import openai
from collections import defaultdict
import pyttsx3

engine = pyttsx3.init()
openai.api_key = ' < YOUR API KEY HERE> '

def chat_with_gpt(prompt):
    # Define the parameters for the API call
    parameters = {
        'model': 'gpt-3.5-turbo',
        'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': prompt}]
    }

    # Call the OpenAI API
    response = openai.ChatCompletion.create(**parameters)

    # Extract the model's reply from the API response
    reply = response.choices[0].message.content

    return reply

def count_items(input_list):
    result = defaultdict(int)
    for item in input_list:
        result[item] += 1
    return dict(result)

def main():
    st.title("AI Assistant")
    st.write("Take a pic and upload here: ")
    #run = st.button('Run', key='button1')
    #capture = st.button('Capture', key='button2')

    img = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if img is not None:
        image = Image.open(img)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        print(type(image))
        image.save('curimage.png')
        
        list1=[]
        model = YOLO("yolov8s.pt")
        results=model.predict(source='curimage.png',show=False,save_txt=False) 
        #st.image('cap.jpg', caption='Image captured doing Object detection')

        for result in results:
            cls = result.boxes.cls
            for c in cls:
                #print(result.names[int(c)],end=" ")
                list1.append(result.names[int(c)])


        list2 = count_items(list1)
        print(list2)
        st.write("The Objects detected are: ", list2)
        #for objectname in list2:
            #engine.say(objectname)
            #engine.runAndWait()
        
        user_prompt = "Given list of objects, guess the place or describe the environment. " + str(list2)
        response = chat_with_gpt(user_prompt)
        print("AI Assistant:", response)
        st.write("AI Assistant:", response)
        #engine.say(response)
        #engine.runAndWait()
        
        user_input = st.text_input("Ask me anything: ") + "Given list of objects are: " + str(list2)
        response = chat_with_gpt(user_prompt)
        print("AI Assistant:", response)
        st.write("AI Assistant:", response)
        #engine.say(response)
        #engine.runAndWait()
        


if __name__ == '__main__':
    main()
