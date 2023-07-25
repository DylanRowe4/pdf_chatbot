# qa_chatbot
This example uses langchain and streamlit to create an app that allows you to upload a pdf and save it in a local temp directory, create a vector document store, and then chat with a bot that will answer your questions about the pdf you uploaded. The app currently uses HuggingFace's flan-t5-large model and searches through the document store using maximal marginal relevence to provide diverse and relevent chunks of the text that are provided to the model as context at query time. In addition, we used some basic HTML to make the app more fun and provide formatting.

In order to use the code you will need to sign up for an account at <a href="https://huggingface.co/">https://huggingface.co/</a> and create a pair of API tokens. Once created, you can save them in a json file and set it to your environment with os.environ['HUGGINGFACEHUB_API_TOKEN']. Feel free to take a look and have some fun with the code!

<img width="958" alt="image" src="https://github.com/DylanRowe4/pdf_chatbot/assets/43864012/314efa19-db18-4f81-a375-3f73221951f7">
