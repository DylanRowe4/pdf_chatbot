# pdf_chatbot
Using langchain and streamlit to create an app that allows you to upload a pdf, create a vector document store, and then chat with a bot that will answer your questions about the pdf you uploaded. The app currently uses HuggingFace's flan-t5-large model and searches through the document store using maximal marginal relevence to provide diverse and relevent chunks of the text that are provided to the model as context at query time.

<img width="958" alt="image" src="https://github.com/DylanRowe4/pdf_chatbot/assets/43864012/314efa19-db18-4f81-a375-3f73221951f7">
