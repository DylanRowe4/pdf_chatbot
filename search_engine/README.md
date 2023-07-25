# search_engine
This example uses langchain and streamlit to create an app that allows you to locally save a FAISS document store, then utilize the vector database for semantic document retrieval and searching. This method is great for large text databases where there is a longer time requirement for text preprocessing, creating the embeddings, and storing the index should be done prior to query time. This allows for quick searching and retrieval by the end user. The app currently uses HuggingFace's flan-t5-large model and searches through the document store using maximal marginal relevence to provide diverse and relevent chunks of the text that are provided to the model as context at query time.

In order to use the code you will need to sign up for an account at <a href="https://huggingface.co/">https://huggingface.co/</a> and create a pair of API tokens. Once created, you can save them in a json file and set it to your environment with os.environ['HUGGINGFACEHUB_API_TOKEN']. Feel free to take a look and have some fun with the code!

<img width="956" alt="image" src="https://github.com/DylanRowe4/langchain_examples/assets/43864012/db5af724-0cc2-4a53-b2a7-8cd5dd5368b1">

