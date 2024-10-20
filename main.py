import os

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer, ChatterBotCorpusTrainer
import pandas as pd
import chatDatasets
from CustomTrainer import CustomTrainer


def loadQaPairs(filename="data/qaPairs.csv"):
    if not os.path.exists(filename):
        chatDatasets.generateQaPairs()
    df = pd.read_csv(filename)
    pairs = []

    for index, row in df.iterrows():
        pairs.append(row['Question'])
        pairs.append(row['Answer'])
    return pairs

def loadQaData(filename="data/qaPairs.csv"):
    return pd.read_csv(filename)

def runChatbot():
    chatbot = ChatBot(
        'MyChatterBot',
        logic_adapters=[
                {
                    'import_path': 'chatterbot.logic.BestMatch',
                    'threshold': 0.90,
                    'default_response': 'I am sorry, but I do not understand.'
                }
        ],
        storage_adapter='chatterbot.storage.SQLStorageAdapter',
        database_uri='sqlite:///db.sqlite3'
    )

    data = loadQaData()
    pairs = loadQaPairs()

    corpusTrainer = ChatterBotCorpusTrainer(chatbot)
    corpusTrainer.train('chatterbot.corpus.english')
    customTrainer = CustomTrainer(chatbot)
    # trainer = ListTrainer(chatbot)


    for i in range(3):
        customTrainer.train(data)
        # trainer.train(pairs)

    print("Hello, what can I help with today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        response = chatbot.get_response(user_input)
        print(f"ChatBot: {response}")



runChatbot()
