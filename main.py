import os
import tkinter as tk
from tkinter import scrolledtext
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer, ChatterBotCorpusTrainer
import pandas as pd
import chatDatasets
from CustomTrainer import CustomTrainer


def loadQaPairs(filename="data/qaPairsAug.csv"):
    if not os.path.exists(filename):
        chatDatasets.generateQaPairs()
    df = pd.read_csv(filename)
    pairs = []

    for index, row in df.iterrows():
        pairs.append(str(row['Question']))
        pairs.append(str(row['Answer']))
    return pairs


def loadQaData(filename="data/qaPairsAug.csv"):
    return pd.read_csv(filename)


def runUi():
    def on_entry_click(event):
        if entry.get() == "Enter your response here":
            entry.delete(0, tk.END)
            entry.insert(0, '')

    def on_focusout(event):
        if entry.get() == '':
            entry.insert(0, 'Enter your response here')

    def on_submit():
        user_input = entry.get()
        if user_input.lower() in ['quit', 'exit', 'thank you']:
            chatOut_text.config(state='normal')
            chatOut_text.insert(tk.END, "Goodbye!\n")
            chatOut_text.config(state='disabled')
            root.quit()
        else:
            response = chatbot.get_response(user_input)
            chatOut_text.config(state='normal')
            chatOut_text.insert(tk.END, "You: \n", 'bold')
            chatOut_text.insert(tk.END, f"{user_input}\n")
            chatOut_text.insert(tk.END, "ChatBot: \n", 'bold')
            chatOut_text.insert(tk.END, f"{response}\n")
            chatOut_text.config(state='disabled')
            entry.delete(0, tk.END)

    root = tk.Tk()
    root.title("SIMetrix Help Desk")

    chatOut = "Hello, what can I help with today?\n"
    chatOut_text = scrolledtext.ScrolledText(root, height=15, width=50, wrap='word')
    chatOut_text.tag_configure('bold', font=('TkDefaultFont', 9, 'bold'))
    chatOut_text.insert(tk.END, chatOut)
    chatOut_text.config(state='disabled')
    chatOut_text.pack(pady=10)

    entry = tk.Entry(root, width=50)
    entry.insert(0, "Enter your response here")
    entry.bind('<FocusIn>', on_entry_click)
    entry.bind('<FocusOut>', on_focusout)
    entry.pack(pady=10)

    submit_button = tk.Button(root, text="Submit", command=on_submit)
    submit_button.pack(pady=10)

    root.mainloop()


def runConsole():
    print("Hello, what can I help with today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        response = chatbot.get_response(user_input)
        print(f"ChatBot: {response}")


dbPath = 'db.sqlite3'

if not os.path.exists(dbPath):
    chatbot = ChatBot(
        'MyChatterBot',
        logic_adapters=[
            {
                'import_path': 'chatterbot.logic.BestMatch',
                'threshold': 0.7,
                'default_response': 'I am sorry, but I do not understand.'
            }
        ],
        storage_adapter='chatterbot.storage.SQLStorageAdapter',
        database_uri=f'sqlite:///{dbPath}'
    )

    data = loadQaData()
    pairs = loadQaPairs()
    chatDatasets.extractTextFromJson("data/dev_Q_A.json", "data/dev_Q_A.csv")
    chatDatasets.extractTextFromJson("data/training_Q_A.json", "data/trn_Q_A.csv")
    devPairs = loadQaPairs("data/dev_Q_A.csv")
    trnPairs = loadQaPairs("data/trn_Q_A.csv")

    corpusTrainer = ChatterBotCorpusTrainer(chatbot)
    corpusTrainer.train('chatterbot.corpus.english')

    customTrainer = CustomTrainer(chatbot)
    trainer = ListTrainer(chatbot)
    for _ in range(3):
        customTrainer.train(data)
        trainer.train(pairs)
        trainer.train(devPairs)
        trainer.train(trnPairs)

    print("Model trained and saved.")
else:
    print("Model already exists. Skipping training.")
    chatbot = ChatBot(
        'MyChatterBot',
        logic_adapters=[
            {
                'import_path': 'chatterbot.logic.BestMatch',
                'threshold': 0.7,
                'default_response': 'I am sorry, but I do not understand.'
            }
        ],
        storage_adapter='chatterbot.storage.SQLStorageAdapter',
        database_uri=f'sqlite:///{dbPath}'
    )

runUi()
# runConsole()
