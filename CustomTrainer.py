from chatterbot.trainers import Trainer
from chatterbot.conversation import Statement

class CustomTrainer(Trainer):
    def train(self, data):
        for index, row in data.iterrows():
            question = Statement(text=row['Question'])
            answer = Statement(text=row['Answer'])
            self.chatbot.learn_response(answer, question)
