from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from contractions import fix  # You may need to install the 'contractions' library: pip install contractions

# Create a new chatbot
chatbot = ChatBot('MyBot')

# Create a new trainer for the chatbot
trainer = ChatterBotCorpusTrainer(chatbot)

# Function for text preprocessing
def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Expand contractions
    text = fix(text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove punctuation
    tokens = [word.translate(str.maketrans("", "", string.punctuation)) for word in tokens]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Join tokens back into a sentence
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Train the chatbot on custom data
custom_data = [
    ('How are you?', 'I am good, thank you!'),
    ('What is your name?', 'My name is MyBot.')
]

# Preprocess custom data
preprocessed_custom_data = [(preprocess_text(question), preprocess_text(answer)) for question, answer in custom_data]

# Train the chatbot on custom data
trainer.train(preprocessed_custom_data)

# Start the conversation loop
while True:
    user_input = input('You: ')

    if user_input.lower() == 'exit':
        break  # Exit the loop if the user types 'exit'

    preprocessed_input = preprocess_text(user_input)
    response = chatbot.get_response(preprocessed_input)
    print('Bot:', response)