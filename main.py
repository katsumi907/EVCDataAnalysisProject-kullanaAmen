import joblib
import speech_recognition as sr
from IPython.display import Audio, display
from googletrans import Translator
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import string


nltk.download('stopwords')
nltk.download('punkt')

# Load your trained model and vectorizer
model = joblib.load('C:/Users/tarek/PycharmProjects/EVCDataAnalysisProject-kullanaAmen/kullanaAmn1.pkl')
vectorizer = joblib.load('C:/Users/tarek/PycharmProjects/EVCDataAnalysisProject-kullanaAmen/vectorizer_filename.pkl')

# Create a speech recognition object
recognizer = sr.Recognizer()

# Create a translation object
translator = Translator()

# Record audio from the user
print("Start speaking")
with sr.Microphone() as source:
    audio = recognizer.listen(source, timeout=15.0)
    print("Recording finished")


def clean_text(text):

    # remove punct
    punctuation_re = re.compile('[%s]' % re.escape(string.punctuation))
    no_punc = punctuation_re.sub('', text)

    # convert to lowercase
    lower_text = no_punc.lower()

    # remove numbers
    number_re = re.compile(r'\d+')
    no_numbers = number_re.sub('', lower_text)

    # tokenize
    tokens = nltk.word_tokenize(no_numbers)

    # stopwords
    stop_words = stopwords.words('english')
    no_stop = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(token) for token in no_stop]

    return ' '.join(stemmed)

# Convert the audio to text using Google's Speech Recognition service
try:
    text = recognizer.recognize_google(audio, language='ar')
    print("Recorded text:", text)

    # Translate the text to English
    translated_text = translator.translate(text, src='ar', dest='en').text
    print("Text translated to English:", translated_text)

    # Preprocess and vectorize the translated text
    preprocessed_text = clean_text(translated_text)
    vectorized_text = vectorizer.transform([preprocessed_text])

    prediction = model.predict(vectorized_text)
    # Convert the prediction to the corresponding class label
    if prediction[0] == 0:
        p = "car accidents"
    elif prediction[0] == 1:
        p = "crime"
    elif prediction[0] == 2:
        p = "fire"
    else:
        p = "robbery"

    # Use the trained model to predict the class of the report
    prediction = model.predict(vectorized_text)
    print("Predicted class:", p)

except sr.UnknownValueError:
    print("Could not understand the audio")
except sr.RequestError as e:
    print("Error connecting to the Speech Recognition service; {0}".format(e))

# Play the recorded audio
display(Audio(audio.get_wav_data(), autoplay=True))
