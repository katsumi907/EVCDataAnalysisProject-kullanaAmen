import streamlit as st
import sounddevice as sd
import wavio
import speech_recognition as sr
from googletrans import Translator
import base64
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
import string

from nltk.tokenize import word_tokenize

nltk.download('punkt')
from nltk.corpus import stopwords

nltk.download('stopwords')
from nltk.stem import PorterStemmer


def record_voice():
    # Record audio for 5 seconds
    duration = 5  # You can change the recording duration as needed
    sample_rate = 44100  # Adjust the sample rate based on your requirements
    channels = 1  # Set to 1 for monaural recording

    placeholder = st.empty()

    placeholder.write("جارٍ التسجيل... يُرجى التحدث إلى الميكروفون.")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
    sd.wait()

    placeholder.empty()
    # st.write("انتهى التسجيل!")

    # Save the recording to a WAV file
    wav_filename = "record.wav"
    wavio.write(wav_filename, recording, sample_rate, sampwidth=2)

    return wav_filename


def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio, language='ar')
        return text

    except sr.UnknownValueError:
        return "النص غير مفهوم"

    except sr.RequestError as e:
        return f"حدث خطأ في الاتصال بخدمة التعرف على الكلام؛ {e}"


def style():
    with open('background.png', "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    st.markdown(
        f"""
        <style>

        body {{
            direction: rtl;
        }}

        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover;
            background-attachment: fixed;
            text-align: center;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


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


def predictReportType(text, model, vectorizer):
    preprocessed_text = clean_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])

    # Make the prediction
    prediction = model.predict(vectorized_text)
    prediction_proba = model.predict_proba(vectorized_text)

    # Get the probability of the predicted class label
    class_index = prediction[0]

    # Get the probability of the predicted class label
    confidence = prediction_proba[0, class_index]
    reportType = {0: 'حادث سيارة', 1: 'جريمة', 2: 'حريق', 3: 'سطو'}
    return reportType[class_index], confidence


def main():
    # Load your trained model and vectorizer
    model = joblib.load('kullanaAmn1.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    # to translate to arabic
    translator = Translator()

    st.set_page_config(page_title="كلنا آمن", page_icon="\U0001F4DE")

    st.text(" \n")
    st.text(" \n")
    st.text(" \n")

    # center the title
    _, col2, _ = st.columns([0.3, 4, 1])

    with col2:
        st.markdown("<h2 style='text-align: center; color: #10704A;'>النظام الذكي لتصنيف البلاغات</h2>",
                    unsafe_allow_html=True)

    style()

    option = st.radio("أرسل بلاغك عن طريق:", ("تسجيل الصوت", "كتابة النص"))

    if option == "تسجيل الصوت":

        if st.button("بدء التسجيل"):
            wav_filename = record_voice()
            st.audio(wav_filename, format='audio/wav')

            try:
                # Transcribe the recorded audio
                text = transcribe_audio(wav_filename)
                st.write("النص المسجل:", text)

                # Translate the text to English
                translated_text = translator.translate(text, src='ar', dest='en').text
                st.write("النص المترجم إلى الإنجليزية:", translated_text)

                # THE MODEL
                result, cnf = predictReportType(translated_text, model, vectorizer)
                st.write("نوع البلاغ: ", result)

            except Exception as e:
                st.write("حدث خطأ أثناء معالجة الملف:", e)


    elif option == "كتابة النص":
        user_text = st.text_area("اكتب النص هنا:")
        st.write(user_text)

        if st.button("إرسال"):
            # Translate the text to English
            translated_text = translator.translate(user_text, src='ar', dest='en').text
            st.write("النص المترجم إلى الإنجليزية:", translated_text)

            # THE MODEL
            result, cnf = predictReportType(translated_text, model, vectorizer)
            st.write("نوع البلاغ: ", result)


if __name__ == "__main__":
    main()
