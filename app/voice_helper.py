import speech_recognition as sr
from gtts import gTTS

def speech_to_text():

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text

    except:
        return None

def text_to_speech(text, filename="response.mp3"):

    tts = gTTS(text=text, lang='en')

    tts.save(filename)

    return filename