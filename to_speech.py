import os
from gtts import gTTS

tts = gTTS(text = 'blablabla', lang = 'en')
tts.save("to_speech.wav")

