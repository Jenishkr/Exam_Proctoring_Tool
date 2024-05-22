import speech_recognition as sr

f = open("audio.txt", "w")

recognizer = sr.Recognizer()


def get_audio_from_microphone():
    with sr.Microphone() as source:
        print("Speak something...")
        recognizer.adjust_for_ambient_noise(source)  # Optional: Adjust for ambient noise
        audio_data = recognizer.listen(source)

    return audio_data


def convert_audio_to_text(audio_data):
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        print("Google Web Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech Recognition service; {e}")

    return None


if __name__ == "__main__":
    audio_data = get_audio_from_microphone()
    if audio_data:
        recognized_text = convert_audio_to_text(audio_data)
        if recognized_text:
            print(f"Recognized Text: {recognized_text}")
            f.writelines(recognized_text)
f.close()