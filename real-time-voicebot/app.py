import assemblyai as aai
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class AI_Assistant:
    def __init__(self):
        aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]
        self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.elevenlabs_api_key = os.environ["ELEVENLABS_API_KEY"]

        self.elevenlabs_client = ElevenLabs(api_key=self.elevenlabs_api_key)

        self.transcriber = None

        self.interaction = [
            {"role": "system",
             "content": "You are a friendly and professional Real-Time Event Planner AI. \
                Your role is to help users plan, organize, and manage events such as weddings, parties, or corporate gatherings. \
                You offer creative suggestions, budget-friendly options, and efficient planning solutions. \
                Be polite, concise, and resourceful in your responses."
             },
        ]
    
    def stop_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None
    
    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        print("Session ID:", session_opened.session_id)
        return
    
    def on_error(self, error: aai.RealtimeError):
        print("An error occured:", error)
        return
    
    def on_close(self):
        print("Closing Session")
        return
    
    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return
        if isinstance(transcript, aai.RealtimeTranscript):
            self.generate_ai_response(transcript)
        else:
            print(transcript.text, end="\r")
    
    def start_transcription(self):
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data = self.on_data,
            on_error=self.on_error,
            on_open=self.on_open,
            on_close=self.on_close,
            end_utterance_silence_threshold=1000
        )

        self.transcriber.connect()
        microphone_system = aai.extras.MicrophoneStream(sample_rate=16000)
        self.transcriber.stream(microphone_system)

    def generate_ai_response(self, transcript):

        self.stop_transcription()

        self.interaction.append(
            {
                "role": "assistant",
                "content": transcript.text
            }
        )
        print(f"\nPerson: {transcript.text}", end="\r\n")

        response=self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.interaction
        )

        ai_response = response.choices[0].message.content

        self.generate_audio(ai_response)

        self.start_transcription()
        print(f"\nReal-time transacription: ", end="\r\n")

    def generate_audio(self, text):

        self.interaction.append(
            {
            "role": "assistant",
            "content": text
            }
        )
        print(f"\nAI Strategist: {text}")

        audio_stream = self.elevenlabs_client.generate(
            text=text,
            voice="Charlotte",
            stream=True
        )

        stream(audio_stream)

greetings = "Hello! I'm your personal Event Planner AI, ready to help you create unforgettable experiences. \
             Let’s start planning your perfect event!"
ai_assistant = AI_Assistant()
ai_assistant.generate_audio(greetings)
ai_assistant.start_transcription()
