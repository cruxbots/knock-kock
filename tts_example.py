from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# download and load all models
preload_models(text_use_small=True,
               coarse_use_small=True,
               fine_use_gpu=False,
               fine_use_small=True)

# generate audio from text

text_prompt = """
     कौन है भाई?! और कहाँ जाना है?!
"""
# text_prompt = """मादरचोद !"""
audio_array = generate_audio(text_prompt, 
                         # text_temp=0.5,
                         # waveform_temp=0.4,
                         history_prompt="v2/en_speaker_6"
                         )

# save audio to disk
write_wav("bark_generation_2.wav", SAMPLE_RATE, audio_array)