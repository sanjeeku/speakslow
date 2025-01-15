Install dependencies
```
brew install portaudio
pip install pyaudio speechrecognition
pip install vosk pyaudio 
```

Download a Vosk model from https://alphacephei.com/vosk/models

For example, an English model can be downloaded from Vosk Models.
Extract the model folder and note its path.
Update MODEL_PATH in the code below to point to the local Vosk model folder.

Run the code
```
python listener.py
```
