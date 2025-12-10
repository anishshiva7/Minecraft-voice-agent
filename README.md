Cloned from https://github.com/mindcraft-bots/mindcraft


To run the MindCraft server:
``` node main.js```

Ollama models required - Written in andy.json
```
ollama/sweaterdog/andy-4:micro-q8_0

ollama/embeddinggemma
```

More agents can be added by creating their json files and adding it to settings.js
The API can also be used to add more agents.


All voice recogntion code is in the voice_recogntion/ folder. The push_to_talk.py is the main code that will be run to send commands to the agent. This script contains both voice recognition and speaker diarization/identification models.
Install its requirements using:

``` pip install -r requirements.txt```


The push to talk script might need sudo access:

```sudo python3 push_to_talk.py```


The example code for creating a python client to connect to the Mindcraft server using websocket is in examples/python_client/python_client.py. This will be used to communicate commands to the LLM.