Cloned from https://github.com/mindcraft-bots/mindcraft

Install npm packages for MindCraft server:
```npm install ```

### To connect to your minecraft server hosted on LAN, change the IP address in settings.js and voice_recognition/settings.py

To start a LAN server in Minecraft:
* Start a single player game.
* Once inside the game, press the escape key 'esc'
* Click on 'Open to LAN'
* Enter the port number - 55916(can be any port, has to be changed in the settings files as required)
* Click confirm and it should be visible to other people on the same LAN 


To join the game from a different computer:
* Go to multiplayer from the home screen in your game.
* Click on add server
* The server address would be the ipadress:port for the machine that started the Minecraft world.
* Make sure that you are running the same minecraft version as the host.


### The steps for the agent to join into your LAN server are given below, after changing the settings

To run the MindCraft server:
```node main.js```

Ollama models required - Written in andy.json
```
ollama/sweaterdog/andy-4:micro-q8_0

ollama/embeddinggemma
```

More agents can be added by creating their json files and adding it to settings.js
The API can also be used to add more agents.

Set up Qdrant database for storing speaker embeddings:

```docker pull qdrant/qdrant```

To run the database, use the compose file in qdrant_compose/docker-compose.yml

```docker compose -f qdrant_compose/docker-compose.yml up -d```


Create a ```.env``` file in voice_recognition with the variable ```HUGGING_FACE = "your_token_here"``` for pyannote model download



All voice recogntion code is in the voice_recogntion/ folder. The ```push_to_talk.py``` is the main code that will be run to send commands to the agent. This script contains both voice recognition and speaker diarization/identification models.

Create a virtual environment for python3.13.3 using:

replace python3.13.3 with your python executable of the same version.


```python3.13.3 -m venv .venv```


Install its requirements using:


``` pip install -r requirements.txt```


You might need to make this change in the library function if an error occurs:

```
.venv/lib/python3.13/site-packages/lightning/fabric/utilities/cloud_io.py

at line 76

return torch.load(
            f,
            map_location=map_location,  # type: ignore[arg-type]
            weights_only=False,
        )
```


Register voices in the database by using the register.py in voice_recognition folder. Make sure the name of the wav file is the name of the person/player. Example:


```python3 ./register.py sahitya.wav```



The push to talk script might need sudo access:

```sudo python3 push_to_talk.py```


The example code for creating a python client to connect to the Mindcraft server using websocket is in examples/python_client/python_client.py. This will be used to communicate commands to the LLM.
