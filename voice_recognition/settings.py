"""
Settings for the Python example client. Mirrors the values from the project's `settings.js`.
Edit or override with environment variables as needed.
"""
import os
import json

# Mirror of settings.js values
SETTINGS = {
    "minecraft_version": "auto",
    "host": "10.8.203.165",
    "port": 55916,
    "auth": "offline",

    # MindServer settings
    "mindserver_port": 8080,
    "mindserver_host": "localhost",
    "auto_open_ui": True,

    "base_profile": "assistant",
    "profiles": [
        "./andy.json",
    ],

    "load_memory": False,
    "init_message": "Respond with hello world and your name",
    "only_chat_with": [],

    "speak": False,
    "chat_ingame": True,
    "language": "en",
    "render_bot_view": False,

    "allow_insecure_coding": False,
    "allow_vision": False,
    "blocked_actions": ["!checkBlueprint", "!checkBlueprintLevel", "!getBlueprint", "!getBlueprintLevel"],
    "code_timeout_mins": -1,
    "relevant_docs_count": 5,

    "max_messages": 15,
    "num_examples": 2,
    "max_commands": -1,
    "show_command_syntax": "full",
    "narrate_behavior": True,
    "chat_bot_messages": True,

    "spawn_timeout": 30,
    "block_place_delay": 0,

    "log_all_prompts": False,
}

# Allow overrides via environment variables
if os.getenv('MINDSERVER_PORT'):
    try:
        SETTINGS['mindserver_port'] = int(os.getenv('MINDSERVER_PORT'))
    except ValueError:
        pass

if os.getenv('MINDSERVER_HOST'):
    SETTINGS['mindserver_host'] = os.getenv('MINDSERVER_HOST')

if os.getenv('MINECRAFT_PORT'):
    try:
        SETTINGS['port'] = int(os.getenv('MINECRAFT_PORT'))
    except ValueError:
        pass

if os.getenv('PROFILES'):
    try:
        p = json.loads(os.getenv('PROFILES'))
        if isinstance(p, list) and len(p) > 0:
            SETTINGS['profiles'] = p
    except Exception:
        pass

# Export convenience variables
MINECRAFT_HOST = SETTINGS['host']
MINECRAFT_PORT = SETTINGS['port']
MINDSERVER_HOST = SETTINGS['mindserver_host']
MINDSERVER_PORT = SETTINGS['mindserver_port']
DEFAULT_PROFILE_PATHS = SETTINGS['profiles']
DEFAULT_INIT_MESSAGE = SETTINGS['init_message']
