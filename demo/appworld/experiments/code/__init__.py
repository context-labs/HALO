import os

import litellm
from dotenv import load_dotenv


load_dotenv()
os.environ["NO_API_KEY"] = "__empty__"
os.environ["LITELLM_LOG"] = "ERROR"
litellm.drop_params = False  # Keep it False to avoid silent errors
litellm.turn_off_message_logging = True
litellm.suppress_debug_info = True
