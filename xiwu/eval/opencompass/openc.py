

from opencompass.cli.main import main

import os, sys
os.environ['OPENAI_API_KEY'] = "EMPTY"

os.environ["https_proxy"] = "http://192.168.32.148:8118"


main()

