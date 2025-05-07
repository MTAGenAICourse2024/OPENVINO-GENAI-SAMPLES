#!/usr/bin/env python3
'''
Launch script for STT app
'''
import os
# Set environment variables to prevent PyTorch/Streamlit issues
os.environ["STREAMLIT_SERVER_WATCH_MODULES"] = "false"

import streamlit.web.cli as stcli
import sys

if __name__ == "__main__":
    # Run the Streamlit app
    sys.argv = ["streamlit", "run", "04_microphone_to_text.py"]
    sys.exit(stcli.main())
