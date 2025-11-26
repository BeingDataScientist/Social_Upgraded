"""
Configuration file for OpenAI API settings

To set up your OpenAI API key:
1. Get your API key from https://platform.openai.com/api-keys
2. Set it as an environment variable:
   - Windows (PowerShell): $env:OPENAI_API_KEY="your-key-here"
   - Linux/Mac: export OPENAI_API_KEY="your-key-here"
3. Or directly set it below (not recommended for production):
   OPENAI_API_KEY = 'your-key-here'

Alternatively, you can use python-dotenv:
   pip install python-dotenv
   Create a .env file with: OPENAI_API_KEY=your-key-here
   Then uncomment the dotenv lines below.
"""
import os

# OpenAI API Configuration
OPENAI_API_KEY = 'sk-proj-1234567890'
#OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')  # Default to gpt-4o-mini, can be changed to gpt-4, gpt-3.5-turbo, etc.

# If you want to load from a .env file, uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

