import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
import os

load_dotenv()

scopes = ["https://www.googleapis.com/auth/spreadsheets", 
          "https://www.googleapis.com/auth/drive"]

creds = Credentials.from_service_account_file(
    os.getenv("GOOGLE_CREDENTIALS_FILE"), 
    scopes=scopes
)

client = gspread.authorize(creds)
sheet = client.open(os.getenv("GOOGLE_SHEET_NAME")).sheet1

def get_sheet():
    """Returns the connected Google Sheet object"""
    return sheet