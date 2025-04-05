import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API key from .env
API_KEY = os.getenv("AIzaSyBrWW2fX7pLYdNLsFhSOhmzAmmH1ZGMHyw")

# OAuth 2.0 credentials
SERVICE_ACCOUNT_FILE = 'credentials/credentials.json'
SCOPES = ['https://www.googleapis.com/auth/drive']
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# Build services
api_key_service = build('drive', 'v3', developerKey=API_KEY)
oauth_service = build('drive', 'v3', credentials=credentials)

class GDriveStorage:
    def __init__(self):
        self.api_key_service = api_key_service
        self.oauth_service = oauth_service

    def list_files(self, folder_id=None):
        """List files using API Key"""
        try:
            query = f"'{folder_id}' in parents" if folder_id else None
            results = self.api_key_service.files().list(q=query, fields="files(id, name)").execute()
            files = results.get('files', [])
            return files
        except Exception as e:
            print(f"Error listing files: {e}")
            return []

    def upload_file(self, file_path, folder_id=None):
        """Upload file using OAuth"""
        try:
            file_metadata = {"name": os.path.basename(file_path)}
            if folder_id:
                file_metadata["parents"] = [folder_id]
            media = MediaFileUpload(file_path, resumable=True)
            file = self.oauth_service.files().create(body=file_metadata, media_body=media, fields="id").execute()
            print(f"Uploaded {file_path} as {file.get('id')}")
        except Exception as e:
            print(f"Error uploading file: {e}")
