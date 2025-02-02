import os
import pyzipper
from datetime import datetime, timedelta
import yaml
from dotenv import load_dotenv
from src.logger import setup_logger
from src.gdrive_storage import GDriveStorage

class BackupManager:
    def __init__(self):
        self.gdrive = GDriveStorage()

    def verify_existing_backups(self):
        """Check existing backups using API Key"""
        files = self.gdrive.list_files()
        if files:
            print("Existing backups found:")
            for file in files:
                print(f"- {file['name']}")
        else:
            print("No existing backups found.")

class BackupManager:
    def __init__(self, config_path, credentials_path):
        load_dotenv()
        self.logger = setup_logger()
        self.encryption_password = os.getenv('ENCRYPTION_PASSWORD').encode()

        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.gdrive = GDriveStorage(credentials_path)
        self.source_directory = self.config['backup']['source_directory']
        self.encrypted_backup_directory = self.config['backup']['encrypted_backup_directory']
        self.retention_days = self.config['backup']['retention_days']

        os.makedirs(self.encrypted_backup_directory, exist_ok=True)

    def create_encrypted_backup(self):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_filename = f"backup_{timestamp}.zip"
        backup_path = os.path.join(self.encrypted_backup_directory, backup_filename)

        try:
            with pyzipper.AESZipFile(backup_path, 'w', compression=pyzipper.ZIP_LZMA) as zf:
                zf.setpassword(self.encryption_password)
                zf.setencryption(pyzipper.WZ_AES, nbits=256)
                for root, _, files in os.walk(self.source_directory):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.source_directory)
                        zf.write(file_path, arcname)

            self.logger.info(f"Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            return None

    def upload_backup(self, backup_path):
        try:
            folder_id = self.gdrive.create_folder(datetime.now().strftime("%Y-%m-%d"))
            file_id = self.gdrive.upload_file(backup_path, folder_id)
            self.logger.info(f"Backup uploaded to Google Drive: {file_id}")
        except Exception as e:
            self.logger.error(f"Error uploading backup: {e}")

    def cleanup_old_backups(self):
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        for file in os.listdir(self.encrypted_backup_directory):
            file_path = os.path.join(self.encrypted_backup_directory, file)
            if os.path.isfile(file_path) and datetime.fromtimestamp(os.path.getmtime(file_path)) < cutoff_date:
                try:
                    os.remove(file_path)
                    self.logger.info(f"Deleted old backup: {file_path}")
                except Exception as e:
                    self.logger.error(f"Error deleting old backup: {e}")
