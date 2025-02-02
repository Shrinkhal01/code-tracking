from src.backup_manager import BackupManager

def main():
    CONFIG_PATH = "config/config.yml"
    CREDENTIALS_PATH = "credentials/credentials.json"

    manager = BackupManager(CONFIG_PATH, CREDENTIALS_PATH)
    backup_path = manager.create_encrypted_backup()
    if backup_path:
        manager.upload_backup(backup_path)
        manager.cleanup_old_backups()

if __name__ == "__main__":
    main()
