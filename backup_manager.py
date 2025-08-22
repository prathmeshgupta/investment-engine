"""Simple backup and version control system for the investment engine."""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path
import hashlib

class BackupManager:
    """Manages backups and versioning without Git."""
    
    def __init__(self):
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
        self.manifest_file = self.backup_dir / "manifest.json"
        self.load_manifest()
        
    def load_manifest(self):
        """Load or create backup manifest."""
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {"versions": [], "current": None}
            self.save_manifest()
    
    def save_manifest(self):
        """Save backup manifest."""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def create_backup(self, description="Manual backup"):
        """Create a new backup of critical files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        # Files and directories to backup
        critical_items = [
            "dashboard/",
            "config/",
            "auto_config.py",
            "launcher.py",
            "main.py",
            "analytics/",
            "backtesting/",
            "execution/",
            "optimization/",
            "risk/",
            "factors/",
            "core/"
        ]
        
        backed_up = []
        for item in critical_items:
            src = Path(item)
            if src.exists():
                dst = backup_path / item
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                backed_up.append(item)
        
        # Create version info
        version_info = {
            "name": backup_name,
            "timestamp": timestamp,
            "description": description,
            "files": backed_up,
            "hash": self.calculate_hash(backup_path)
        }
        
        self.manifest["versions"].append(version_info)
        self.manifest["current"] = backup_name
        self.save_manifest()
        
        print(f"[OK] Backup created: {backup_name}")
        print(f"    Files backed up: {len(backed_up)}")
        return backup_name
    
    def calculate_hash(self, path):
        """Calculate hash of backup for integrity."""
        hasher = hashlib.sha256()
        for root, dirs, files in os.walk(path):
            for file in sorted(files):
                file_path = Path(root) / file
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        hasher.update(f.read())
        return hasher.hexdigest()[:16]
    
    def restore_backup(self, version_name=None):
        """Restore from a backup."""
        if not version_name:
            version_name = self.manifest.get("current")
        
        if not version_name:
            print("[ERROR] No backup version specified")
            return False
        
        backup_path = self.backup_dir / version_name
        if not backup_path.exists():
            print(f"[ERROR] Backup {version_name} not found")
            return False
        
        # Restore files
        for item in os.listdir(backup_path):
            src = backup_path / item
            dst = Path(item)
            
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        
        print(f"[OK] Restored from backup: {version_name}")
        return True
    
    def list_backups(self):
        """List all available backups."""
        print("\n[*] Available Backups:")
        print("-" * 50)
        for version in self.manifest["versions"]:
            current = " (CURRENT)" if version["name"] == self.manifest["current"] else ""
            print(f"  - {version['name']}{current}")
            print(f"    Created: {version['timestamp']}")
            print(f"    Description: {version['description']}")
            print(f"    Hash: {version['hash']}")
        print("-" * 50)

if __name__ == "__main__":
    backup = BackupManager()
    backup.create_backup("Initial backup before restoration")
    backup.list_backups()
