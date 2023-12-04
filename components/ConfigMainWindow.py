import os
import sys
import json

from PySide6.QtCore import Signal, Slot, QPoint
from PySide6.QtWidgets import QApplication, QMainWindow


class ConfigMainWindow(QMainWindow):
    def __init__(self, config_path="config.json"):
        super().__init__()

        # 配置文件
        if os.path.isabs(config_path):
            self.config_path = config_path
        else:
            self.config_path = os.path.join(os.getcwd(), "config.json")

    def getConfig(self, key=None):
        config = None
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if key:
            try:
                value = config[key]
            except KeyError:
                raise KeyError(f"key: {key} not found in config.json")
            return value
        return config
    
    def setConfig(self, config):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

    def addConfig(self, key, value):
        config = self.getConfig()
        config[key] = value
        self.setConfig(config)
    
    def delConfig(self, key):
        config = self.getConfig()
        del config[key]
        self.setConfig(config)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ConfigMainWindow()
    window.show()
    sys.exit(app.exec())
