import os
import sys
from ui.app_controller import AppController
from utils.logger import get_logger

# Ensure the root directory of the project is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger("main")

def init_workspace():
    """Create all required data and logging folders."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    folders = [
        os.path.join(app_dir, "data", "db"),
        os.path.join(app_dir, "data", "exports"),
        os.path.join(app_dir, "models"),
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        logger.info(f"Directory verified: {folder}")

    # Create model files stubs README if missing
    readme_path = os.path.join(app_dir, "models", "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# Models Directory\n\nPlace any ONNX model files or deep learning weights here.\n")

    gitkeep_path = os.path.join(app_dir, "models", ".gitkeep")
    if not os.path.exists(gitkeep_path):
        with open(gitkeep_path, "w") as f:
            pass

def main():
    logger.info("Initializing DocuScan India application...")
    try:
        init_workspace()
    except Exception as e:
        print(f"Failed to initialize workspace folders: {e}")
        sys.exit(1)

    try:
        logger.info("Launching Tkinter desktop interface...")
        app = AppController()
        app.mainloop()
        logger.info("Application closed normally.")
    except Exception as e:
        logger.critical(f"Unhandled critical crash in main loop: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
