import mediapipe  # noqa: F401  must load before PyQt6, otherwise its DLL fails to initialize on some Windows setups

from gui_modern import run

if __name__ == "__main__":
    run()
