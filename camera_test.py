"""Small utility to discover usable camera indices before running the main app."""
import cv2

from camera import CameraSource


def main():
    print("Checking camera indices 0..5")
    found = []
    for i in range(6):
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if not (ok and frame is not None):
            print(f"Camera {i}: unavailable")
            continue
        h, w = frame.shape[:2]
        found.append(i)
        source = CameraSource(i)
        source.open()  # prints the mode the app will actually capture
        source.release()
        print(f"Camera {i}: OK (default {w}x{h})")
    print("\nUsable indices:", found)
    print("Set these as Front/Side in the app or config.py.")


if __name__ == "__main__":
    main()
