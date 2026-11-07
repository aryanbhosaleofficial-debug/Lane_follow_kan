"""Run: python -m camera_input.demo --source 0, or --source videos/test.mp4 --file."""
import argparse
import time
import cv2
from .core import CameraConfig, CameraManager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--file', action='store_true')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--switch-to', help='switch once after five seconds; prefix file paths with file:')
    args = parser.parse_args()
    def config(value, is_file=False):
        return CameraConfig(source=int(value) if value.isdecimal() and not is_file else value, kind='file' if is_file else 'live', pace_video=not args.fast)
    manager = CameraManager(config(args.source, args.file))
    manager.start(); started = time.monotonic(); switched = False
    try:
        while True:
            if args.switch_to and not switched and time.monotonic() - started >= 5:
                target = args.switch_to
                manager.switch_source(config(target[5:] if target.startswith('file:') else target, target.startswith('file:')))
                switched = True
            packet = manager.read()
            if packet is not None:
                cv2.imshow('M01', packet.image)
            if cv2.waitKey(1) & 255 == ord('q'): break
            if manager.status()['state'] in ('eof', 'error'): break
            time.sleep(.001)
    finally:
        manager.stop(); cv2.destroyAllWindows()
        print(manager.status())

if __name__ == '__main__': main()
