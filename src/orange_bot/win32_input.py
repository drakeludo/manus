import ctypes
import time
from ctypes import wintypes


INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

SCAN_CODES = {
    "w": 0x11,
    "a": 0x1E,
    "s": 0x1F,
    "d": 0x20,
    "e": 0x12,
}


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    )


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    )


class INPUT_UNION(ctypes.Union):
    _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT))


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = (("type", wintypes.DWORD), ("u", INPUT_UNION))


user32 = ctypes.WinDLL("user32", use_last_error=True)


def send_input(*inputs: INPUT) -> None:
    input_array = (INPUT * len(inputs))(*inputs)
    user32.SendInput(len(inputs), input_array, ctypes.sizeof(INPUT))


def key_press(key: str, duration: float = 0.05) -> None:
    key_down(key)
    time.sleep(duration)
    key_up(key)


def key_down(key: str) -> None:
    scan = SCAN_CODES.get(key.lower())
    if scan is None:
        return
    send_input(
        INPUT(
            type=INPUT_KEYBOARD,
            ki=KEYBDINPUT(wVk=0, wScan=scan, dwFlags=KEYEVENTF_SCANCODE),
        )
    )


def key_up(key: str) -> None:
    scan = SCAN_CODES.get(key.lower())
    if scan is None:
        return
    send_input(
        INPUT(
            type=INPUT_KEYBOARD,
            ki=KEYBDINPUT(
                wVk=0,
                wScan=scan,
                dwFlags=KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP,
            ),
        )
    )


def click_at(x: int, y: int, settle_delay: float = 0.0, press_delay: float = 0.0) -> None:
    user32.SetCursorPos(x, y)
    if settle_delay > 0:
        time.sleep(settle_delay)
    send_input(INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, None)))
    if press_delay > 0:
        time.sleep(press_delay)
    send_input(INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, None)))


def click_many(
    points: list[tuple[int, int]],
    settle_delay: float = 0.0,
    press_delay: float = 0.0,
) -> None:
    for x, y in points:
        click_at(x, y, settle_delay=settle_delay, press_delay=press_delay)
