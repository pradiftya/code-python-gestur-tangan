import cv2
import numpy as np
import mediapipe as mp
import time
import subprocess
from enum import Enum, auto

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARNING] pygame tidak tersedia. Fitur audio dinonaktifkan.")

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MUSIC_PATH = os.path.join(
    BASE_DIR,
    "FOTO KITA BLUR - SAL PRIADI (mp3cut.net).mp3"
)

MUSIC_PATH_2 = os.path.join(
    BASE_DIR,
    "Hidup_jokowi.mp3"
)
BLUR_KERNEL_SIZE = 25
EDGE_CANNY_LOW = 50
EDGE_CANNY_HIGH = 150
FADE_DURATION = 0.4
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
FONT_SCALE_MAIN = 2.0
FONT_SCALE_STATUS = 0.6


class GestureMode(Enum):
    NORMAL = auto()
    V_SIGN = auto()
    THUMBS_UP = auto()
    FIST = auto()


class AudioManager:
    """Manages audio playback via pygame with state tracking."""

    def __init__(self):
        self._initialized = False
        self._tracks: dict[str, pygame.mixer.Sound] = {}
        self._channels: dict[str, pygame.mixer.Channel | None] = {}

        if not PYGAME_AVAILABLE:
            return

        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._initialized = True
        except Exception as e:
            print(f"[AudioManager] Init failed: {e}")

    def load(self, name: str, path: str):
        """Load an audio file and register it by name."""
        if not self._initialized:
            return
        try:
            self._tracks[name] = pygame.mixer.Sound(path)
            self._channels[name] = None
        except Exception as e:
            print(f"[AudioManager] Failed to load '{path}': {e}")

    def play(self, name: str, loop: bool = True):
        """Play a registered track. No-op if already playing."""
        if not self._initialized or name not in self._tracks:
            return
        if self._channels.get(name) and self._channels[name].get_busy():
            return
        try:
            ch = self._tracks[name].play(-1 if loop else 0)
            self._channels[name] = ch
        except Exception as e:
            print(f"[AudioManager] Playback error '{name}': {e}")

    def stop(self, name: str):
        """Stop a specific track."""
        if not self._initialized:
            return
        ch = self._channels.get(name)
        if ch and ch.get_busy():
            ch.stop()
            self._channels[name] = None

    def stop_all(self):
        """Stop all active tracks."""
        if not self._initialized:
            return
        for name in list(self._channels.keys()):
            self.stop(name)


class GestureDetector:
    """Detects hand gestures from MediaPipe landmarks using finger position analysis."""

    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_PIPS = [6, 10, 14, 18]
    THUMB_TIP = 4
    THUMB_IP = 3
    THUMB_MCP = 2
    WRIST = 0
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    DEBOUNCE_FRAMES = 3

    def __init__(self):
        self._debounce_buffer: list[GestureMode] = []
        self._stable_mode: GestureMode = GestureMode.NORMAL

    @staticmethod
    def _finger_up(lm, tip_idx: int, pip_idx: int) -> bool:
        """Check if fingertip is above its PIP joint (finger extended)."""
        return lm[tip_idx].y < lm[pip_idx].y

    def _is_thumbs_up(self, lm, fingers_up: list[bool]) -> bool:
        """Thumb extended far above index knuckle, all other fingers closed."""
        if any(fingers_up):
            return False

        thumb_tip = lm[self.THUMB_TIP]
        index_mcp = lm[self.INDEX_MCP]
        wrist = lm[self.WRIST]

        hand_height = abs(wrist.y - index_mcp.y)
        if hand_height < 0.01:
            return False

        thumb_above = index_mcp.y - thumb_tip.y
        return thumb_above > hand_height * 0.3

    def _is_fist(self, lm, fingers_up: list[bool]) -> bool:
        """All fingers closed and thumb not protruding above knuckle."""
        if any(fingers_up):
            return False

        thumb_tip = lm[self.THUMB_TIP]
        index_mcp = lm[self.INDEX_MCP]
        wrist = lm[self.WRIST]

        hand_height = abs(wrist.y - index_mcp.y)
        if hand_height < 0.01:
            return False

        thumb_above = index_mcp.y - thumb_tip.y
        return thumb_above <= hand_height * 0.3

    def _debounce(self, raw_mode: GestureMode) -> GestureMode:
        """Require N consecutive identical frames before switching mode."""
        self._debounce_buffer.append(raw_mode)
        if len(self._debounce_buffer) > self.DEBOUNCE_FRAMES:
            self._debounce_buffer.pop(0)

        if (len(self._debounce_buffer) == self.DEBOUNCE_FRAMES
                and all(m == raw_mode for m in self._debounce_buffer)):
            self._stable_mode = raw_mode

        return self._stable_mode

    def detect(self, landmarks) -> GestureMode:
        """Analyze landmarks and return debounced GestureMode."""
        lm = landmarks
        fingers_up = [
            self._finger_up(lm, tip, pip)
            for tip, pip in zip(self.FINGER_TIPS, self.FINGER_PIPS)
        ]
        index_up, middle_up, ring_up, pinky_up = fingers_up

        if index_up and middle_up and not ring_up and not pinky_up:
            return self._debounce(GestureMode.V_SIGN)

        if self._is_thumbs_up(lm, fingers_up):
            return self._debounce(GestureMode.THUMBS_UP)

        if self._is_fist(lm, fingers_up):
            return self._debounce(GestureMode.FIST)

        return self._debounce(GestureMode.NORMAL)


class EffectRenderer:
    """Applies visual effects and text overlays to camera frames."""

    TYPEWRITER_TEXT = "Foto kita blur"
    TYPEWRITER_CHAR_DELAY = 0.12
    TYPEWRITER_FADE_DURATION = 0.08

    def __init__(self):
        self._thumbs_anim_start = 0.0
        self._fist_anim_start = 0.0
        self._vsign_start_time = 0.0

    def reset_vsign_anim(self):
        """Reset V Sign typewriter animation timer."""
        self._vsign_start_time = time.time()

    def _put_text_centered(self, frame, text, font_scale, color,
                           thickness=3, outline_color=(0, 0, 0), outline_thickness=6):
        """Draw centered text with outline for readability."""
        font = cv2.FONT_HERSHEY_DUPLEX
        h, w = frame.shape[:2]
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (w - text_w) // 2
        y = (h + text_h) // 2

        cv2.putText(frame, text, (x, y), font, font_scale,
                    outline_color, outline_thickness, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, font_scale,
                    color, thickness, cv2.LINE_AA)

    def _put_text_typewriter(self, frame, full_text, elapsed, font_scale, color,
                             thickness=3, outline_color=(0, 0, 0), outline_thickness=7):
        """Draw text with typewriter + per-character fade-in effect.
        Position is anchored to full_text width so it doesn't shift as chars appear."""
        font = cv2.FONT_HERSHEY_DUPLEX
        h, w = frame.shape[:2]

        (full_tw, text_h), _ = cv2.getTextSize(full_text, font, font_scale, thickness)
        base_x = (w - full_tw) // 2
        base_y = (h + text_h) // 2

        total_chars = len(full_text)
        char_progress = elapsed / self.TYPEWRITER_CHAR_DELAY
        visible_count = min(int(char_progress) + 1, total_chars)

        solid_text = full_text[:max(0, visible_count - 1)]
        if solid_text:
            cv2.putText(frame, solid_text, (base_x, base_y), font, font_scale,
                        outline_color, outline_thickness, cv2.LINE_AA)
            cv2.putText(frame, solid_text, (base_x, base_y), font, font_scale,
                        color, thickness, cv2.LINE_AA)

        if visible_count <= total_chars:
            fade_char = full_text[visible_count - 1]
            prefix = full_text[:visible_count - 1]
            (prefix_w, _), _ = cv2.getTextSize(prefix, font, font_scale, thickness)
            char_x = base_x + prefix_w

            frac = char_progress - int(char_progress)
            alpha = min(frac / (self.TYPEWRITER_FADE_DURATION / self.TYPEWRITER_CHAR_DELAY), 1.0)
            if visible_count > int(char_progress) + 1:
                alpha = 1.0

            fade_color = tuple(
                int(outline_color[i] + (color[i] - outline_color[i]) * alpha) for i in range(3)
            )
            fade_outline = tuple(int(outline_color[i] * alpha) for i in range(3))

            cv2.putText(frame, fade_char, (char_x, base_y), font, font_scale,
                        fade_outline, outline_thickness, cv2.LINE_AA)
            cv2.putText(frame, fade_char, (char_x, base_y), font, font_scale,
                        fade_color, thickness, cv2.LINE_AA)

    def apply_v_sign(self, frame: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur + typewriter text overlay."""
        k = BLUR_KERNEL_SIZE if BLUR_KERNEL_SIZE % 2 == 1 else BLUR_KERNEL_SIZE + 1
        blurred = cv2.GaussianBlur(frame, (k, k), 0)
        elapsed = time.time() - self._vsign_start_time

        self._put_text_typewriter(
            blurred, self.TYPEWRITER_TEXT, elapsed,
            font_scale=FONT_SCALE_MAIN, color=(255, 255, 255),
            thickness=3, outline_color=(0, 0, 0), outline_thickness=7,
        )
        return blurred

    def apply_thumbs_up(self, frame: np.ndarray, first_frame: bool) -> np.ndarray:
        """Apply colored edge detection + animated 'Mantap!' text."""
        if first_frame:
            self._thumbs_anim_start = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, EDGE_CANNY_LOW, EDGE_CANNY_HIGH)

        edge_colored = np.zeros_like(frame)
        edge_colored[:, :, 0] = edges
        edge_colored[:, :, 1] = edges // 2
        edge_colored[:, :, 2] = edges

        output = cv2.addWeighted(frame, 0.7, edge_colored, 0.8, 0)

        elapsed = time.time() - self._thumbs_anim_start
        if elapsed < FADE_DURATION:
            scale = 0.5 + (FONT_SCALE_MAIN - 0.5) * (elapsed / FADE_DURATION)
        else:
            scale = FONT_SCALE_MAIN

        self._put_text_centered(
            output, "Mantap!", font_scale=scale,
            color=(0, 255, 200), thickness=3,
            outline_color=(0, 0, 0), outline_thickness=7,
        )
        return output

    def apply_fist(self, frame: np.ndarray, first_frame: bool) -> np.ndarray:
        """Apply red overlay + animated 'Hidup Jokowi!!!' text with bounce effect."""
        if first_frame:
            self._fist_anim_start = time.time()

        red_overlay = frame.copy()
        cv2.rectangle(red_overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 180), -1)
        output = cv2.addWeighted(frame, 0.7, red_overlay, 0.3, 0)

        elapsed = time.time() - self._fist_anim_start
        if elapsed < 0.15:
            progress = elapsed / 0.15
            scale = 0.3 + (FONT_SCALE_MAIN + 0.5 - 0.3) * progress
        elif elapsed < 0.3:
            progress = (elapsed - 0.15) / 0.15
            scale = (FONT_SCALE_MAIN + 0.5) - 0.5 * progress
        else:
            scale = FONT_SCALE_MAIN

        self._put_text_centered(
            output, "Hidup Jokowi!!!", font_scale=scale,
            color=(0, 0, 255), thickness=4,
            outline_color=(255, 255, 255), outline_thickness=8,
        )
        return output

    def draw_status(self, frame: np.ndarray, mode: GestureMode):
        """Draw current mode label on top-left corner."""
        status_map = {
            GestureMode.NORMAL:    ("NORMAL MODE",       (180, 180, 180)),
            GestureMode.V_SIGN:    ("V SIGN DETECTED",   (0, 220, 255)),
            GestureMode.THUMBS_UP: ("THUMBS UP DETECTED", (0, 255, 100)),
            GestureMode.FIST:      ("FIST DETECTED",     (0, 0, 255)),
        }
        label, color = status_map[mode]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (340, 36), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, label, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_STATUS,
                    color, 2, cv2.LINE_AA)

    def draw_hint(self, frame: np.ndarray):
        """Draw exit hint on bottom-right corner."""
        h, w = frame.shape[:2]
        hint = "Q / ESC: Keluar"
        (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, hint, (w - tw - 10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)


class HandGestureApp:
    """Main application: orchestrates webcam, MediaPipe detection, state management, and effects."""

    SKIP_CAMERAS = ["droidcam"]

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

        self.gesture_detector = GestureDetector()
        self.renderer = EffectRenderer()
        self.audio = AudioManager()
        self.audio.load("vsign", MUSIC_PATH)
        self.audio.load("fist", MUSIC_PATH_2)

        self.current_mode: GestureMode = GestureMode.NORMAL
        self._thumbs_first_frame = False
        self._fist_first_frame = False

    def _handle_mode_transition(self, new_mode: GestureMode):
        """Handle side effects when switching between gesture modes."""
        if new_mode == self.current_mode:
            return

        leaving = self.current_mode
        entering = new_mode

        if leaving == GestureMode.V_SIGN:
            self.audio.stop("vsign")
        if leaving == GestureMode.FIST:
            self.audio.stop("fist")

        if entering == GestureMode.V_SIGN:
            self.audio.play("vsign")
            self.renderer.reset_vsign_anim()
        if entering == GestureMode.THUMBS_UP:
            self._thumbs_first_frame = True
        if entering == GestureMode.FIST:
            self.audio.play("fist", loop=True)
            self._fist_first_frame = True

        self.current_mode = entering

    @staticmethod
    def _get_camera_names() -> list[str]:
        """Get camera names via PowerShell (order matches DirectShow index)."""
        try:
            cmd = (
                'powershell -Command "Get-PnpDevice -Class Camera -Status OK '
                '| Select-Object -ExpandProperty FriendlyName"'
            )
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, shell=True)
            return [n.strip() for n in result.stdout.strip().splitlines() if n.strip()]
        except Exception:
            return []

    def _find_camera_index(self) -> int:
        """Find the first non-virtual camera index. Falls back to 0."""
        names = self._get_camera_names()
        if names:
            print(f"[INFO] Cameras detected: {names}")
            for idx, name in enumerate(names):
                if not any(skip in name.lower() for skip in self.SKIP_CAMERAS):
                    print(f"[INFO] Using: {name} (index {idx})")
                    return idx
            print("[WARNING] All cameras skipped, falling back to index 0")
        return 0

    def run(self):
        """Start the real-time detection loop."""
        cam_index = 0

        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam. Ensure camera is not in use by another app.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("[INFO] App running. Press Q or ESC to exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            detected_mode = GestureMode.NORMAL

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                detected_mode = self.gesture_detector.detect(hand_landmarks.landmark)

                if detected_mode == GestureMode.NORMAL:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2),
                    )

            self._handle_mode_transition(detected_mode)

            if self.current_mode == GestureMode.V_SIGN:
                frame = self.renderer.apply_v_sign(frame)
            elif self.current_mode == GestureMode.THUMBS_UP:
                frame = self.renderer.apply_thumbs_up(frame, self._thumbs_first_frame)
                self._thumbs_first_frame = False
            elif self.current_mode == GestureMode.FIST:
                frame = self.renderer.apply_fist(frame, self._fist_first_frame)
                self._fist_first_frame = False

            self.renderer.draw_status(frame, self.current_mode)
            self.renderer.draw_hint(frame)

            cv2.imshow("Hand Gesture Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break

        self.audio.stop_all()
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("[INFO] App closed.")


if __name__ == "__main__":
    app = HandGestureApp()
    app.run()
