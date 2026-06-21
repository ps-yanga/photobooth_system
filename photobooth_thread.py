import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
from pathlib import Path
from datetime import datetime
import threading
import time
from typing import Optional, List, Tuple, Callable
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[photobooth] %(message)s"
)
logger = logging.getLogger(__name__)

# Safe resample filter
try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_FILTER = Image.LANCZOS  # type: ignore

# Template Constants (4-photo collage)
TEMPLATE_WIDTH: int = 819
TEMPLATE_HEIGHT: int = 2048
MAX_PHOTOS: int = 4

# Photo Slot Specifications
PHOTO_WIDTH: int = 650
PHOTO_HEIGHT: int = 380
PHOTO_ASPECT_RATIO: float = 16.0 / 9.0  # 1.78

# Camera Configuration
CAMERA_WIDTH: int = 1280
CAMERA_HEIGHT: int = 720
DISPLAY_CAMERA_WIDTH: int = 400
DISPLAY_CAMERA_HEIGHT: int = 225

# Asset Paths
ROOT_PATH: Path = Path.cwd()
ASSETS_FRAMES_PATH: Path = ROOT_PATH / "public" / "assets" / "frames"
ASSETS_STICKERS_PATH: Path = ROOT_PATH / "public" / "assets" / "stickers"
ASSETS_LOGO_PATH: Path = ROOT_PATH / "public" / "logo192.png"
ASSETS_TEMPLATE_PATH: Path = ROOT_PATH / "public" / "template.png"

# Photo slot positions on template
PHOTO_SLOTS: List[Tuple[int, int]] = [
    (84, 59),      # Top photo
    (85, 469),     # 2nd photo
    (86, 879),     # 3rd photo
    (86, 1293),    # Bottom photo
]


class FileError(Exception):
    """Custom exception for file operations."""
    pass


class Config:
    """Configuration handler."""

    def __init__(self) -> None:
        """Initialize configuration."""
        self.settings: dict = {
            "photo_dir": Path.cwd() / "photos",
            "photo_quality": 95
        }

    def get(self, key: str):
        """Get configuration value."""
        return self.settings.get(key)

    def set(self, key: str, value) -> None:
        """Set configuration value."""
        self.settings[key] = value


class FileManager:
    """File operations handler."""

    def __init__(self, config: Config) -> None:
        """
        Initialize file manager.

        Args:
            config: Configuration object
        """
        self.config = config
        self.photo_dir: Path = Path(config.get("photo_dir"))
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary directories."""
        try:
            self.photo_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Directories created: %s", self.photo_dir)
        except OSError as e:
            raise FileError(f"Failed to create directories: {e}")

    def save_photo_strip(
        self,
        canvas_image: Image.Image,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save photo strip.

        Args:
            canvas_image: PIL Image to save
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"collage_{timestamp}.png"

            filepath = self.photo_dir / filename
            canvas_image.save(str(filepath))
            logger.info("Strip saved: %s", filepath)
            return filepath

        except (IOError, OSError, AttributeError) as e:
            raise FileError(f"Failed to save strip: {e}")

    def get_photo_strips(self, limit: int = 20) -> List[Path]:
        """
        Get saved photo strips.

        Args:
            limit: Maximum number of strips to return

        Returns:
            List of Path objects (most recent first)
        """
        try:
            strips = sorted(
                self.photo_dir.glob("collage_*.png"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            return strips[:limit]
        except OSError as e:
            raise FileError(f"Failed to retrieve strips: {e}")

    def delete_strip(self, filepath: Path) -> None:
        """
        Delete a photo strip.

        Args:
            filepath: Path to file to delete
        """
        try:
            if filepath.exists():
                filepath.unlink()
                logger.info("Strip deleted: %s", filepath)
            else:
                raise FileError(f"Strip not found: {filepath}")
        except OSError as e:
            raise FileError(f"Failed to delete strip: {e}")


class CameraThread(threading.Thread):
    """Thread-safe camera capture handler."""

    def __init__(
        self,
        camera_index: int = 0,
        on_frame_callback: Optional[Callable] = None
    ) -> None:
        """
        Initialize camera thread.

        Args:
            camera_index: OpenCV camera device index
            on_frame_callback: Callback function when frame is captured
        """
        super().__init__(daemon=True)
        self.camera_index: int = camera_index
        self.on_frame_callback: Optional[Callable] = on_frame_callback
        self.capture_device: Optional[cv2.VideoCapture] = None
        self.is_running: bool = False

    def run(self) -> None:
        """Main camera capture loop."""
        try:
            self._initialize_camera()
            if not self.capture_device or not self.capture_device.isOpened():
                logger.warning("Camera not available or cannot be opened.")
                return

            self._configure_camera()
            self.is_running = True
            self._capture_loop()
        except (OSError, RuntimeError) as exc:
            logger.error("Camera thread exception: %s", str(exc))
        finally:
            self._cleanup()

    def _initialize_camera(self) -> None:
        """Initialize camera device."""
        self.capture_device = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_ANY
        )

    def _configure_camera(self) -> None:
        """Configure camera resolution."""
        if self.capture_device:
            self.capture_device.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                CAMERA_WIDTH
            )
            self.capture_device.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                CAMERA_HEIGHT
            )

    def _capture_loop(self) -> None:
        """Main capture loop."""
        while self.is_running:
            frame_captured, frame = self.capture_device.read()

            if not frame_captured or frame is None:
                time.sleep(0.03)
                continue

            processed_frame = self._process_frame(frame)
            self._invoke_callback(processed_frame)
            time.sleep(0.03)

    @staticmethod
    def _process_frame(frame: cv2.Mat) -> Image.Image:
        """
        Process raw camera frame.

        Args:
            frame: OpenCV frame

        Returns:
            PIL Image in RGB format
        """
        flipped_frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    def _invoke_callback(self, pil_image: Image.Image) -> None:
        """
        Safely invoke frame callback.

        Args:
            pil_image: PIL Image to pass to callback
        """
        if self.on_frame_callback:
            try:
                self.on_frame_callback(pil_image)
            except (OSError, RuntimeError, ValueError) as exc:
                logger.error("Error in frame callback: %s", str(exc))

    def _cleanup(self) -> None:
        """Clean up camera resources."""
        if self.capture_device:
            try:
                self.capture_device.release()
            except (OSError, RuntimeError) as exc:
                logger.error("Error releasing camera: %s", str(exc))

    def stop(self) -> None:
        """Stop the capture thread."""
        self.is_running = False


class AssetManager:
    """Manages loading and caching of frames and stickers."""

    def __init__(self) -> None:
        """Initialize asset manager."""
        self.frames: List[Path] = []
        self.frame_thumbnails: List[ImageTk.PhotoImage] = []
        self.stickers: List[Path] = []
        self.sticker_thumbnails: List[ImageTk.PhotoImage] = []
        self.logo: Optional[Image.Image] = None
        self.template: Optional[Image.Image] = None

    def load_all_assets(self) -> None:
        """Load logo, frames, stickers, and template."""
        self._load_logo()
        self._load_template()
        self._load_frames()
        self._load_stickers()

    def _load_logo(self) -> None:
        """Load and cache application logo."""
        try:
            if ASSETS_LOGO_PATH.exists():
                logo_image = Image.open(ASSETS_LOGO_PATH).convert("RGBA")
                logo_image.thumbnail((60, 60), RESAMPLE_FILTER)
                self.logo = logo_image
        except OSError as exc:
            logger.error("Failed to load logo: %s", str(exc))
        except (RuntimeError, ValueError) as exc:
            logger.error("Unexpected error loading logo: %s", str(exc))

    def _load_template(self) -> None:
        """Load and cache template background."""
        try:
            if ASSETS_TEMPLATE_PATH.exists():
                template_image = Image.open(ASSETS_TEMPLATE_PATH).convert("RGBA")
                self.template = template_image
                logger.info("Template loaded: %sx%s", template_image.width, template_image.height)
        except OSError as exc:
            logger.error("Failed to load template: %s", str(exc))
        except (RuntimeError, ValueError) as exc:
            logger.error("Unexpected error loading template: %s", str(exc))

    def _load_frames(self) -> None:
        """Load frame assets."""
        try:
            if not ASSETS_FRAMES_PATH.exists():
                logger.warning("Frames folder missing: %s", ASSETS_FRAMES_PATH)
                return

            for frame_path in sorted(ASSETS_FRAMES_PATH.glob("*.png")):
                try:
                    frame_image = Image.open(frame_path).convert("RGBA")
                    thumbnail = frame_image.copy()
                    thumbnail.thumbnail((80, 120), RESAMPLE_FILTER)

                    self.frames.append(frame_path)
                    self.frame_thumbnails.append(ImageTk.PhotoImage(thumbnail))
                except OSError as exc:
                    logger.error("Failed to load frame %s: %s", frame_path, str(exc))
                except (RuntimeError, ValueError) as exc:
                    logger.error("Unexpected error loading frame %s: %s", frame_path, str(exc))
        except (OSError, RuntimeError) as exc:
            logger.error("Error enumerating frames: %s", str(exc))

    def _load_stickers(self) -> None:
        """Load sticker assets."""
        try:
            if not ASSETS_STICKERS_PATH.exists():
                logger.warning("Stickers folder missing: %s", ASSETS_STICKERS_PATH)
                return

            for sticker_path in sorted(ASSETS_STICKERS_PATH.glob("*.png")):
                try:
                    sticker_image = Image.open(sticker_path).convert("RGBA")
                    thumbnail = sticker_image.copy()
                    thumbnail.thumbnail((40, 40), RESAMPLE_FILTER)

                    self.stickers.append(sticker_path)
                    self.sticker_thumbnails.append(ImageTk.PhotoImage(thumbnail))
                except OSError as exc:
                    logger.error("Failed to load sticker %s: %s", sticker_path, str(exc))
                except (RuntimeError, ValueError) as exc:
                    logger.error("Unexpected error loading sticker %s: %s", sticker_path, str(exc))
        except (OSError, RuntimeError) as exc:
            logger.error("Error enumerating stickers: %s", str(exc))

    def get_frame_count(self) -> int:
        """Get number of available frames."""
        return len(self.frames)

    def get_sticker_count(self) -> int:
        """Get number of available stickers."""
        return len(self.stickers)


class Sticker:
    """Represents a sticker placed on the collage."""

    def __init__(
        self,
        image: Image.Image,
        position_x: int,
        position_y: int
    ) -> None:
        """
        Initialize sticker.

        Args:
            image: PIL Image of sticker
            position_x: X coordinate
            position_y: Y coordinate
        """
        self.image: Image.Image = image
        self.position_x: int = position_x
        self.position_y: int = position_y
        self.width: int = image.width
        self.height: int = image.height

    def contains_point(self, point_x: int, point_y: int) -> bool:
        """
        Check if point is within sticker bounds.

        Args:
            point_x: X coordinate
            point_y: Y coordinate

        Returns:
            True if point is within sticker bounds
        """
        x_in_bounds = self.position_x <= point_x <= self.position_x + self.width
        y_in_bounds = self.position_y <= point_y <= self.position_y + self.height
        return x_in_bounds and y_in_bounds

    def set_position(
        self,
        new_x: int,
        new_y: int,
        max_width: int,
        max_height: int
    ) -> None:
        """
        Set sticker position with boundary constraints.

        Args:
            new_x: New X coordinate
            new_y: New Y coordinate
            max_width: Maximum container width
            max_height: Maximum container height
        """
        self.position_x = max(0, min(new_x, max_width - self.width))
        self.position_y = max(0, min(new_y, max_height - self.height))

    def get_bounds(self) -> Tuple[int, int, int, int]:
        """
        Get sticker bounding box.

        Returns:
            Tuple of (x, y, width, height)
        """
        return self.position_x, self.position_y, self.width, self.height


class PhotoCollage:
    """Manages the 4-photo collage composition and rendering."""

    def __init__(self) -> None:
        """Initialize photo collage."""
        self.photos: List[Image.Image] = []
        self.stickers: List[Sticker] = []
        self.selected_frame: Optional[Path] = None
        self.collage_width: int = TEMPLATE_WIDTH
        self.collage_height: int = TEMPLATE_HEIGHT
        self.template: Optional[Image.Image] = None

    def set_template(self, template: Optional[Image.Image]) -> None:
        """
        Set background template.

        Args:
            template: Template PIL Image or None
        """
        self.template = template

    def add_photo(self, photo: Image.Image) -> None:
        """
        Add photo to collage if not full.

        Args:
            photo: PIL Image to add (will be cropped to 16:9 aspect ratio)
        """
        if len(self.photos) < MAX_PHOTOS:
            # Crop to 16:9 aspect ratio
            cropped_photo = self._crop_to_aspect_ratio(photo)
            self.photos.append(cropped_photo)

    def remove_last_photo(self) -> None:
        """Remove last added photo."""
        if self.photos:
            self.photos.pop()

    def add_sticker(self, sticker: Sticker) -> None:
        """
        Add sticker to collage.

        Args:
            sticker: Sticker object to add
        """
        self.stickers.append(sticker)

    def remove_sticker(self, index: int) -> None:
        """
        Remove sticker by index.

        Args:
            index: Index of sticker to remove
        """
        if 0 <= index < len(self.stickers):
            self.stickers.pop(index)

    def find_sticker_at(self, point_x: int, point_y: int) -> Optional[int]:
        """
        Find sticker index at given coordinates (highest z-order first).

        Args:
            point_x: X coordinate
            point_y: Y coordinate

        Returns:
            Index of sticker or None
        """
        for sticker_index in range(len(self.stickers) - 1, -1, -1):
            if self.stickers[sticker_index].contains_point(point_x, point_y):
                return sticker_index
        return None

    def is_full(self) -> bool:
        """
        Check if collage has maximum photos.

        Returns:
            True if collage is full
        """
        return len(self.photos) >= MAX_PHOTOS

    @staticmethod
    def _crop_to_aspect_ratio(photo: Image.Image) -> Image.Image:
        """
        Crop photo to 16:9 aspect ratio (1.78).

        Args:
            photo: Original image

        Returns:
            Cropped image sized to 650x380 (16:9)
        """
        photo_width, photo_height = photo.size
        current_ratio = photo_width / photo_height

        if current_ratio > PHOTO_ASPECT_RATIO:
            # Photo is too wide, crop width
            new_width = int(photo_height * PHOTO_ASPECT_RATIO)
            left = (photo_width - new_width) // 2
            photo = photo.crop((left, 0, left + new_width, photo_height))
        else:
            # Photo is too tall, crop height
            new_height = int(photo_width / PHOTO_ASPECT_RATIO)
            top = (photo_height - new_height) // 2
            photo = photo.crop((0, top, photo_width, top + new_height))

        # Resize to exact photo slot size
        return photo.resize((PHOTO_WIDTH, PHOTO_HEIGHT), RESAMPLE_FILTER)

    def compose_image(self) -> Image.Image:
        """
        Compose final collage image.

        Returns:
            Composed PIL Image
        """
        # Start with template or white background
        if self.template:
            base_collage = self.template.convert("RGBA").copy()
        else:
            base_collage = Image.new(
                "RGBA",
                (self.collage_width, self.collage_height),
                (255, 255, 255, 255)
            )

        # Paste photos into slots
        for photo_index, photo in enumerate(self.photos):
            base_collage = self._paste_photo_to_slot(base_collage, photo, photo_index)

        # Apply overlay frame if selected
        if self.selected_frame:
            base_collage = self._apply_frame(base_collage)

        # Add stickers
        for sticker in self.stickers:
            base_collage = self._paste_sticker(base_collage, sticker)

        return base_collage

    @staticmethod
    def _paste_photo_to_slot(
        base: Image.Image,
        photo: Image.Image,
        index: int
    ) -> Image.Image:
        """
        Paste photo at correct slot position.

        Args:
            base: Base image
            photo: Photo to paste
            index: Photo slot index (0-3)

        Returns:
            Updated image
        """
        try:
            if index >= len(PHOTO_SLOTS):
                return base

            slot_x, slot_y = PHOTO_SLOTS[index]
            base.paste(photo, (slot_x, slot_y))
        except ValueError as exc:
            logger.error("Error pasting photo to slot %d: %s", index, str(exc))
        except (OSError, RuntimeError) as exc:
            logger.error("Unexpected error pasting photo: %s", str(exc))

        return base

    def _apply_frame(self, base: Image.Image) -> Image.Image:
        """
        Apply decorative frame overlay.

        Args:
            base: Base image

        Returns:
            Image with frame applied
        """
        try:
            if not self.selected_frame:
                return base

            frame_image = Image.open(
                self.selected_frame
            ).convert("RGBA").resize(
                (self.collage_width, self.collage_height),
                RESAMPLE_FILTER
            )
            return Image.alpha_composite(base, frame_image)
        except OSError as exc:
            logger.error("Error applying frame: %s", str(exc))
        except ValueError as exc:
            logger.error("Error applying frame (invalid data): %s", str(exc))
        except (RuntimeError, TypeError) as exc:
            logger.error("Unexpected error applying frame: %s", str(exc))

        return base

    @staticmethod
    def _paste_sticker(
        base: Image.Image,
        sticker: Sticker
    ) -> Image.Image:
        """
        Paste sticker onto collage.

        Args:
            base: Base image
            sticker: Sticker to paste

        Returns:
            Updated image
        """
        try:
            sticker_x, sticker_y, _, _ = sticker.get_bounds()
            base.paste(sticker.image, (sticker_x, sticker_y), sticker.image)
        except ValueError as exc:
            logger.error("Error pasting sticker: %s", str(exc))
        except (OSError, RuntimeError) as exc:
            logger.error("Unexpected error pasting sticker: %s", str(exc))

        return base


class CountdownOverlay:
    """Handles countdown overlay rendering."""

    OVERLAY_ALPHA: int = 120
    TEXT_COLOR: Tuple[int, int, int, int] = (255, 255, 255, 255)

    @staticmethod
    def create_overlay(
        base_image: Image.Image,
        countdown_number: int
    ) -> Image.Image:
        """
        Create countdown overlay image.

        Args:
            base_image: Base image
            countdown_number: Number to display

        Returns:
            Image with countdown overlay
        """
        try:
            overlay = base_image.copy().convert("RGBA")
            image_width, image_height = overlay.size

            draw = ImageDraw.Draw(overlay)
            draw.rectangle(
                [(0, 0), (image_width, image_height)],
                fill=(0, 0, 0, CountdownOverlay.OVERLAY_ALPHA)
            )

            font = CountdownOverlay._load_font(image_width, image_height)
            text_str = str(countdown_number)
            text_bbox = draw.textbbox((0, 0), text_str, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_x = (image_width - text_width) / 2
            text_y = (image_height - text_height) / 2

            draw.text(
                (text_x, text_y),
                text_str,
                font=font,
                fill=CountdownOverlay.TEXT_COLOR
            )

            return overlay
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error("Error creating overlay: %s", str(exc))
            return base_image

    @staticmethod
    def _load_font(width: int, height: int) -> ImageFont.FreeTypeFont:
        """
        Load font with fallback.

        Args:
            width: Image width
            height: Image height

        Returns:
            Font object
        """
        font_size = max(24, int(min(width, height) * 0.18))

        try:
            return ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            pass

        try:
            return ImageFont.load_default(size=font_size)
        except (OSError, RuntimeError):
            pass
        return ImageFont.load_default()


class PhotoboothUI:
    """Main UI controller for the photobooth application."""

    def __init__(self, root_window: tk.Tk) -> None:
        """
        Initialize photobooth UI.

        Args:
            root_window: Tkinter root window
        """
        self.root: tk.Tk = root_window
        self.root.title("4-Photo Collage Maker")
        self.root.geometry("1200x900")
        self.root.configure(bg="#FFF5F8")

        # Initialize Configuration and FileManager
        config = Config()
        self.file_manager: FileManager = FileManager(config)

        self.asset_manager: AssetManager = AssetManager()
        self.photo_collage: PhotoCollage = PhotoCollage()
        self.camera_thread: Optional[CameraThread] = None

        self.current_camera_image: Optional[Image.Image] = None
        self.selected_sticker_index: Optional[int] = None
        self.dragging_sticker_index: Optional[int] = None
        self.drag_offset: Tuple[int, int] = (0, 0)

        # Responsive preview scale
        self.preview_scale: float = 0.20
        self.display_width: int = int(TEMPLATE_WIDTH * self.preview_scale)
        self.display_height: int = int(TEMPLATE_HEIGHT * self.preview_scale)

        self.camera_label: Optional[tk.Label] = None
        self.camera_tk_image: Optional[ImageTk.PhotoImage] = None
        self.canvas: Optional[tk.Canvas] = None
        self.canvas_tk_image: Optional[ImageTk.PhotoImage] = None
        self.thumbnail_frame: Optional[tk.Frame] = None
        self.save_button: Optional[tk.Button] = None

        self._load_assets()
        self._build_interface()
        self._start_camera()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _load_assets(self) -> None:
        """Load all assets."""
        self.asset_manager.load_all_assets()
        self.photo_collage.set_template(self.asset_manager.template)

    def _build_interface(self) -> None:
        """Build UI components."""
        self._build_header()
        self._build_frame_selector()
        self._build_main_content()
        self._build_controls()
        self._bind_keyboard_events()

    def _build_header(self) -> None:
        """Build header section."""
        header_frame = tk.Frame(self.root, bg="#FFF5F8")
        header_frame.pack(fill=tk.X, pady=4, padx=4)

        if self.asset_manager.logo:
            try:
                logo_tk = ImageTk.PhotoImage(self.asset_manager.logo)
                logo_label = tk.Label(header_frame, image=logo_tk, bg="#FFF5F8")
                logo_label.image = logo_tk
                logo_label.pack(side=tk.LEFT, padx=4)
            except (OSError, RuntimeError) as exc:
                logger.error("Error displaying logo: %s", str(exc))

        title_label = tk.Label(
            header_frame,
            text="4-Photo Collage Maker",
            font=("Arial", 16, "bold"),
            bg="#FFF5F8"
        )
        title_label.pack(side=tk.LEFT, padx=4)

    def _build_frame_selector(self) -> None:
        """Build frame selection row."""
        frames_row = tk.Frame(self.root, bg="#FFF5F8")
        frames_row.pack(fill=tk.X, padx=4, pady=3)

        tk.Label(frames_row, text="Frames:", font=("Arial", 9), bg="#FFF5F8").pack(side=tk.LEFT, padx=2)

        if self.asset_manager.get_frame_count() > 0:
            for frame_index, thumbnail in enumerate(self.asset_manager.frame_thumbnails):
                frame_path = self.asset_manager.frames[frame_index]
                button = tk.Label(
                    frames_row,
                    image=thumbnail,
                    cursor="hand2",
                    bd=1,
                    relief=tk.RIDGE,
                    bg="#FFF5F8"
                )
                button.image = thumbnail
                button.pack(side=tk.LEFT, padx=2)
                button.bind(
                    "<Button-1>",
                    lambda event, path=frame_path: self._select_frame(path)
                )
        else:
            no_frames_label = tk.Label(frames_row, text="(no frames)", font=("Arial", 8), bg="#FFF5F8")
            no_frames_label.pack(side=tk.LEFT)

    def _build_main_content(self) -> None:
        """Build main content area."""
        main_frame = tk.Frame(self.root, bg="#FFF5F8")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: Camera
        left_frame = tk.Frame(main_frame, bg="#FFE8F0")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        camera_box = tk.LabelFrame(left_frame, text="Camera", bg="#FFE8F0", font=("Arial", 9))
        camera_box.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.camera_label = tk.Label(camera_box, bg="#E6D7F0")
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        # Control buttons
        control_frame = tk.Frame(left_frame, bg="#FFE8F0")
        control_frame.pack(fill=tk.X, padx=4, pady=3)

        tk.Button(
            control_frame,
            text="Capture",
            command=lambda: self._start_countdown(3),
            font=("Arial", 8),
            height=1
        ).pack(side=tk.LEFT, padx=2, fill=tk.BOTH, expand=True)

        tk.Button(
            control_frame,
            text="Upload",
            command=self._upload_photo,
            font=("Arial", 8),
            height=1
        ).pack(side=tk.LEFT, padx=2, fill=tk.BOTH, expand=True)

        tk.Button(
            control_frame,
            text="Undo",
            command=self._undo_last_photo,
            font=("Arial", 8),
            height=1
        ).pack(side=tk.LEFT, padx=2, fill=tk.BOTH, expand=True)

        # Thumbnails
        thumb_label = tk.Label(left_frame, text="Captured (4 slots):", font=("Arial", 8), bg="#FFE8F0")
        thumb_label.pack(anchor="w", padx=4)

        self.thumbnail_frame = tk.Frame(left_frame, bg="#FFE8F0")
        self.thumbnail_frame.pack(fill=tk.X, padx=4, pady=2)

        # Right: Collage preview
        right_frame = tk.Frame(main_frame, bg="#FFF5F8")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        preview_box = tk.LabelFrame(right_frame, text="Collage Preview", bg="#FFF5F8", font=("Arial", 9))
        preview_box.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.canvas = tk.Canvas(
            preview_box,
            width=self.display_width,
            height=self.display_height,
            bg="white"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.canvas.bind("<ButtonPress-1>", self._canvas_press)
        self.canvas.bind("<B1-Motion>", self._canvas_move)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)

        # Stickers
        sticker_label = tk.Label(right_frame, text="Stickers:", font=("Arial", 8), bg="#FFF5F8")
        sticker_label.pack(anchor="w", padx=4)

        palette_frame = tk.Frame(right_frame, bg="#FFF5F8")
        palette_frame.pack(fill=tk.X, padx=4, pady=2)

        if self.asset_manager.get_sticker_count() > 0:
            for sticker_index, thumbnail in enumerate(self.asset_manager.sticker_thumbnails):
                sticker_path = self.asset_manager.stickers[sticker_index]
                label = tk.Label(palette_frame, image=thumbnail, cursor="hand2", bg="#FFF5F8")
                label.image = thumbnail
                label.pack(side=tk.LEFT, padx=2)
                label.bind(
                    "<Button-1>",
                    lambda event, path=sticker_path: self._add_sticker(path)
                )
        else:
            no_stickers_label = tk.Label(palette_frame, text="(no stickers)", font=("Arial", 8), bg="#FFF5F8")
            no_stickers_label.pack()

    def _build_controls(self) -> None:
        """Build control buttons."""
        bottom_frame = tk.Frame(self.root, bg="#FFF5F8")
        bottom_frame.pack(fill=tk.X, padx=4, pady=(0, 4))

        self.save_button = tk.Button(
            bottom_frame,
            text="Save Collage",
            command=self._save_final_image,
            state=tk.DISABLED,
            font=("Arial", 10, "bold"),
            bg="#FFB6D9",
            activebackground="#FF99C4",
            padx=20,
            pady=8
        )
        self.save_button.pack(side=tk.RIGHT)

    def _bind_keyboard_events(self) -> None:
        """Bind keyboard event handlers."""
        self.root.bind("<Delete>", self._delete_selected_sticker)
        self.root.bind("<BackSpace>", self._delete_selected_sticker)

    def _select_frame(self, frame_path: Path) -> None:
        """
        Select a frame template.

        Args:
            frame_path: Path to frame image
        """
        self.photo_collage.selected_frame = frame_path
        logger.info("Frame selected: %s", frame_path.name)
        self._render_preview()

    def _start_camera(self) -> None:
        """Start camera capture thread."""
        def on_camera_frame(pil_image: Image.Image) -> None:
            """Handle incoming camera frame."""
            self.current_camera_image = pil_image
            self._update_camera_display(pil_image)

        self.camera_thread = CameraThread(camera_index=0, on_frame_callback=on_camera_frame)
        self.camera_thread.start()

    def _update_camera_display(self, image: Image.Image) -> None:
        """
        Update camera preview display.

        Args:
            image: PIL Image to display
        """
        try:
            display_image = image.copy()
            display_image.thumbnail(
                (DISPLAY_CAMERA_WIDTH, DISPLAY_CAMERA_HEIGHT),
                RESAMPLE_FILTER
            )
            tk_image = ImageTk.PhotoImage(display_image)

            def update_label() -> None:
                if self.camera_label:
                    self.camera_tk_image = tk_image  # type: ignore
                    self.camera_label.config(image=tk_image)  # type: ignore

            self.root.after(0, update_label)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error("Error updating camera display: %s", str(exc))

    def _start_countdown(self, seconds: int) -> None:
        """
        Start capture countdown.

        Args:
            seconds: Number of seconds to countdown
        """
        def countdown_tick(remaining_seconds: int) -> None:
            if remaining_seconds <= 0:
                self._capture_photo()
                return

            self._show_countdown_overlay(remaining_seconds)
            self.root.after(1000, lambda: countdown_tick(remaining_seconds - 1))

        countdown_tick(seconds)

    def _show_countdown_overlay(self, countdown_num: int) -> None:
        """
        Display countdown number overlay.

        Args:
            countdown_num: Number to display
        """
        if not self.current_camera_image:
            return

        overlay_image = CountdownOverlay.create_overlay(
            self.current_camera_image,
            countdown_num
        )
        self._update_camera_display(overlay_image)

    def _capture_photo(self) -> None:
        """Capture current camera frame."""
        if not self.current_camera_image:
            messagebox.showwarning("No camera", "Camera frame not available.")
            return

        try:
            photo = self.current_camera_image.copy().convert("RGB")
            self.photo_collage.add_photo(photo)

            if self.photo_collage.is_full():
                self.save_button.config(state=tk.NORMAL)

            self._refresh_thumbnails()
            self._render_preview()
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error("Capture failed: %s", str(exc))

    def _upload_photo(self) -> None:
        """Upload photo from file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )

        if not file_path:
            return

        try:
            photo = Image.open(file_path).convert("RGB")
            self.photo_collage.add_photo(photo)

            if self.photo_collage.is_full():
                self.save_button.config(state=tk.NORMAL)

            self._refresh_thumbnails()
            self._render_preview()
        except OSError as exc:
            logger.error("Upload failed: %s", str(exc))
            messagebox.showerror("Upload failed", "Failed to load image. See console for details.")
        except (ValueError, RuntimeError) as exc:
            logger.error("Unexpected error during upload: %s", str(exc))
            messagebox.showerror("Upload failed", "An unexpected error occurred.")

    def _undo_last_photo(self) -> None:
        """Remove last captured photo."""
        self.photo_collage.remove_last_photo()
        self.save_button.config(state=tk.DISABLED)
        self._refresh_thumbnails()
        self._render_preview()

    def _refresh_thumbnails(self) -> None:
        """Update thumbnail display."""
        if not self.thumbnail_frame:
            return

        for widget in self.thumbnail_frame.winfo_children():
            widget.destroy()

        for photo_index, photo in enumerate(self.photo_collage.photos):
            thumbnail = photo.copy()
            thumbnail.thumbnail((100, 60), RESAMPLE_FILTER)
            tk_image = ImageTk.PhotoImage(thumbnail)

            label = tk.Label(
                self.thumbnail_frame,
                image=tk_image,
                bg="#FFE8F0",
                bd=1,
                relief=tk.SOLID
            )
            label.image = tk_image
            label.pack(side=tk.LEFT, padx=2)

            slot_num = tk.Label(
                self.thumbnail_frame,
                text=f"#{photo_index + 1}",
                font=("Arial", 7),
                bg="#FFE8F0"
            )
            slot_num.pack(side=tk.LEFT)

    def _add_sticker(self, sticker_path: Path) -> None:
        """
        Add sticker to collage.

        Args:
            sticker_path: Path to sticker image
        """
        try:
            sticker_image = Image.open(sticker_path).convert("RGBA")
            sticker_width, sticker_height = sticker_image.size

            scale_factor = 120 / max(sticker_width, sticker_height)
            scaled_width = int(sticker_width * scale_factor)
            scaled_height = int(sticker_height * scale_factor)

            scaled_image = sticker_image.resize(
                (scaled_width, scaled_height),
                RESAMPLE_FILTER
            )

            new_sticker = Sticker(scaled_image, 50, 50)
            self.photo_collage.add_sticker(new_sticker)
            self._render_preview()
        except OSError as exc:
            logger.error("Add sticker failed: %s", str(exc))
        except (ValueError, RuntimeError) as exc:
            logger.error("Unexpected error adding sticker: %s", str(exc))

    def _render_preview(self) -> None:
        """Render and display collage preview."""
        if not self.canvas:
            return

        try:
            composed_image = self.photo_collage.compose_image()
            preview_image = composed_image.convert("RGB")
            preview_image.thumbnail(
                (self.display_width, self.display_height),
                RESAMPLE_FILTER
            )

            tk_image = ImageTk.PhotoImage(preview_image)

            self.canvas.delete("all")
            self.canvas_tk_image = tk_image  # type: ignore
            self.canvas.create_image(0, 0, anchor="nw", image=tk_image)  # type: ignore
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error("Preview render failed: %s", str(exc))

    def _canvas_point(self, event: tk.Event) -> Tuple[int, int]:
        """
        Convert canvas coordinates to full image coordinates.

        Args:
            event: Tkinter event

        Returns:
            Tuple of (x, y) coordinates
        """
        full_x = int(event.x / self.preview_scale)
        full_y = int(event.y / self.preview_scale)
        return full_x, full_y

    def _canvas_press(self, event: tk.Event) -> None:
        """
        Handle canvas mouse press.

        Args:
            event: Tkinter event
        """
        if not self.photo_collage.stickers:
            return

        full_x, full_y = self._canvas_point(event)
        sticker_index = self.photo_collage.find_sticker_at(full_x, full_y)

        if sticker_index is not None:
            self.dragging_sticker_index = sticker_index
            self.selected_sticker_index = sticker_index

            sticker = self.photo_collage.stickers[sticker_index]
            sticker_x, sticker_y, _, _ = sticker.get_bounds()
            self.drag_offset = (full_x - sticker_x, full_y - sticker_y)
            self._render_preview()
        else:
            self.selected_sticker_index = None
            self._render_preview()

    def _canvas_move(self, event: tk.Event) -> None:
        """
        Handle canvas mouse drag.

        Args:
            event: Tkinter event
        """
        if self.dragging_sticker_index is None:
            return

        full_x, full_y = self._canvas_point(event)
        sticker = self.photo_collage.stickers[self.dragging_sticker_index]

        new_x = full_x - self.drag_offset[0]
        new_y = full_y - self.drag_offset[1]

        sticker.set_position(
            new_x,
            new_y,
            self.photo_collage.collage_width,
            self.photo_collage.collage_height
        )
        self._render_preview()

    def _canvas_release(self, _event: tk.Event) -> None:
        """
        Handle canvas mouse release.

        Args:
            _event: Tkinter event (unused)
        """
        self.dragging_sticker_index = None

    def _delete_selected_sticker(self, _event: Optional[tk.Event] = None) -> None:
        """
        Delete selected sticker.

        Args:
            _event: Optional Tkinter event (unused)
        """
        if self.selected_sticker_index is not None:
            self.photo_collage.remove_sticker(self.selected_sticker_index)
            self.selected_sticker_index = None
            self._render_preview()

    def _save_final_image(self) -> None:
        """Save final collage image using FileManager."""
        try:
            final_image = self.photo_collage.compose_image()
            rgb_image = final_image.convert("RGB")

            # Use FileManager to save
            filepath = self.file_manager.save_photo_strip(rgb_image)

            logger.info("Image saved to: %s", filepath)
            messagebox.showinfo(
                "Saved",
                f"Collage saved to\n{filepath}\n\nSize: {rgb_image.size}"
            )
        except FileError as exc:
            logger.error("Save failed: %s", str(exc))
            messagebox.showerror("Save failed", f"Error: {str(exc)}")
        except (ValueError, RuntimeError) as exc:
            logger.error("Unexpected error during save: %s", str(exc))
            messagebox.showerror("Save failed", "An unexpected error occurred.")

    def _on_closing(self) -> None:
        """Handle window close event."""
        if self.camera_thread:
            self.camera_thread.stop()
        self.root.destroy()

def main() -> None:
    """Main entry point."""
    root_window = tk.Tk()
    PhotoboothUI(root_window)
    root_window.mainloop()


if __name__ == "__main__":
    main()
