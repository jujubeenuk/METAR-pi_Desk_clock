# ----------------------------------------------------------------------
# 1. CONTENT FROM layout.py
# (Defines layout classes, colors, and special characters)
# ----------------------------------------------------------------------
"""Screen layout definitions and loading."""

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Self, TypeAlias

Coord: TypeAlias = tuple[int, int]
ColorT: TypeAlias = tuple[int, int, int]


class SpChar(StrEnum):
    """Special Characters."""

    CANCEL = "\u2715"
    CHECKMARK = "\u2713"
    DEGREES = "\u00b0"
    DOWN_TRIANGLE = "\u25bc"
    INFO = "\u2139"
    MOON = "\u263e"
    RELOAD = "\u21ba"
    SETTINGS = "\u2699"
    SUN = "\u2600"
    UP_TRIANGLE = "\u25b2"


# Because we swap black/white values for invert, we can't use enum here
class Color:
    """RGB color values."""

    WHITE: ColorT = 255, 255, 255
    BLACK: ColorT = 0, 0, 0
    RED: ColorT = 255, 0, 0
    GREEN: ColorT = 0, 255, 0
    BLUE: ColorT = 0, 0, 255
    PURPLE: ColorT = 150, 0, 255
    GRAY: ColorT = 60, 60, 60

    def __getitem__(self, key: str) -> ColorT:
        try:
            color: ColorT = getattr(self, key)
        except AttributeError as exc:
            msg = f"{key} is not a set color"
            raise KeyError(msg) from exc
        else:
            return color


@dataclass
class FontSize:
    """Font sizes for various text elements."""

    s1: int
    s2: int
    s3: int
    m1: int
    m2: int
    l1: int
    l2: int | None

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> Self:
        """Load font sizes from a dictionary."""
        return cls(
            s1=data["s1"],
            s2=data["s2"],
            s3=data["s3"],
            m1=data["m1"],
            m2=data["m2"],
            l1=data["l1"],
            l2=data.get("l2"),
        )


@dataclass
class ButtonLayout:
    """Button layout settings."""

    radius: int
    outline: int

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> Self:
        """Load button layout settings from a dictionary."""
        return cls(
            radius=data["radius"],
            outline=data["outline"],
        )


@dataclass
class FlightRulesLayout:
    """Flight rules layout settings."""

    vfr: tuple[ColorT, int]
    mvfr: tuple[ColorT, int]
    ifr: tuple[ColorT, int]
    lifr: tuple[ColorT, int]
    na: tuple[ColorT, int]

    def __getitem__(self, key: str) -> tuple[ColorT, int]:
        return getattr(self, key, self.na)

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> Self:
        """Load flight rules layout settings from a dictionary."""
        return cls(
            vfr=(Color.GREEN, data["VFR"]),
            mvfr=(Color.BLUE, data["MVFR"]),
            ifr=(Color.RED, data["IFR"]),
            lifr=(Color.PURPLE, data["LIFR"]),
            na=(Color.BLACK, data["N/A"]),
        )


def as_tuple(data: Any) -> Coord:
    """Convert input data to a 2D coordinate tuple."""
    if data is None:
        msg = "Expected coordinate data, got None"
        raise ValueError(msg)
    if not isinstance(data, list | tuple) or len(data) != 2:
        msg = f"Expected coordinate data, got {data!r}"
        raise ValueError(msg)
    return tuple(data)


def opt_tuple(data: Any) -> Coord | None:
    """Convert input data to a 2D coordinate tuple or return None."""
    if data is None:
        return None
    return as_tuple(data)


@dataclass
class MainLayout:
    """Main layout settings."""

    title: Coord | None
    clock: Coord | None
    clock_label: Coord | None
    station: Coord | None
    timestamp: Coord | None
    timestamp_label: Coord | None
    flight_rules: Coord
    wind_compass: Coord
    wind_compass_radius: int
    wind_speed: Coord
    wind_gust: Coord
    temp: Coord
    temp_icon: Coord | None
    temp_stdv: Coord
    dew: Coord
    humid: Coord
    altim: Coord
    vis: Coord
    cloud_graph: tuple[Coord, Coord]
    wxrmk: tuple[int, int, int, int]
    util_spacing: int
    util_back: Coord

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Load main layout settings from a dictionary."""
        radius: int = data["wind-compass-radius"]
        util_spacing: int = data["util-spacing"]
        clouds = tuple(as_tuple(c) for c in data["cloud-graph"])
        cloud_graph = (clouds[0], clouds[1])
        wxrmk: tuple[int, int, int, int] = tuple(data["wxrmk"])
        return cls(
            title=opt_tuple(data.get("title")),
            clock=opt_tuple(data.get("clock")),
            clock_label=opt_tuple(data.get("clock-label")),
            station=opt_tuple(data.get("station")),
            timestamp=opt_tuple(data.get("timestamp")),
            timestamp_label=opt_tuple(data.get("timestamp-label")),
            flight_rules=as_tuple(data["flight-rules"]),
            wind_compass=as_tuple(data["wind-compass"]),
            wind_compass_radius=radius,
            wind_speed=as_tuple(data["wind-speed"]),
            wind_gust=as_tuple(data["wind-gust"]),
            temp=as_tuple(data["temp"]),
            temp_icon=opt_tuple(data.get("temp-icon")),
            temp_stdv=as_tuple(data["temp-stdv"]),
            dew=as_tuple(data["dew"]),
            humid=as_tuple(data["humid"]),
            altim=as_tuple(data["altim"]),
            vis=as_tuple(data["vis"]),
            cloud_graph=cloud_graph,
            wxrmk=wxrmk,
            util_spacing=util_spacing,
            util_back=as_tuple(data["util-back"]),
        )


@dataclass
class WxRmkLayout:
    """Weather remark layout settings."""

    padding: int
    line_space: int
    col1: int
    col2: int
    wx_length: int
    rmk_length: int

    @classmethod
    def from_dict(cls, data: dict[str, int] | None) -> Self:
        """Load weather remark layout settings from a dictionary."""
        if data is None:
            data = {}
        return cls(
            padding=data.get("padding", 0),
            line_space=data.get("line-space", 0),
            col1=data.get("col1", 0),
            col2=data.get("col2", 0),
            wx_length=data.get("wx-length", 0),
            rmk_length=data.get("rmk-length", 0),
        )


@dataclass
class WxRawLayout:
    """Raw weather layout settings."""

    offset: int
    line_space: int
    cols: int

    @classmethod
    def from_dict(cls, data: dict[str, int] | None) -> Self:
        """Load raw weather layout settings from a dictionary."""
        if data is None:
            data = {}
        return cls(
            offset=data.get("offset", 0),
            line_space=data.get("line-space", 0),
            cols=data.get("cols", 0),
        )


@dataclass
class SelectLayout:
    """Station selection screen layout settings."""

    row_up: int
    row_char: int
    row_down: int
    col_offset: int
    col_spacing: int
    yes: Coord
    no: Coord

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Load station selection layout settings from a dictionary."""
        return cls(
            row_up=data["row-up"],
            row_char=data["row-char"],
            row_down=data["row-down"],
            col_offset=data["col-offset"],
            col_spacing=data["col-spacing"],
            yes=as_tuple(data["yes"]),
            no=as_tuple(data["no"]),
        )


@dataclass
class InfoLayout:
    """Info screen layout settings."""

    text_pos: Coord
    text_line_space: int
    exit: Coord

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Load info layout settings from a dictionary."""
        return cls(
            text_pos=as_tuple(data["text-pos"]),
            text_line_space=data["text-line-space"],
            exit=as_tuple(data["exit"]),
        )


@dataclass
class QuitLayout:
    """Quit screen layout settings."""

    text_pos: Coord
    text_line_space: int
    confirm: Coord
    cancel: Coord

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Load quit layout settings from a dictionary."""
        return cls(
            text_pos=as_tuple(data["text-pos"]),
            text_line_space=data["text-line-space"],
            confirm=as_tuple(data["confirm"]),
            cancel=as_tuple(data["cancel"]),
        )


@dataclass
class ErrorLayout:
    """Error screen layout settings."""

    line1: Coord
    line2: Coord
    line3: Coord
    button_pos: Coord
    button_size: tuple[int, int]
    button_text: Coord
    cancel_pos: Coord

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Load error layout settings from a dictionary."""
        return cls(
            line1=as_tuple(data["line1"]),
            line2=as_tuple(data["line2"]),
            line3=as_tuple(data["line3"]),
            button_pos=as_tuple(data["button-pos"]),
            button_size=as_tuple(data["button-size"]),
            button_text=as_tuple(data["button-text"]),
            cancel_pos=as_tuple(data["cancel-pos"]),
        )


@dataclass
class Layout:
    """Screen layout settings."""

    width: int
    height: int
    large_display: bool
    fonts: FontSize
    button: ButtonLayout
    flight_rules: FlightRulesLayout
    util_pos: Coord
    main: MainLayout
    wx_rmk: WxRmkLayout | None
    wx_raw: WxRawLayout | None
    select: SelectLayout
    info: InfoLayout
    quit: QuitLayout
    error: ErrorLayout

    @property
    def size(self) -> Coord:
        """Get the size of the screen."""
        return self.width, self.height

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Load screen layout settings from a dictionary."""
        wx_rmk = data.get("wxrmk")
        wx_raw = data.get("wxraw")
        return cls(
            width=data["width"],
            height=data["height"],
            large_display=data["large-display"],
            fonts=FontSize.from_dict(data["fonts"]),
            button=ButtonLayout.from_dict(data["button"]),\
            flight_rules=FlightRulesLayout.from_dict(data["fr-display"]),\
            util_pos=as_tuple(data["util"]),\
            main=MainLayout.from_dict(data["main"]),\
            wx_rmk=WxRmkLayout.from_dict(wx_rmk) if wx_rmk else None,\
            wx_raw=WxRawLayout.from_dict(wx_raw) if wx_raw else None,\
            select=SelectLayout.from_dict(data["select"]),\
            info=InfoLayout.from_dict(data["info"]),\
            quit=QuitLayout.from_dict(data["quit"]),\
            error=ErrorLayout.from_dict(data["error"]),\
        )

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Load screen layout settings from a JSON file."""
        with path.open() as fin:
            return cls.from_dict(json.load(fin))


# ----------------------------------------------------------------------
# 2. CONTENT FROM config.py
# (Defines global configuration variables)
# ----------------------------------------------------------------------
"""Shared METAR display settings."""

import logging
# Path is already imported by layout.py

# Seconds between server pings
update_interval = 600

# Seconds between connection retries
timeout_interval = 60

# Set log level - CRITICAL, ERROR, WARNING, INFO, DEBUG
log_level = logging.DEBUG

# Send METAR Pi logs to a file. Ex: "output.log"
log_file = None

# Set to True to shutdown the Pi when exiting the program
shutdown_on_exit = False

# ------- Plate Settings ------- #

# Seconds between plate button reads
button_interval = 0.2

# Seconds between row 2 char scroll
scroll_interval = 0.2

# Remarks section in scroll line
include_remarks = False

# ------- Screen Settings ------ #

# Size of the screen. Loads the layout from "metar_raspi/screen_settings"
screen_size = "800x400"

LOC = Path(__file__).parent
layout_path = LOC / "screen_settings" / f"{screen_size}.json"

# Run the program fullscreen or windowed
fullscreen = True

# Hide the mouse on a touchscreen
hide_mouse = True

# Clock displays UTC or local time
clock_utc = True

# Clock strftime format string
clock_format = r"%H:%M"  # 24-hour
# clock_format = r"%#I:%M" # 12-hour

# Report timestamp strftime format string
timestamp_format = r"%d-%H:%M"  # 24-hour
# timestamp_format = r"%d-%#I:%M" # 12-hour


# ----------------------------------------------------------------------
# 3. CONTENT FROM common.py
# (Defines utility functions and initializes logging)
# NOTE: 'cfg.' prefixes are removed as variables are now global
# ----------------------------------------------------------------------
"""Shared global methods."""

# json and logging are already imported

# IDENT_CHARS is used by screen.py and defined here
IDENT_CHARS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

# Logging initialization using global variables from config.py
logger = logging.getLogger()
logger.setLevel(log_level)
if log_file is not None:
    log_file_handler = logging.FileHandler(log_file)
    log_file_handler.setLevel(log_level)
    logger.addHandler(log_file_handler)


def ident_to_station(idents: list[int]) -> str:
    """Converts 'ident' ints to station string."""
    return "".join([IDENT_CHARS[num] for num in idents])


def station_to_ident(station: str) -> list[int]:
    """Converts station string to 'ident' ints."""
    ret = []
    for char in station:
        try:
            ret.append(IDENT_CHARS.index(char.upper()))
        except ValueError:
            ret.append(0)
    # Pads to 4 if smaller
    return ret + [0] * (4 - len(ret))


def load_session() -> dict:
    """Loads the last saved session from file, or empty dict if not found."""
    try:
        with open(LOC / "session.json") as fin:
            return json.load(fin)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_session(session: dict) -> None:
    """Saves the current session to a file."""
    with open(LOC / "session.json", "w") as fout:
        json.dump(session, fout)


# ----------------------------------------------------------------------
# 4. CONTENT FROM screen.py
# (Main application logic and UI drawing)
# NOTE: All 'metar_raspi.*' and 'common.' prefixes are removed
# NOTE: All 'cfg.' prefixes are removed
# ----------------------------------------------------------------------
"""Display ICAO METAR weather data with a Raspberry Pi and touchscreen."""

import asyncio as aio
import math
import sys
import time
from collections.abc import Callable, Coroutine
from copy import copy
from datetime import UTC, datetime
from os import system
from typing import Any, Self

import pygame
from avwx import Metar, Station
from avwx.exceptions import BadStation, InvalidRequest, SourceError
from avwx.structs import Cloud, MetarData, Number, Units
from dateutil.tz import tzlocal

# LAYOUT is defined by classes from layout.py and layout_path from config.py
LAYOUT = Layout.from_file(layout_path)


# Init pygame and fonts
pygame.init()
ICON_PATH = LOC / "icons"
FONT_PATH = str(ICON_PATH / "DejaVuSans.ttf")

FONT_S1 = pygame.font.Font(FONT_PATH, LAYOUT.fonts.s1)
FONT_S2 = pygame.font.Font(FONT_PATH, LAYOUT.fonts.s2)
FONT_S3 = pygame.font.Font(FONT_PATH, LAYOUT.fonts.s3)
FONT_M1 = pygame.font.Font(FONT_PATH, LAYOUT.fonts.m1)
FONT_M2 = pygame.font.Font(FONT_PATH, LAYOUT.fonts.m2)
FONT_L1 = pygame.font.Font(FONT_PATH, LAYOUT.fonts.l1)
if LAYOUT.fonts.l2:
    FONT_L2 = pygame.font.Font(FONT_PATH, LAYOUT.fonts.l2)


def midpoint(p1: Coord, p2: Coord) -> Coord:
    """Returns the midpoint between two points."""
    return (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2


def centered(rendered_text: pygame.Surface, around: Coord) -> Coord:
    """Returns the top left point for rendered text at a center point."""
    width, height = rendered_text.get_size()
    return around[0] - width // 2 + 1, around[1] - height // 2 + 1


def radius_point(degree: int, center: Coord, radius: int) -> Coord:
    """Returns the degree point on the circumference of a circle."""
    degree %= 360
    x = center[0] + radius * math.cos((degree - 90) * math.pi / 180)
    y = center[1] + radius * math.sin((degree - 90) * math.pi / 180)
    return int(x), int(y)


def hide_mouse() -> None:
    """This makes the mouse transparent."""
    pygame.mouse.set_cursor((8, 8), (0, 0), (0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0))


class Button:
    """Base button class.

    Runs a function when clicked
    """

    # Function to run when clicked. Cannot accept args
    onclick: Callable
    # Text settings
    text: str
    fontsize: int
    # Color strings must match Color.attr names
    fontcolor: str

    def draw(self, win: pygame.Surface, color: Color) -> None:
        """Draw the button on the window with the current color palette."""
        raise NotImplementedError

    def is_clicked(self, pos: Coord) -> bool:
        """Returns True if the position is within the button bounds."""
        raise NotImplementedError


class RectButton(Button):
    """Rectangular buttons can contain text."""

    # Top left
    x1: int
    y1: int
    # Bottom right
    x2: int
    y2: int

    width: int
    height: int

    # Box outline thickness
    thickness: int

    def __init__(
        self,
        bounds: tuple[int, int, int, int],
        action: Callable,
        text: str,
        fontsize: int = LAYOUT.fonts.s3,
        fontcolor: str = "Black",
        thickness: int = LAYOUT.button.outline,
    ):
        self.x1, self.y1, self.width, self.height = bounds
        self.x2 = self.x1 + self.width
        self.y2 = self.y1 + self.height
        self.onclick = action
        self.text = text
        self.fontsize = fontsize
        self.fontcolor = fontcolor
        self.thickness = thickness

    def __repr__(self) -> str:
        return f'<RectButton "{self.text}" at ({self.x1}, {self.y1}), ({self.x2}, {self.y2})>'

    def draw(self, win: pygame.Surface, color: Color) -> None:
        """Draw the button on the window with the current color palette."""
        if self.width is not None:
            bounds = ((self.x1, self.y1), (self.width, self.height))
            pygame.draw.rect(win, color[self.fontcolor], bounds, self.thickness)
        if self.text is not None:
            font = pygame.font.Font(FONT_PATH, self.fontsize)
            rendered = font.render(self.text, 1, color[self.fontcolor])
            rwidth, rheight = rendered.get_size()
            x = self.x1 + (self.width - rwidth) / 2
            y = self.y1 + (self.height - rheight) / 2 + 1
            win.blit(rendered, (x, y))

    def is_clicked(self, pos: Coord) -> bool:
        """Returns True if the position is within the button bounds."""
        return self.x1 < pos[0] < self.x2 and self.y1 < pos[1] < self.y2


class RoundButton(Button):
    """Round buttons."""

    # Center pixel and radius
    x: int
    y: int
    radius: int

    def __init__(
        self,
        center: Coord,
        action: Callable,
        radius: int = LAYOUT.button.radius,
    ):
        self.center = center
        self.radius = radius
        self.onclick = action

    def is_clicked(self, pos: Coord) -> bool:
        """Returns True if the position is within the button bounds."""
        x, y = self.center
        return self.radius > math.hypot(x - pos[0], y - pos[1])


class IconButton(RoundButton):
    """Round button which contain a letter or symbol."""

    # Fill color
    fill: str = "WHITE"
    fontcolor: str = "BLACK"
    fontsize: int = LAYOUT.fonts.l1
    radius: int = LAYOUT.button.radius

    def __init__(
        self,
        center: Coord | None = None,
        action: Callable | None = None,
        icon: str | None = None,
        fontcolor: str | None = None,
        fill: str | None = None,
        radius: int | None = None,
        fontsize: int | None = None,
    ):
        if center:
            self.center = center
        if radius:
            self.radius = radius
        if action:
            self.onclick = action
        if icon:
            self.icon = icon
        if fontsize:
            self.fontsize = fontsize
        if fontcolor:
            self.fontcolor = fontcolor
        if fill:
            self.fill = fill

    def __repr__(self) -> str:
        return f"<IconButton at {self.center} rad {self.radius}>"

    def draw(self, win: pygame.Surface, color: Color) -> None:
        """Draw the button on the window with the current color palette."""
        if self.fill is not None:
            pygame.draw.circle(win, color[self.fill], self.center, self.radius)
        if self.icon is not None:
            font = pygame.font.Font(FONT_PATH, self.fontsize)
            rendered = font.render(self.icon, 1, color[self.fontcolor])
            win.blit(rendered, centered(rendered, self.center))


class ShutdownButton(RoundButton):
    """Round button with a drawn shutdown symbol."""

    fontcolor: str = "WHITE"
    fill: str = "RED"

    def draw(self, win: pygame.Surface, color: Color) -> None:
        """Draw the button on the window with the current color palette."""
        pygame.draw.circle(win, color[self.fill], self.center, self.radius)
        pygame.draw.circle(win, color[self.fontcolor], self.center, self.radius - 6)
        pygame.draw.circle(win, color[self.fill], self.center, self.radius - 9)
        rect = ((self.center[0] - 2, self.center[1] - 10), (4, 20))
        pygame.draw.rect(win, color[self.fontcolor], rect)


class SelectionButton(RoundButton):
    """Round button with icons resembling selection screen."""

    fontcolor: str = "WHITE"
    fill: str = "GREEN"

    def draw(self, win: pygame.Surface, color: Color) -> None:
        """Draw the button on the window with the current color palette."""
        pygame.draw.circle(win, color[self.fill], self.center, self.radius)
        font = FONT_S3 if LAYOUT.large_display else FONT_M1
        for char, direction in ((SpChar.UP_TRIANGLE, -1), (SpChar.DOWN_TRIANGLE, 1)):
            tri = font.render(char, 1, color[self.fontcolor])
            topleft = list(centered(tri, self.center))
            topleft[1] += int(self.radius * 0.5) * direction - 3
            win.blit(tri, topleft)


class CancelButton(IconButton):
    center: Coord = LAYOUT.util_pos
    icon: str = SpChar.CANCEL
    fontcolor: str = "WHITE"
    fill: str = "GRAY"


def draw_func(func: Callable[["METARScreen"], None]) -> Callable[["METARScreen"], None]:
    """Decorator wraps drawing functions with common commands."""

    def wrapper(screen: "METARScreen") -> None:
        screen.on_main = False
        screen.buttons = []
        func(screen)
        screen.draw_buttons()
        pygame.display.flip()
        # This line is a hack to force the screen to redraw
        pygame.event.get()

    return wrapper


class METARScreen:
    """Controls and draws UI elements."""

    ident: list[int]
    old_ident: list[int]
    width: int
    height: int
    win: pygame.Surface
    c: Color
    inverted: bool
    update_time: float
    buttons: list[Button]
    layout: Layout
    is_large: bool

    on_main: bool = False

    def __init__(self, station: str, size: Coord, *, inverted: bool):
        logger.debug("Running init")
        try:
            self.metar = Metar(station)
        except BadStation:
            self.metar = Metar("KJFK")
        # common.station_to_ident -> station_to_ident
        self.ident = station_to_ident(station)
        self.old_ident = copy(self.ident)
        self.width, self.height = size
        # cfg.fullscreen -> fullscreen
        if fullscreen:
            self.win = pygame.display.set_mode(size, pygame.FULLSCREEN)
        else:
            self.win = pygame.display.set_mode(size)
        self.c = Color()
        self.inverted = inverted
        if inverted:
            self.c.BLACK, self.c.WHITE = self.c.WHITE, self.c.BLACK
        # cfg.hide_mouse -> hide_mouse
        if hide_mouse:
            hide_mouse()
        self.reset_update_time()
        self.buttons = []
        self.layout = LAYOUT
        self.is_large = self.layout.large_display
        logger.debug("Finished running init")

    @property
    def station(self) -> str:
        """The current station."""
        # common.ident_to_station -> ident_to_station
        return ident_to_station(self.ident)

    @classmethod
    def from_session(cls, session: dict, size: Coord) -> Self:
        """Returns a new Screen from a saved session."""
        station = session.get("station", "KJFK")
        inverted = session.get("inverted", True)
        return cls(station, size, inverted=inverted)

    def export_session(self, *, save: bool = True) -> dict:
        """Saves or returns a dictionary representing the session's state."""
        session = {"station": self.station, "inverted": self.inverted}
        if save:
            # common.save_session -> save_session
            save_session(session)
        return session

    def reset_update_time(self, interval: int | None = None) -> None:
        """Call to reset the update time to now plus the update interval."""
        # cfg.update_interval -> update_interval
        self.update_time = time.time() + (interval or update_interval)

    async def refresh_data(self, *, force_main: bool = False, ignore_updated: bool = False) -> None:
        """Refresh existing station data."""
        logger.info("Calling refresh update")
        try:
            updated = await self.metar.async_update()
        except ConnectionError:
            await self.wait_for_network()
        except (TimeoutError, SourceError):
            self.error_connection()
        except InvalidRequest:
            self.error_station()
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"An unknown error has occurred: {exc}")
            self.error_unknown()
        else:
            logger.info(self.metar.raw)
            self.reset_update_time()
            if ignore_updated:
                updated = True
            if updated and (self.on_main or force_main):
                self.draw_main()
            elif force_main and not updated:
                self.error_no_data()

    async def new_station(self) -> None:
        """Update the current station from ident and display new main screen."""
        logger.info("Calling new update")
        self.draw_loading_screen()
        new_metar = Metar(self.station)
        try:
            if not await new_metar.async_update():
                self.error_no_data()
                return
        except (TimeoutError, ConnectionError, SourceError):
            self.error_connection()
        except InvalidRequest:
            self.error_station()
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"An unknown error has occurred: {exc}")
            self.error_unknown()
        else:
            logger.info(new_metar.raw)
            self.metar = new_metar
            self.old_ident = copy(self.ident)
            self.reset_update_time()
            self.export_session()
            self.draw_main()

    async def verify_station(self) -> None:
        """Verifies the station value before calling new data."""
        try:
            station = Station.from_icao(self.station)
            if not station.sends_reports:
                self.error_reporting()
                return
        except BadStation:
            self.error_station()
        else:
            await self.new_station()

    def cancel_station(self) -> None:
        """Revert ident and redraw main screen."""
        self.ident = self.old_ident
        if self.metar.data is None:
            self.error_no_data()
        else:
            self.draw_main()

    def draw_buttons(self) -> None:
        """Draw all current buttons."""
        for button in self.buttons:
            button.draw(self.win, self.c)

    @draw_func
    def draw_selection_screen(self) -> None:
        """Load selection screen elements."""
        self.win.fill(self.c.WHITE)
        # Draw Selection Grid
        yes, no = self.layout.select.yes, self.layout.select.no
        self.buttons = [
            IconButton(yes, self.verify_station, SpChar.CHECKMARK, "WHITE", "GREEN"),
            CancelButton(no, self.cancel_station, fill="RED"),
        ]
        upy = self.layout.select.row_up
        chary = self.layout.select.row_char
        downy = self.layout.select.row_down
        for col in range(4):
            x = self.__selection_get_x(col)
            self.buttons.append(IconButton((x, upy), self.__incr_ident(col, down=True), SpChar.UP_TRIANGLE))
            self.buttons.append(IconButton((x, downy), self.__incr_ident(col, down=False), SpChar.DOWN_TRIANGLE))
            rendered = FONT_L1.render(IDENT_CHARS[self.ident[col]], 1, self.c.BLACK)
            self.win.blit(rendered, centered(rendered, (x, chary)))

    def __selection_get_x(self, col: int) -> int:
        """Returns the top left x pixel for a desired column."""
        offset = self.layout.select.col_offset
        spacing = self.layout.select.col_spacing
        return offset + col * spacing

    def __incr_ident(self, pos: int, *, down: bool) -> Callable:
        """Returns a function to update and replace ident char on display.

        pos: 0-3 column
        down: increment/decrement counter
        """

        def update_func() -> None:
            # Update ident
            if down:
                if self.ident[pos] == 0:
                    self.ident[pos] = len(IDENT_CHARS)
                self.ident[pos] -= 1
            else:
                self.ident[pos] += 1
                if self.ident[pos] == len(IDENT_CHARS):
                    self.ident[pos] = 0
            # Update display
            rendered = FONT_L1.render(IDENT_CHARS[self.ident[pos]], 1, self.c.BLACK)
            x = self.__selection_get_x(pos)
            chary = self.layout.select.row_char
            spacing = self.layout.select.col_spacing
            region = (x - spacing / 2, chary - spacing / 2, spacing, spacing)
            pygame.draw.rect(self.win, self.c.WHITE, region)
            self.win.blit(rendered, centered(rendered, (x, chary)))
            pygame.display.update(region)

        return update_func

    @draw_func
    def draw_loading_screen(self) -> None:
        """Display load screen."""
        # Reset on_main because the main screen should always display on success
        self.on_main = True
        self.win.fill(self.c.WHITE)
        point = self.layout.error.line1
        self.win.blit(FONT_M2.render("Fetching weather", 1, self.c.BLACK), point)
        point = self.layout.error.line2
        self.win.blit(FONT_M2.render("data for " + self.station, 1, self.c.BLACK), point)

    def __draw_clock(self) -> None:
        """Draw the clock components."""
        if not (self.layout.main.clock and self.layout.main.clock_label):
            return
        # cfg.clock_utc -> clock_utc
        now = datetime.now(UTC) if clock_utc else datetime.now(tzlocal())
        label = now.tzname() or "UTC"
        clock_font = globals().get("FONT_L2") or FONT_L1
        # cfg.clock_format -> clock_format
        clock_text = clock_font.render(now.strftime(clock_format), 1, self.c.BLACK)
        x, y = self.layout.main.clock
        w, h = clock_text.get_size()
        pygame.draw.rect(self.win, self.c.WHITE, ((x, y), (x + w, (y + h) * 0.9)))
        self.win.blit(clock_text, (x, y))
        label_font = FONT_M1 if self.is_large else FONT_S3
        point = self.layout.main.clock_label
        self.win.blit(label_font.render(label, 1, self.c.BLACK), point)

    def update_clock(self) -> None:
        """Update clock display elements."""
        if self.on_main:
            self.__draw_clock()
            pygame.display.flip()

    def __draw_wind_compass(self, data: MetarData, center: Coord, radius: int) -> None:
        """Draw the wind direction compass."""
        wdir = data.wind_direction
        var = data.wind_variable_direction
        pygame.draw.circle(self.win, self.c.GRAY, center, radius, 3)
        if data.wind_speed and not data.wind_speed.value:
            text = FONT_S3.render("Calm", 1, self.c.BLACK)
        elif wdir and wdir.repr == "VRB":
            text = FONT_S3.render("VRB", 1, self.c.BLACK)
        elif wdir and (wdir_value := wdir.value):
            text = FONT_M1.render(str(wdir_value).zfill(3), 1, self.c.BLACK)
            rad_point = radius_point(int(wdir_value), center, radius)
            width = 4 if self.is_large else 2
            pygame.draw.line(self.win, self.c.RED, center, rad_point, width)
            if var:
                for point in var:
                    if point.value is not None:
                        rad_point = radius_point(int(point.value), center, radius)
                        pygame.draw.line(self.win, self.c.BLUE, center, rad_point, width)
        else:
            text = FONT_L1.render(SpChar.CANCEL, 1, self.c.RED)
        self.win.blit(text, centered(text, center))

    def __draw_wind(self, data: MetarData, unit: str) -> None:
        """Draw the dynamic wind elements."""
        speed, gust = data.wind_speed, data.wind_gust
        point = self.layout.main.wind_compass
        radius = self.layout.main.wind_compass_radius
        self.__draw_wind_compass(data, point, radius)
        if speed and speed.value:
            rendered = FONT_S3.render(f"{speed.value} {unit}", 1, self.c.BLACK)
            point = self.layout.main.wind_speed
            self.win.blit(rendered, centered(rendered, point))
            text = f"G: {gust.value}" if gust else "No Gust"
            rendered = FONT_S3.render(text, 1, self.c.BLACK)
            self.win.blit(rendered, centered(rendered, self.layout.main.wind_gust))

    def __draw_temp_icon(self, temp: int) -> None:
        """Draw the temperature icon."""
        if not self.layout.main.temp_icon:
            return
        therm_level = 0
        if temp:
            therm_level = temp // 12 + 2
            if therm_level < 0:
                therm_level = 0
        add_i = "I" if self.inverted else ""
        therm_icon = f"Therm{therm_level}{add_i}.png"
        point = self.layout.main.temp_icon
        self.win.blit(pygame.image.load(str(ICON_PATH / therm_icon)), point)

    def __draw_temp_dew_humidity(self, data: MetarData) -> None:
        """Draw the dynamic temperature, dewpoint, and humidity elements."""
        temp = data.temperature
        dew = data.dewpoint
        if self.is_large:
            temp_text = "Temp "
            diff_text = "Std Dev "
            dew_text = "Dewpoint "
            hmd_text = "Humidity "
        else:
            temp_text = "TMP: "
            diff_text = "STD: "
            dew_text = "DEW: "
            hmd_text = "HMD: "
        # Dewpoint
        dew_text += f"{dew.value}{SpChar.DEGREES}" if dew else "--"
        point = self.layout.main.dew
        self.win.blit(FONT_S3.render(dew_text, 1, self.c.BLACK), point)
        # Temperature
        if temp and temp.value is not None:
            temp_text += f"{temp.value}{SpChar.DEGREES}"
            if self.is_large and self.metar.units:
                temp_text += self.metar.units.temperature
            temp_diff = temp.value - 15
            diff_sign = "-" if temp_diff < 0 else "+"
            diff_text += f"{diff_sign}{abs(temp_diff)}{SpChar.DEGREES}"
        else:
            temp_text += "--"
            diff_text += "--"
        point = self.layout.main.temp
        self.win.blit(FONT_S3.render(temp_text, 1, self.c.BLACK), point)
        point = self.layout.main.temp_stdv
        self.win.blit(FONT_S3.render(diff_text, 1, self.c.BLACK), point)
        if temp and temp.value is not None and self.layout.main.temp_icon:
            self.__draw_temp_icon(int(temp.value))
        # Humidity
        if temp and dew and isinstance(temp.value, int) and isinstance(dew.value, int):
            # Calculate humidity: RH = 100 * (EXP((17.625 * TD) / (243.04 + TD)) / EXP((17.625 * T) / (243.04 + T)))
            try:
                rh = 100 * math.exp(
                    (17.625 * dew.value) / (243.04 + dew.value)
                    - (17.625 * temp.value) / (243.04 + temp.value)
                )
            except ValueError:
                rh = -1
            if rh >= 0:
                hmd_text += f"{int(rh)}%"
            else:
                hmd_text += "--"
        else:
            hmd_text += "--"
        point = self.layout.main.humid
        self.win.blit(FONT_S3.render(hmd_text, 1, self.c.BLACK), point)

    def __draw_altim_vis(self, data: MetarData) -> None:
        """Draw the dynamic altimeter and visibility elements."""
        # Altimeter
        altim = data.altimeter
        point = self.layout.main.altim
        if altim:
            rendered = FONT_M1.render(altim.repr, 1, self.c.BLACK)
        else:
            rendered = FONT_M1.render("--", 1, self.c.BLACK)
        self.win.blit(rendered, centered(rendered, point))
        # Visibility
        vis = data.visibility
        point = self.layout.main.vis
        if vis and vis.repr:
            rendered = FONT_M1.render(vis.repr, 1, self.c.BLACK)
        else:
            rendered = FONT_M1.render("--", 1, self.c.BLACK)
        self.win.blit(rendered, centered(rendered, point))

    def __draw_cloud_graph(self, clouds: list[Cloud]) -> None:
        """Draw the cloud cover graph."""
        top, bottom = self.layout.main.cloud_graph
        mid = midpoint(top, bottom)
        pygame.draw.line(self.win, self.c.BLACK, top, bottom, 2)
        height = bottom[1] - top[1]
        for cloud in clouds:
            # Skip vertical vis
            if cloud.modifier and cloud.modifier.startswith("VV"):
                continue
            # Draw cloud layer
            height_ft = cloud.height.value if cloud.height and cloud.height.value else 12000
            # Scale up to 15k ft for 15k display height
            y_perc = height_ft / 15000
            y = int(bottom[1] - height * y_perc)
            line = ((top[0] - 5, y), (top[0] + 5, y))
            pygame.draw.line(self.win, self.c.BLACK, line[0], line[1], 1)
            # Draw cover text
            text = f"{cloud.repr} "
            if cloud.height and cloud.height.value:
                text += str(cloud.height.value)
            rendered = FONT_S3.render(text, 1, self.c.BLACK)
            x = mid[0] + 10
            y -= FONT_S3.get_height() // 2
            self.win.blit(rendered, (x, y))

    def __draw_flight_rules(self, data: MetarData) -> None:
        """Draw the flight rules display area."""
        fr_data = self.layout.flight_rules[data.flight_rules]
        color, height = fr_data
        rect = ((0, 0), (self.width, height))
        pygame.draw.rect(self.win, color, rect)
        text = FONT_M2.render(data.flight_rules, 1, self.c.WHITE)
        self.win.blit(text, centered(text, self.layout.main.flight_rules))

    def __draw_info_elements(self, data: MetarData) -> None:
        """Draw the main information elements for the current METAR."""
        # Station ID
        point = self.layout.main.station
        rendered = FONT_L1.render(self.station, 1, self.c.BLACK)
        self.win.blit(rendered, centered(rendered, point))
        # Time Stamp
        if data.time:
            point = self.layout.main.timestamp
            # cfg.timestamp_format -> timestamp_format
            rendered = FONT_M2.render(data.time.dt.strftime(timestamp_format) + "Z", 1, self.c.BLACK)
            self.win.blit(rendered, centered(rendered, point))
            point = self.layout.main.timestamp_label
            rendered = FONT_S2.render("Updated", 1, self.c.BLACK)
            self.win.blit(rendered, centered(rendered, point))
        # Wind components
        units = data.units.wind_speed
        self.__draw_wind(data, units)
        # Temp/Dew/Humid components
        self.__draw_temp_dew_humidity(data)
        # Altimeter and Visibility
        self.__draw_altim_vis(data)
        # Cloud Graph
        self.__draw_cloud_graph(data.clouds)

    def __draw_util_bar(self) -> None:
        """Draw the utility bar buttons for the main screen."""
        self.buttons = []
        # Background
        util_back = self.layout.main.util_back
        pygame.draw.rect(self.win, self.c.GRAY, util_back)
        # Options button
        self.buttons.append(IconButton(self.layout.util_pos, self.draw_options_bar, SpChar.SETTINGS))
        # WX/RMK button
        if self.layout.main.wxrmk:
            wxrmk = self.layout.main.wxrmk
            bounds = ((wxrmk[0], wxrmk[1]), (wxrmk[2], wxrmk[3]))
            self.buttons.append(RectButton(bounds, self.draw_wx_rmk, "WX/RMK", self.layout.fonts.s3, "WHITE", 0))
        # Clock
        self.__draw_clock()

    @draw_func
    def draw_main(self) -> None:
        """Draw the METAR main screen."""
        self.on_main = True
        self.win.fill(self.c.WHITE)
        data = self.metar.data
        if data is None:
            self.error_no_data()
            return
        self.__draw_flight_rules(data)
        self.__draw_info_elements(data)
        self.__draw_util_bar()

    @draw_func
    def draw_options_bar(self) -> None:
        """Draw the options buttons."""
        self.win.fill(self.c.GRAY, self.layout.main.util_back)
        pos = list(self.layout.util_pos)
        spacing = self.layout.main.util_spacing
        # Cancel
        self.buttons.append(CancelButton(self.layout.util_pos, self.draw_main, fill="GRAY"))
        # Selection
        pos[0] -= spacing
        self.buttons.append(SelectionButton(tuple(pos), self.draw_selection_screen))
        # Shutdown
        pos[0] -= spacing
        self.buttons.append(ShutdownButton(tuple(pos), self.draw_quit_screen))
        # Invert
        pos[0] -= spacing
        icon = SpChar.SUN if self.inverted else SpChar.MOON
        self.buttons.append(IconButton(tuple(pos), self.invert_wb, icon, fill="WHITE", fontcolor="BLACK"))
        # Info
        pos[0] -= spacing
        self.buttons.append(IconButton(tuple(pos), self.draw_info_screen, SpChar.INFO, fill="BLUE"))

    def invert_wb(self) -> None:
        """Invert black/white colors and redraw."""
        self.inverted = not self.inverted
        if self.inverted:
            self.c.BLACK, self.c.WHITE = self.c.WHITE, self.c.BLACK
        else:
            self.c.BLACK, self.c.WHITE = self.c.WHITE, self.c.BLACK
        self.export_session()
        self.draw_options_bar()

    @draw_func
    def draw_quit_screen(self) -> None:
        """Draw the quit/shutdown screen."""
        self.win.fill(self.c.WHITE)
        quit_layout = self.layout.quit
        point = quit_layout.text_pos
        self.win.blit(FONT_M2.render("Are you sure you want to quit?", 1, self.c.BLACK), point)
        point = (point[0], point[1] + quit_layout.text_line_space)
        if shutdown_on_exit:
            self.win.blit(FONT_M2.render("The system will shut down.", 1, self.c.BLACK), point)
        else:
            self.win.blit(FONT_M2.render("The application will exit.", 1, self.c.BLACK), point)
        # Confirm Button
        self.buttons.append(IconButton(quit_layout.confirm, self.do_quit, SpChar.CHECKMARK, "WHITE", "RED"))
        # Cancel Button
        self.buttons.append(CancelButton(quit_layout.cancel, self.draw_main, fill="GREEN"))

    def do_quit(self) -> None:
        """Quits the program and shuts down the Pi if configured."""
        logger.info("Quitting")
        self.export_session()
        pygame.quit()
        if shutdown_on_exit:
            system("sudo shutdown now")
        sys.exit()

    @draw_func
    def draw_info_screen(self) -> None:
        """Draw the info screen."""
        self.win.fill(self.c.WHITE)
        info = self.layout.info
        text = [
            "METAR-RasPi",
            f"Layout: {screen_size}",
            f"Station: {self.station}",
            f"Update: {update_interval}s",
            "metar-raspi.com",
        ]
        point = info.text_pos
        line_space = info.text_line_space
        for line in text:
            rendered = FONT_M2.render(line, 1, self.c.BLACK)
            self.win.blit(rendered, point)
            point = (point[0], point[1] + line_space)
        # Exit Button
        self.buttons.append(CancelButton(info.exit, self.draw_main, fill="GREEN"))

    def __draw_error(self, message: str, line: str, button_action: Callable | None = None) -> None:
        """Draw a common error screen."""
        self.on_main = False
        self.win.fill(self.c.WHITE)
        error = self.layout.error
        # Error text
        self.win.blit(FONT_M2.render(message, 1, self.c.RED), error.line1)
        self.win.blit(FONT_M2.render(line, 1, self.c.BLACK), error.line2)
        # Reload/Fix button
        if button_action:
            button_bounds = ((error.button_pos), (error.button_size))
            self.buttons.append(
                RectButton(button_bounds, button_action, SpChar.RELOAD, self.layout.fonts.l1, "WHITE", 0)
            )
            self.win.blit(
                FONT_S3.render("Reload", 1, self.c.WHITE),
                centered(FONT_S3.render("Reload", 1, self.c.WHITE), error.button_text),
            )
        # Cancel button
        else:
            self.buttons.append(CancelButton(error.cancel_pos, self.cancel_station, fill="RED"))
        self.draw_buttons()
        pygame.display.flip()
        pygame.event.get()

    def error_connection(self) -> None:
        """Draw a connection error screen with a retry button."""
        self.__draw_error(
            "Connection Error",
            "Check network connection or source API status.",
            self.refresh_data,
        )

    def error_no_data(self) -> None:
        """Draw a no data error screen with a retry button."""
        self.__draw_error("No Data", f"No METAR found for {self.station}", self.refresh_data)

    def error_station(self) -> None:
        """Draw an invalid station error screen with a cancel button."""
        self.__draw_error(
            "Invalid Station",
            f"{self.station} is not a valid ICAO identifier",
            None,
        )

    def error_reporting(self) -> None:
        """Draw an invalid station error screen with a cancel button."""
        self.__draw_error(
            "Non-Reporting Station",
            f"{self.station} does not report METAR data",
            None,
        )

    def error_unknown(self) -> None:
        """Draw an unknown error screen with a reload button."""
        self.__draw_error("Unknown Error", "An unexpected error has occurred", self.refresh_data)

    @draw_func
    def draw_wx_rmk(self) -> None:
        """Draw the WX/RMK/Raw data screen."""
        self.win.fill(self.c.WHITE)
        data = self.metar.data
        if data is None:
            self.error_no_data()
            return
        wx_rmk_layout = self.layout.wx_rmk
        if wx_rmk_layout is None or self.metar.raw is None:
            # If no layout or raw data, go to raw screen if available, else main
            if self.layout.wx_raw:
                self.draw_wx_raw()
            else:
                self.draw_main()
            return

        # WX/RMK screen logic
        # ... (implementation needed here based on the 800x400.json layout)
        
        # Cancel button to return to main
        self.buttons.append(CancelButton(self.layout.util_pos, self.draw_main, fill="GRAY"))


    @draw_func
    def draw_wx_raw(self) -> None:
        """Draw the raw METAR screen."""
        self.win.fill(self.c.WHITE)
        if self.metar.raw is None:
            self.error_no_data()
            return
        
        # Raw screen logic
        # ... (implementation needed here based on the 800x400.json layout)

        # Cancel button to return to main
        self.buttons.append(CancelButton(self.layout.util_pos, self.draw_main, fill="GRAY"))


    async def wait_for_network(self) -> None:
        """Draw a waiting screen and pause while waiting for network connection."""
        self.on_main = False
        self.win.fill(self.c.WHITE)
        error = self.layout.error
        self.win.blit(FONT_M2.render("Waiting for network connection...", 1, self.c.RED), error.line1)
        self.win.blit(FONT_M2.render("Retrying in 60 seconds", 1, self.c.BLACK), error.line2)
        pygame.display.flip()
        pygame.event.get()
        await aio.sleep(timeout_interval)
        await self.refresh_data()


async def update_loop(screen: METARScreen) -> None:
    """Handles updating the METAR data in the background."""
    while True:
        if screen.on_main and time.time() > screen.update_time:
            await screen.refresh_data()
        await aio.sleep(10)


async def input_loop(screen: METARScreen) -> None:
    """Handles touch input events."""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                screen.do_quit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if hide_mouse:
                    hide_mouse()
                for button in screen.buttons:
                    if button.is_clicked(pos):
                        if aio.iscoroutinefunction(button.onclick):
                            await button.onclick()
                        else:
                            button.onclick()
                        break
        await aio.sleep(0.01)


async def clock_loop(screen: METARScreen) -> None:
    """Handles updating the clock while on the main screen."""
    while True:
        if screen.on_main:
            screen.update_clock()
        await aio.sleep(1)


def run_with_touch_input(screen: METARScreen, *tasks: Coroutine[Any, Any, None]) -> None:
    """Runs an async screen function with touch input enabled."""
    coros = [*tasks, input_loop(screen)]

    async def run_tasks() -> None:
        await aio.wait((aio.create_task(coro) for coro in coros), return_when=aio.FIRST_COMPLETED)

    aio.run(run_tasks())


def main() -> None:
    """Program main handles METAR data handling and user interaction flow."""
    logger.debug("Booting")
    # common.load_session -> load_session
    screen = METARScreen.from_session(load_session(), LAYOUT.size)
    screen.draw_loading_screen()
    run_with_touch_input(screen, screen.refresh_data(force_main=True))
    logger.debug("Setup complete")
    coros = [update_loop(screen)]
    if screen.layout.main.clock:
        coros.append(clock_loop(screen))
    run_with_touch_input(screen, *coros)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()