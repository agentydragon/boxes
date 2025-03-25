from __future__ import annotations

import colorsys
import contextlib
import dataclasses
import functools
import inspect
import itertools
import logging
import random
from decimal import ROUND_HALF_UP, Decimal
from math import cos, radians, sin
from numbers import Number
from textwrap import indent
from typing import Iterable

import numpy as np
import tabulate
from hamcrest import assert_that, contains_exactly, equal_to

from boxes import Boxes, Color
from boxes.edges import BaseEdge, FingerJointEdge, FingerJointSettings

PLAIN = "e"
DEG_SIGN = "°"
ALPHA_SIGN = "α"


def random_color():
    # Generate a random hue and saturation
    h = random.random()  # Hue: 0-1
    s = random.uniform(0.5, 1)  # Ensure some saturation (avoid washed-out colors)
    v = random.uniform(0.6, 0.7)  # Avoid too-bright colors that blend into white
    # Convert HSV to RGB
    return colorsys.hsv_to_rgb(h, s, v)


def coord(x, y):
    return np.array([x, y], dtype=float)


def dict_only_keys(d, want):
    want, have = set(want), set(d.keys())
    assert want <= have, f"{want - have = }"
    return {k: d[k] for k in want}


def fmt(x, show_sign=False):
    d = Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    s = str(d).rstrip("0").rstrip(".")
    if show_sign and x >= 0:
        s = "+" + s
    return s


def fmt_mm(x):
    return f"{fmt(x)}mm"


def fmt_deg(x):
    return f"{fmt(x)}°"


def fmt_reldeg(x):
    if x == 0:
        return "0"
    return f"{fmt(x, show_sign=True)}°"


class SkippingFingerJoint(FingerJointEdge):
    def __init__(self, *args, idx_predicate, reversed: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []
        self.idx_predicate = idx_predicate
        self.reversed = reversed

        self.logger = logging.getLogger("SkippingFingerJoint")

    def edge(self, length, *args, **kwargs):
        self.calls.append(("edge", length, args, kwargs))

    def draw_finger(self, f, h, *args, **kwargs):
        self.calls.append(("draw_finger", f, h, args, kwargs))

    def _rewrite_calls(self, calls):
        finger = 0
        for call in calls:
            match call:
                case ("edge", length, args, kwargs):
                    self.logger.info(f"  edge({length})")
                    yield call
                case ("draw_finger", f, h, args, kwargs):
                    prefix = f"finger({f=}, {h=}) #{finger} =>"
                    if self.idx_predicate(finger):
                        self.logger.info(f"  {prefix} accept")
                        yield call
                    else:
                        self.logger.info(f"  {prefix} reject, edge")
                        yield ("edge", f, [], {})
                    finger += 1
                case _:
                    raise ValueError(f"Unknown call: {call}")

    def __call__(self, length, bedBolts=None, **kw):
        self.logger.info(f"<skipping finger joint {length=} {self.reversed=}>")
        self.calls = []
        super().__call__(length, bedBolts, **kw)

        calls = list(self._rewrite_calls(self.calls))
        if self.reversed:
            calls = reversed(calls)
        self.logger.info("calling:")
        for c in calls:
            match c:
                case ("edge", length, args, kwargs):
                    self.logger.info(f"edge({length})")
                    self.boxes.edge(length, *args, **kwargs)
                case ("draw_finger", f, h, args, kwargs):
                    self.logger.info(f"draw_finger({f=}, {h=})")
                    super().draw_finger(f, h, *args, **kwargs)
                case _:
                    raise ValueError(f"Unknown call: {c}")
        self.logger.info("</skipping finger joint>")


def inject_shortcuts(func):
    """
    Decorator for instance methods:
    1) Pull some parameters from self.shortcuts (a dict) first.
    2) Map positional arguments onto the *remaining* parameters in signature order.
    3) Apply any remaining kwargs last.

    Example usage:

        class Example:
            @property
            def shortcuts(self):
                return {
                    'f': "injectedF",
                    'sx': [1, 2, 3],
                    # anything you want auto-injected
                }

            @inject_after_shortcuts
            def sx_separated(self, f, sx, edge, mode="outer"):
                print("f =", f)
                print("sx =", sx)
                print("edge =", edge)
                print("mode =", mode)

        e = Example()
        # The single positional argument "F" will go to 'edge'
        # because 'f' and 'sx' are already taken from self.shortcuts.
        e.sx_separated("F")

    This prints:
        f = injectedF
        sx = [1, 2, 3]
        edge = F
        mode = outer
    """

    # Original function's signature
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # 1) Grab shortcuts (dict) from self
        shortcuts = getattr(self, "shortcuts", {})
        if not isinstance(shortcuts, dict):
            raise TypeError("self.shortcuts must be a dict for @inject_shortcuts")

        # 2) Separate out "self" from the parameters
        parameters = [p for p in sig.parameters.values() if p.name != "self"]

        # 3) Start building an argument map (like a manual 'bind')
        #    Fill in whatever we have in shortcuts
        arg_map = {}
        for p in parameters:
            if p.name in shortcuts:
                arg_map[p.name] = shortcuts[p.name]

        # 4) We now assign the user's *positional* arguments to the
        #    still-unfilled parameters in order
        unfilled_params = [p for p in parameters if p.name not in arg_map]
        pos_args = list(args)
        if len(pos_args) > len(unfilled_params):
            raise TypeError(
                f"{func.__name__}() got {len(pos_args)} positional args, "
                f"but only {len(unfilled_params)} available after injection"
            )
        for i, val in enumerate(pos_args):
            p = unfilled_params[i]
            arg_map[p.name] = val

        # 5) Apply kwargs, checking for conflicts
        for k, v in kwargs.items():
            if k in arg_map:
                raise TypeError(
                    f"{func.__name__}() got multiple values for argument '{k}'"
                )
            arg_map[k] = v

        # 6) Now assemble the final call-args in the correct order
        final_args = []
        for p in sig.parameters.values():
            if p.name == "self":
                final_args.append(self)
                continue

            if p.name in arg_map:
                final_args.append(arg_map[p.name])
            else:
                # If missing but has a default, use it. Otherwise error.
                if p.default is not inspect.Parameter.empty:
                    final_args.append(p.default)
                else:
                    raise TypeError(
                        f"{func.__name__}() missing required argument: '{p.name}'"
                    )

        return func(*final_args)

    return wrapper


#
# def inject_shortcuts(func):
#    """
#    Returns a decorator that looks at the decorated function's parameter names
#    and, for any that aren't provided in the call, attempts to inject them
#    from 'data' (a dict).
#    """
#
#    sig = inspect.signature(func)
#
#    @functools.wraps(func)
#    def wrapper(self, *args, **kwargs):
#        shortcuts = self.shortcuts
#        assert set(kwargs).isdisjoint(shortcuts), "would overwrite shortcuts"
#        used_shortcuts = set(sig.parameters) & set(shortcuts.keys())
#
#        # Bind call arguments to their parameter names
#        bound = sig.bind_partial(
#            self, *args, **kwargs, **dict_only_keys(shortcuts, used_shortcuts)
#        )
#        bound.apply_defaults()
#        return func(*bound.args, **bound.kwargs)
#
#    return wrapper
#


class RaiBase(Boxes):
    def add_float_arg(self, name, default):
        self.argparser.add_argument(
            f"--{name}",
            action="store",
            type=float,
            default=default,
        )

    def add_str_arg(self, name, default):
        self.argparser.add_argument(
            f"--{name}",
            action="store",
            type=str,
            default=default,
        )

    @property
    def finger_hole_width(self) -> float:
        return self.finger_joint_settings.width + self.finger_joint_settings.play

    @property
    def finger_joint_settings(self):
        return FingerJointSettings(
            self.thickness, relative=True, **self.edgesettings.get("FingerJoint", {})
        )

    def make_standard_finger_joint_settings(self):
        return FingerJointSettings(
            relative=True,
            thickness=self.thickness,
            **self.edgesettings.get("FingerJoint", {}),
        )

    def make_angled_finger_joint_settings(self, angle_deg):
        return FingerJointSettings(
            relative=True,
            thickness=self.thickness,
            **self.edgesettings.get("FingerJoint", {}),
            angle=angle_deg,
        )

    def random_color(self):
        self.ctx.set_source_rgb(*random_color())

    def show_cc(self, i):
        c = random_color()
        self.ctx.set_source_rgb(*c)
        self.text(str(i), color=c, fontsize=5)
        self.circle(0, 0, r=1)

    def wall_builder(self, label):
        return WallBuilder(self, label=label)

    @contextlib.contextmanager
    def moved(self, move, bbox: BBox, label=None):
        assert isinstance(bbox, BBox)

        self.move(bbox.width, bbox.height, move, before=True)
        self.moveTo(-bbox.minx, -bbox.miny)
        yield

        self.move(bbox.width, bbox.height, move, label=label)

    def ystack(self, *elements):
        if len(elements) == 1 and isinstance(elements[0], Iterable):
            elements = elements[0]
        return self.stack(elements, orient="y")

    def xstack(self, *elements):
        if len(elements) == 1 and isinstance(elements[0], Iterable):
            elements = elements[0]
        return self.stack(elements, orient="x")

    def stack(
        self,
        elements,
        orient,
    ):
        assert orient in "xy"
        arranged = []

        elements = list(elements)

        for e in elements:
            # arrange all elements to be (0,0)-aligned
            e = e.translate(coord(-e.bbox.minx, -e.bbox.miny))

            if arranged:
                bbox = Element.union(self, arranged).bbox
                if orient == "y":
                    prev = coord(0, bbox.maxy + self.spacing)
                if orient == "x":
                    prev = coord(bbox.maxx + self.spacing, 0)
            else:
                prev = coord(0, 0)
            arranged.append(e.translate(prev))
        return Element.union(self, arranged)

    def build_element_grid(
        self,
        factory,
        nx=1,
        ny=1,
    ):
        return self.ystack(
            self.xstack(factory(xi, yi) for xi in range(nx)) for yi in range(ny)
        )

    def hole(self, x, y=None, r=0.0, d=0.0, tabs=0):
        """Allow invoking as hole(numpy coordinate)"""
        if y is None and isinstance(x, np.ndarray):
            x, y = x.astype(float)
        assert y is not None
        super().hole(x, y, r, d, tabs)

    def moveTo(self, x, y=0.0, degrees=0):
        """Allow invoking as moveTo(numpy coordinate)"""
        if isinstance(x, np.ndarray):
            degrees = y
            x, y = x.astype(float)
        super().moveTo(x, y, degrees)


@dataclasses.dataclass
class BBox:
    minx: float
    miny: float
    maxx: float
    maxy: float

    @property
    def width(self):
        return self.maxx - self.minx

    @property
    def center_x(self):
        return self.minx + self.width / 2

    @property
    def center_y(self):
        return self.miny + self.height / 2

    @property
    def height(self):
        return self.maxy - self.miny

    def shift(self, d: np.array):
        x, y = d.astype(float)
        return BBox(
            minx=self.minx + x,
            miny=self.miny + y,
            maxx=self.maxx + x,
            maxy=self.maxy + y,
        )

    @staticmethod
    def combine(bboxes):
        bboxes = list(bboxes)
        return BBox(
            minx=min(b.minx for b in bboxes),
            miny=min(b.miny for b in bboxes),
            maxx=max(b.maxx for b in bboxes),
            maxy=max(b.maxy for b in bboxes),
        )


COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    # (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]

# class HalfFingerEdge(FingerJointEdge):
#    def fingerLength(self, angle: float) -> tuple[float, float]:
#        length, recess = super().fingerLength(angle)
#        return length / 2, recess


###@dataclasses.dataclass
###class WallItem:
###    section: Section
###
###    @property
###    def length(self):
###        return self.section.length
###
###    @property
###    def edge(self):
###        return self.section.edge
###
###    @property
###    def text(self):
###        return self.section.text
###
###    # start position
###    start: np.array
###    # end position
###    end: np.array
###
###    @property
###    def angle(self):
###        delta = self.end - self.start
###        angle = np.degrees(np.arctan2(delta[1], delta[0]))
###        return angle % 360
###
###    @property
###    def center(self):
###        return (self.start + self.end) / 2
###
###
###class Compound:
###    def __init__(self, sections: Iterable[Section]):
###        self.sections = list(sections)
###
###    @property
###    def length(self):
###        return sum(s.length for s in self.sections)
###
###    @classmethod
###    def intersperse(
###        cls,
###        sep: Section | Iterable[Section],
###        items: Iterable[Section],
###        start: bool,
###        end: bool,
###    ) -> Compound:
###        if isinstance(sep, Section):
###            sep = itertools.repeat(sep)
###        return cls(list(intersperse(sep, items, start=start, end=end)))
###
###    def __reversed__(self):
###        return Compound(reversed(self.sections))
###
###    def __iter__(self):
###        return iter(self.sections)
###
###    def __getitem__(self, key):
###        """implement index access and slicing"""
###        assert isinstance(key, (int, slice))
###        return self.sections[key]
###
###
###class Section:
###    def __init__(self, length, edge, text=None):
###        if isinstance(length, np.float64):
###            length = float(length)
###        assert isinstance(length, (int, float))
###
###        assert (isinstance(edge, str) and len(edge) == 1) or isinstance(edge,BaseEdge)
###        # assert edge in "efFaAbB", f"unknown {edge=}"
###
###        self.length = length
###        self.edge = edge
###        self.text = text


def intersperse(sep, items, start: bool = False, end: bool = False):
    """
    Examples:
      intersperse([1,2,3,4,5,6,7], [10,20,30], start=True, end=True)
        -> 1,10,2,20,3,30,4

      intersperse([1,2,3], [10,20,30], start=True, end=True)
        -> 10,1,20,2,30
    """
    # sep_iter = iter(sep)  # iterate sep so we can pull next-sep each time
    items_iter = iter(items)

    first = True
    for x in items_iter:
        if first and start:
            yield sep  # next(sep_iter)

        if not first:
            yield sep  # next(sep_iter)
        yield x
        first = False

    if end and not first:
        yield sep  # next(sep_iter)


@dataclasses.dataclass
class Turn:
    angle: float


@dataclasses.dataclass
class Edge:
    length: float
    edge_type: str
    text: str | None = None


def Plain(*args, **kwargs):
    return Edge(*args, **kwargs, edge_type="e")


@dataclasses.dataclass
class MultiEdge:
    edges: list[Edge]


class Close:
    pass


Command = Turn | Edge | Close | MultiEdge | list["Command"]


def test_consolidate1():
    w = WallBuilder(boxes=None, label="test").plain(10)
    assert_that(w.commands, contains_exactly(Edge(10, "e")))
    assert_that(w.consolidate(), contains_exactly((Edge(10, "e"), Turn(angle=0))))


def test_consolidate2():
    w = WallBuilder(boxes=None, label="test").plain(10).turn(50)
    assert_that(w.commands, contains_exactly(Edge(10, "e"), Turn(50)))
    assert_that(w.consolidate(), contains_exactly((Edge(10, "e"), Turn(angle=50))))


def test_consolidate3():
    w = WallBuilder(boxes=None, label="test").plain(10).turn(30).turn(30).plain(20)
    assert_that(w.commands, contains_exactly(Edge(10, "e"), Turn(60), Edge(20, "e")))
    assert_that(
        w.consolidate(),
        contains_exactly((Edge(10, "e"), Turn(60)), (Edge(20, "e"), Turn(0))),
    )


def _direction_vector(angle):
    return np.array([cos(radians(angle)), sin(radians(angle))])


@dataclasses.dataclass
class WallBuilder:
    """
    angle: current bearing in radians
    """

    boxes: Boxes
    label: str
    angle: float = 0
    commands: list[Command] = dataclasses.field(default_factory=list)
    _closed: bool = False

    @property
    def position(self):
        position, angle = np.array([0, 0], dtype=float), 0
        for c in self.commands:
            if isinstance(c, Turn):
                angle = (angle + c.angle) % 360
                continue
            if not isinstance(c, Edge):
                raise ValueError(f"Unsupported {c = }")
            position += c.length * _direction_vector(angle)
        return position

    @property
    def vector(self) -> np.array:
        return coord(cos(self.angle), sin(self.angle))

    def slot(self, depth: float, length: float):
        """Start: facing into slot. End: facing continuation."""
        assert isinstance(depth, (int, float))
        assert isinstance(length, (int, float))
        self.plain(depth).turn(-90).plain(length).turn(-90).plain(depth).turn(-90)

    def plain(self, length: float) -> WallBuilder:
        return self.edge(length, PLAIN)

    def edge(self, length: float, edge_type: str) -> WallBuilder:
        return self.add(Edge(length, edge_type))

    def turn(self, angle: float) -> WallBuilder:
        return self.add(Turn(angle))

    def _add_one(self, command: Command):
        if any(isinstance(i, Close) for i in self.commands):
            raise ValueError("Cannot add to closed WallBuilder")

        match command:
            case list():
                self.add(*command)

            case Turn(angle) if self.commands and isinstance(self.commands[-1], Turn):
                self.commands[-1].angle += angle

            case Turn() | Edge() | Close():
                self.commands.append(command)

            case _:
                raise ValueError(f"Unhandled {command=}")

    def add(self, *commands: Command | Iterable[Command]) -> WallBuilder:
        for c in commands:
            self._add_one(c)
        return self

    def close(self):
        return self.add(Close())

    def consolidate(self):
        edge, turn = None, None

        def _flush():
            nonlocal edge, turn
            if edge is not None:
                yield (edge, turn or Turn(0))
                edge, turn = None, None
            else:
                assert turn is None, f"{turn = } but should be None when flushing"

        close = None

        for c in self.commands:
            if isinstance(c, Edge):
                yield from _flush()
                edge = c
            elif isinstance(c, Turn):
                assert turn is None
                turn = c
            elif isinstance(c, Close):
                assert not close
                close = c
            else:
                raise ValueError(f"Unhandled {x=}")

        yield from _flush()

    def borders_edges(self):
        borders, edges = [], []
        for edge_turn in self.consolidate():
            if isinstance(edge_turn, Close):
                edges.append(None)
                continue
            edge, turn = edge_turn
            borders.extend((float(edge.length), float(turn.angle)))
            edges.append(edge.edge_type)

        return (
            self.boxes._closePolygon(borders),
            edges,
        )

    def rendering_table(self):
        """display borders as passed to polygonWall."""
        rows = []
        for edge_turn in self.consolidate():
            if isinstance(edge_turn, Close):
                rows.append(("Close", "", "", "", ""))
                continue
            edge, turn = edge_turn
            rows.append(
                (
                    f"{edge.edge_type}",
                    fmt_mm(edge.length),
                    fmt_reldeg(turn.angle),
                    edge.text,
                    edge,
                )
            )
        return tabulate.tabulate(
            rows,
            headers=["type", "length", f"{ALPHA_SIGN} {DEG_SIGN}", "text", "edge"],
            showindex="always",
            tablefmt="presto",
        )

    # def surround(self, lengths, last_angle, positive_edge, gap_size, gap_edge):
    #    lengths = list(lengths)
    #    for l in lengths:
    #        self.add(gap_size, 0, gap_edge)
    #        self.add(l, 0, positive_edge)
    #    self.add(gap_size, last_angle, gap_edge)

    @property
    def debug(self) -> bool:
        return self.boxes.debug

    @property
    def bbox(self) -> BBox:
        borders, edges = self.borders_edges()
        edges = [self.boxes.edges.get(edge, edge) for edge in edges]
        # print(f"{borders = }")
        # print(f"{edges = }")
        minx, miny, maxx, maxy = self.boxes._polygonWallExtend(borders, edges)
        return BBox(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    def _wall_debug_callback(self, i):
        c = random_color()
        self.boxes.ctx.set_source_rgb(*c)
        self.boxes.text(str(i), color=c, fontsize=5)
        self.boxes.circle(0, 0, r=1)

    def render(
        self,
        move=None,
        callback=None,
        turtle: bool = True,
        correct_corners: bool = True,
    ):
        # print(f"Rendering WallBuilder {self.label}:")
        callback = self._wall_debug_callback if self.debug else None

        position, angle = np.array([0, 0], dtype=float), 0
        for c in self.commands:
            if isinstance(c, Close):
                continue  # ...
            if isinstance(c, Turn):
                angle = (angle + c.angle) % 360
                continue
            if not isinstance(c, Edge):
                raise ValueError(f"Unsupported {c = }")

            delta = c.length * _direction_vector(angle)
            middle = position + (delta / 2)
            position += delta

            # draw text in middle of item
            x, y = middle.astype(float)
            if self.debug:
                # make the angle always upright
                a = angle
                if 90 < a < 270:
                    a -= 180
                self.boxes.text(
                    f"{(c.text or '')} {fmt_mm(c.length)}",
                    x=x,
                    y=y,
                    angle=a,
                    fontsize=3,
                    align="center bottom",
                )

        borders, edges = self.borders_edges()
        self.boxes.polygonWall(
            borders,
            edge=edges,
            correct_corners=correct_corners,
            callback=callback,
            move=move,
            turtle=turtle,
        )

        self.boxes.text(
            self.label,
            x=self.bbox.center_x,
            y=self.bbox.center_y,
            fontsize=5,
            align="middle center",
            color=Color.ANNOTATIONS,
        )

    @contextlib.contextmanager
    def moved(self, move):
        with self.boxes.moved(move, self.bbox, label=self.label):
            yield
            self.render()


@dataclasses.dataclass
class Element:
    position: np.array  # dx,dy
    bbox: BBox
    render: list[callable]
    boxes: Boxes

    @property
    def height(self):
        return self.bbox.height

    @property
    def width(self):
        return self.bbox.width

    def __post_init__(self):
        assert isinstance(self.position, np.ndarray)

    @classmethod
    def from_item(cls, obj):
        match obj:
            case Element():
                return obj
            case WallBuilder():
                return cls(
                    position=coord(0, 0),
                    bbox=obj.bbox,
                    render=[obj.render],
                    boxes=obj.boxes,
                )
            case _:
                raise ValueError(f"Unsupported {obj = }")

    def add_render(self, render):
        self.render.append(render)
        return self

    def translate(self, d):
        return Element(
            boxes=self.boxes,
            position=self.position + d,
            bbox=self.bbox.shift(d),
            render=self.render,
        )

    def do_render(self):
        x, y = self.position.astype(float)
        for c in self.render:
            with self.boxes.saved_context():
                self.boxes.moveTo(x, y)
                c()

    @classmethod
    def union(cls, boxes, elements):
        elements = list(elements)
        bbox = BBox.combine(e.bbox for e in elements)

        def render():
            for e in elements:
                e.do_render()

        return Element(position=coord(0, 0), bbox=bbox, render=[render], boxes=boxes)

    def mirror(self):
        def render():
            self.boxes.ctx.scale(-1, 1)
            self.do_render()

        return Element(
            position=self.position,
            bbox=self.bbox,
            render=[render],
            boxes=self.boxes,
        )

    def is_part(self, name: str):
        self.render = [lambda: self.boxes.ctx.new_part(name)] + self.render
        return self

    def close_part(self, name: str):
        self.render.append(lambda: self.boxes.ctx.new_part(name))
        return self


import sys

import pytest
from hamcrest import assert_that, equal_to

# @pytest.mark.parametrize(
#    "input,start,end,expected",
#    [
#        ("abc", True, True, "_a_b_c_"),
#        ("a", True, True, "_a_"),
#        ("", True, True, ""),
#        ("abc", False, False, "a_b_c"),
#        ("a", False, False, "a"),
#        ("", False, False, ""),
#    ],
# )
# def test_intersperse_repeat_sep(input, start, end, expected):
#    assert_that(
#        "".join(intersperse(itertools.repeat("_"), input, start=start, end=end)),
#        equal_to(expected),
#    )


class TestBox:
    @property
    def shortcuts(self):
        return {"x": "foo", "y": "bar"}

    @inject_shortcuts
    def fn(self, x, y, arg, default="default"):
        return f"{x=} {y=} {arg=} {default=}"


def test_inject_nothing_passed():
    """Fill only one positional arg."""
    box = TestBox()
    result = box.fn("beep")
    assert_that(result, equal_to("x='foo' y='bar' arg='beep' default='default'"))


def test_too_many_pos_args():
    """
    If we pass more positional arguments than parameters available,
    after injection, we should get a TypeError.
    """
    box = TestBox()
    with pytest.raises(TypeError, match="got 4 positional"):
        box.fn(1, 2, 3, 4)


from pathlib import Path

if (retcode := pytest.main([Path(__file__).resolve()])) != 0:
    sys.exit(retcode)
