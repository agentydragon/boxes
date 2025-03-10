from __future__ import annotations

import contextlib
import dataclasses
import functools
import inspect
import itertools
import random
from decimal import ROUND_HALF_UP, Decimal
from math import cos, radians, sin
from textwrap import indent
from typing import Iterable

import numpy as np
import tabulate

from boxes import Boxes
from boxes.edges import FingerJointSettings

DEG_SIGN = "°"
ALPHA_SIGN = "α"


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

    @property
    def finger_hole_width(self) -> float:
        return self.finger_joint_settings.width + self.finger_joint_settings.play

    @property
    def finger_joint_settings(self):
        return FingerJointSettings(
            self.thickness, relative=True, **self.edgesettings.get("FingerJoint", {})
        )

    def make_angled_finger_joint_settings(self, angle_deg):
        return FingerJointSettings(
            relative=True,
            thickness=self.thickness,
            **self.edgesettings.get("FingerJoint", {}),
            angle=angle_deg,
        )

    def random_color(self):
        self.ctx.set_source_rgb(*random.choice(COLORS))

    def show_cc(self, i):
        c = COLORS[i % len(COLORS)]
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

    def render_moved_elements(self, elems, move):
        assert isinstance(elems, Iterable)
        with self.moved(move=move, bbox=BBox.combine(e.bbox for e in elems)):
            for e in elems:
                with self.saved_context():
                    self.moveTo(*e.position.astype(float))
                    e.render()

    def build_element_grid(self, nx, ny, element_factory):
        elems = []
        pos = coord(0, 0)
        for yi in range(ny):
            pos[0] = 0
            row = []
            for xi in range(nx):
                element = Element.from_item(element_factory(xi, yi)).translate(pos)
                row.append(element)
                # sligthly overlap - we can do this
                pos += coord(element.bbox.width + self.spacing, 0)

            elems.extend(row)
            row_bbox = BBox.combine(e.bbox for e in row)
            pos += coord(0, row_bbox.height + self.spacing)
        return elems

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
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]

# class HalfFingerEdge(FingerJointEdge):
#    def fingerLength(self, angle: float) -> tuple[float, float]:
#        length, recess = super().fingerLength(angle)
#        return length / 2, recess


@dataclasses.dataclass
class WallItem:
    section: Section

    @property
    def length(self):
        return self.section.length

    @property
    def edge(self):
        return self.section.edge

    @property
    def text(self):
        return self.section.text

    # start position
    start: np.array
    # end position
    end: np.array

    @property
    def angle(self):
        delta = self.end - self.start
        angle = np.degrees(np.arctan2(delta[1], delta[0]))
        return angle % 360

    @property
    def center(self):
        return (self.start + self.end) / 2


class Compound:
    def __init__(self, sections: Iterable[Section]):
        self.sections = list(sections)

    @property
    def length(self):
        return sum(s.length for s in self.sections)

    @classmethod
    def intersperse(
        cls,
        sep: Section | Iterable[Section],
        items: Iterable[Section],
        start: bool,
        end: bool,
    ) -> Compound:
        if isinstance(sep, Section):
            sep = itertools.repeat(sep)
        return cls(list(intersperse(sep, items, start=start, end=end)))

    def __reversed__(self):
        return Compound(reversed(self.sections))

    def __iter__(self):
        return iter(self.sections)

    def __getitem__(self, key):
        """implement index access and slicing"""
        assert isinstance(key, (int, slice))
        return self.sections[key]


class Section:
    def __init__(self, length, edge, text=None):
        if isinstance(length, np.float64):
            length = float(length)
        assert isinstance(length, (int, float))

        assert isinstance(edge, str) and len(edge) == 1
        assert edge in "efFbB", f"unknown {edge=}"

        self.length = length
        self.edge = edge
        self.text = text


def intersperse(sep, items, start: bool = False, end: bool = False):
    """
    Examples:
      intersperse([1,2,3,4,5,6,7], [10,20,30], start=True, end=True)
        -> 1,10,2,20,3,30,4

      intersperse([1,2,3], [10,20,30], start=True, end=True)
        -> 10,1,20,2,30
    """
    sep_iter = iter(sep)  # iterate sep so we can pull next-sep each time
    items_iter = iter(items)

    first = True
    for x in items_iter:
        if first and start:
            yield next(sep_iter)

        if not first:
            yield next(sep_iter)
        yield x
        first = False

    if end and not first:
        yield next(sep_iter)


@dataclasses.dataclass
class WallBuilder:
    """
    angle: current bearing in radians
    """

    boxes: Boxes
    label: str
    angle: float = 0
    items: list[WallItem] = dataclasses.field(default_factory=list)
    position: np.array = dataclasses.field(default_factory=lambda: np.zeros(2))

    @property
    def vector(self) -> np.array:
        return coord(cos(self.angle), sin(self.angle))

    def add(
        self,
        what: float | Section | Compound,
        angle: float | None = None,
        edge: str | None = None,
        text: str | None = None,
    ) -> WallBuilder:
        """
        Invocation options:

        what: length ->
            use `edge`, `text` to construct a Section
            delegate to Section case, pass `angle` along.

            must have `edge` set

        what: Section ->
            use `length`, `edge`, `text` from Section
            to construct a WallItem

            must not have `edge`, `text`.

            `angle` is 0 by default

        what: list[Section] ->
            add each Section in turn.
            apply `angle` only to the last one.

            must not have `edge`, `text`.
        """

        match what:
            case int() | float() as length:
                section = Section(length, edge, text)
                return self.add(section, angle)
            case Section() as section:
                assert edge is None and text is None

                next = self.position + section.length * self.vector
                item = WallItem(
                    section=section,
                    start=self.position,
                    end=next,
                )
                print(f"  {item=}")
                self.items.append(item)
                self.angle += radians(angle)
                self.position = next
                return self
            case Compound() as compound:
                assert angle is not None
                for section in compound[:-1]:
                    self.add(section, 0)
                return self.add(compound[-1], angle)
            case _:
                raise ValueError(f"Invalid what: {what}")

    def get_borders(self):
        borders = []
        last_angle = 0
        for i, item in enumerate(self.items):
            next_item = self.items[(i + 1) % len(self.items)]
            delta = next_item.angle - last_angle
            last_angle = next_item.angle
            if delta < 0:
                # for some reason boxes.py is picky about this
                delta += 360
            borders.extend((item.length, delta))
        return borders

    def rendering_table(self):
        """display borders as passed to polygonWall."""
        borders = self.get_borders()
        edges = self.get_edges()

        rows = []
        assert len(borders) == 2 * len(self.items)
        for i, (item, edge) in enumerate(zip(self.items, edges)):
            b_length = borders[2 * i]
            b_angle = borders[2 * i + 1]
            rows.append(
                (
                    f"{fmt(b_length)}",
                    f"{fmt(b_angle, show_sign=True)}" if b_angle else "",
                    item.text,
                    edge,
                )
            )

        return tabulate.tabulate(
            rows,
            headers=["D mm", f"{ALPHA_SIGN} {DEG_SIGN}", "Text", "Edge"],
            showindex="always",
            tablefmt="presto",
        )

    def get_edges(self):
        return [i.edge for i in self.items]

    def surround(self, lengths, last_angle, positive_edge, gap_size, gap_edge):
        lengths = list(lengths)
        for l in lengths:
            self.add(gap_size, 0, gap_edge)
            self.add(l, 0, positive_edge)
        self.add(gap_size, last_angle, gap_edge)

    @property
    def bbox(self) -> BBox:
        edges = self.get_edges()
        edges = [self.boxes.edges[e] for e in edges]
        borders = self.boxes._closePolygon(self.get_borders())
        minx, miny, maxx, maxy = self.boxes._polygonWallExtend(borders, edges)
        return BBox(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    def render(self, move=None, callback=None, turtle=True, correct_corners=True):
        print(f"Rendering WallBuilder {self.label}:")

        assert callback is None, "TODO: combine callbacks"
        callback = self.boxes.show_cc

        for item in self.items:
            if not item.text:
                continue
            # draw text in middle of item
            x, y = item.center.astype(float)
            with self.boxes.saved_context():
                self.boxes.moveTo(x, y)
                angle = item.angle
                # make the angle always upright
                if 90 < angle < 270:
                    angle -= 180
                text = item.text + f"={fmt(item.length)}"
                self.boxes.text(text, fontsize=3, align="center bottom", angle=angle)

        print(indent(self.rendering_table(), "    "))

        self.boxes.polygonWall(
            self.get_borders(),
            edge=self.get_edges(),
            correct_corners=correct_corners,
            callback=callback,
            move=move,
            turtle=turtle,
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
    render: callable

    @classmethod
    def from_item(cls, obj):
        match obj:
            case Element():
                return obj
            case WallBuilder():
                return cls(
                    position=(0, 0),
                    bbox=obj.bbox,
                    render=obj.render,
                )
            case _:
                raise ValueError(f"Unsupported {obj = }")

    def translate(self, d):
        return Element(
            position=self.position + d,
            bbox=self.bbox.shift(d),
            render=self.render,
        )


##### unit tests ####

import sys

import pytest
from hamcrest import assert_that, equal_to


@pytest.mark.parametrize(
    "input,start,end,expected",
    [
        ("abc", True, True, "_a_b_c_"),
        ("a", True, True, "_a_"),
        ("", True, True, ""),
        ("abc", False, False, "a_b_c"),
        ("a", False, False, "a"),
        ("", False, False, ""),
    ],
)
def test_intersperse_repeat_sep(input, start, end, expected):
    assert_that(
        "".join(intersperse(itertools.repeat("_"), input, start=start, end=end)),
        equal_to(expected),
    )


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
