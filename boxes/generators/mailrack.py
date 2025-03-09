"""
scripts/boxes MailRack \
        --sh '50*3' \
        --sx '50*3' \
        --debug=True

"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import math
import random
from math import cos, radians, sin, tan
from typing import Iterable, Sequence

import numpy as np

from boxes import Boxes, Color, argparseSections, boolarg, restore
from boxes.edges import FingerJointEdge, FingerJointEdgeCounterPart, FingerJointSettings

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def shift_bbox(bbox, x, y):
    minx, miny, maxx, maxy = bbox
    return minx + x, miny + y, maxx + x, maxy + y


def combine_bboxes(bboxes):
    minx = min(minx for minx, _, _, _ in bboxes)
    miny = min(miny for _, miny, _, _ in bboxes)
    maxx = max(maxx for _, _, maxx, _ in bboxes)
    maxy = max(maxy for _, _, _, maxy in bboxes)
    return minx, miny, maxx, maxy


# class HalfFingerEdge(FingerJointEdge):
#    def fingerLength(self, angle: float) -> tuple[float, float]:
#        length, recess = super().fingerLength(angle)
#        return length / 2, recess


class MailRack(Boxes):
    def __init__(self):
        super().__init__()

        self.buildArgParser("sh", "sx")
        self.argparser.add_argument(
            "--angle_deg",
            action="store",
            type=float,
            default=60,
        )
        self.argparser.add_argument(
            "--front_length",
            action="store",
            type=float,
            default=100,
        )
        self.argparser.add_argument(
            "--floor_depth",
            action="store",
            type=float,
            default=30,
        )

    ANGLED_POS = "b"
    ANGLED_NEG = "B"

    def render(self):
        # w = WallBuilder(self)
        # d = 43.30127018922193
        # w.add(20, 60, "e")
        # w.add(d, 120, "f")
        # w.add(20, 60, "e")
        # w.add(d, 90, "F")
        # w.render()
        ## edge called with 43.30127018922193
        s = FingerJointSettings(
            relative=True,
            thickness=self.thickness,
            **self.edgesettings.get("FingerJoint", {}),
            angle=self.angle_deg,
        )
        self.edges[self.ANGLED_POS] = FingerJointEdge(self, s)
        self.edges[self.ANGLED_NEG] = FingerJointEdgeCounterPart(self, s)

        # self.back(move="right")
        # for i in range(len(self.sx) + 1):
        #    move = "right"
        #    if i == len(self.sx):
        #        move += " up"
        #    self.outer_side(move=move)
        # self.floor(move="left")
        self.floor_pieces(move="left")
        self.hole(0, 0, 3)
        # self.front_pieces(move="left")
        # self.first_front(move="left")

    def show_cc(self, i):
        c = COLORS[i % len(COLORS)]
        self.ctx.set_source_rgb(*c)
        self.text(str(i), color=c, fontsize=5)
        self.circle(0, 0, r=2)

    @property
    def sin_a(self):
        return sin(radians(self.angle_deg))

    @property
    def cos_a(self):
        return cos(radians(self.angle_deg))

    @property
    def tan_a(self):
        return tan(radians(self.angle_deg))

    def last_height(self):
        x = np.array([self.floor_depth, 0]) + self.front_length * np.array(
            [self.cos_a, self.sin_a]
        )
        vec = np.array([-self.sin_a, self.cos_a])
        x += vec * x[0] / (-vec[0])
        return float(x[1])

    def outer_side(self, move):
        w = WallBuilder(self)
        w.add((self.floor_depth / 2, self.floor_depth / 2), self.angle_deg, "F")
        w.add(self.front_length, 90, "f")
        for sh in self.sh:
            w.add(sh * self.cos_a, -90, "e")
            d = sh * self.sin_a
            w.add(d, 90, "F")
            print(f"{d = }")
        w.add(float(w.current[0]) / self.sin_a, 180 - self.angle_deg, "e")
        w.add(reversed(self.sh_effective), 0, "f")
        with self.moved(move=move, bbox=w.bbox(), label="outer_side"):
            # self.hole(w.current[0], w.current[1], 3)
            with self.saved_context():
                self.moveTo(0, self.thickness)  # <--- fix this
                for h in self.sh:
                    self.moveTo(0, h)
                    with self.saved_context():
                        self.fingerHolesAt(0, 0, self.floor_depth / 2, 0)
                        self.fingerHolesAt(
                            self.floor_depth / 2, 0, self.floor_depth / 2, 0
                        )
                        self.fingerHolesAt(
                            self.floor_depth,
                            0,
                            self.front_length - self.sin_a * (sh + self.thickness),
                            self.angle_deg,
                        )

            w.render(callback=self.show_cc, turtle=True, correct_corners=False)

    def floor(self, move):
        w = WallBuilder(self)
        w.add(self.sx, 90, self.ANGLED_NEG)  # <- side towards front
        w.add((self.floor_depth / 2, self.floor_depth / 2), 90, "f")
        w.add(reversed(self.sx), 90, edge="f")
        w.add((self.floor_depth / 2, self.floor_depth / 2), 0, "f")

        with self.moved(move=move, bbox=w.bbox(), label="floor"):

            with self.saved_context():
                for x in self.sx[:-1]:
                    self.fingerHolesAt(x, 0, self.floor_depth, 90)
                    self.moveTo(x, 0)
            w.render(callback=self.show_cc, turtle=True)

    def floor_pieces(self, move):
        # edges = self.get_edges()
        # edges = [self.boxes.edges[e] for e in edges]
        # borders = self.boxes._closePolygon(self.get_borders())
        # minx, miny, maxx, maxy = self.boxes._polygonWallExtend(borders, edges)
        # return minx, miny, maxx, maxy

        elements = []

        yshift = 0
        for j in range(len(self.sh)):
            xshift = 0
            row_elements = []
            for i, x in enumerate(self.sx):
                w = WallBuilder(self)
                w.add(x, 90, self.ANGLED_NEG)  # <- side towards front
                w.add(self.floor_depth / 2, 0, ("f" if i == len(self.sx) - 1 else "e"))
                w.add(self.floor_depth / 2, 90, "f")
                w.add(x, 90, "f")
                w.add(self.floor_depth / 2, 0, ("f" if i == 0 else "e"))
                w.add(self.floor_depth / 2, 90, "f")
                minx, miny, maxx, maxy = w.bbox()
                print(f"{minx=} {miny=} {maxx=} {maxy=}")

                row_elements.append((xshift, yshift, w))
                # with self.saved_context():
                #    self.moveTo(xshift, yshift)
                #    with self.saved_context():
                #        self.moveTo(-minx, -miny)
                #        w.render(callback=self.show_cc, turtle=True)

                # sligthly overlap - we can do this
                xshift += maxx - minx - self.thickness * 0.7

            row_bboxes = [
                shift_bbox(w.bbox(), xshift, yshift)
                for xshift, yshift, w in row_elements
            ]
            maxy = max(maxy for _, _, _, maxy in row_bboxes)
            miny = min(miny for _, miny, _, _ in row_bboxes)
            elements.extend(row_elements)
            yshift += maxy - miny + self.spacing

        bboxes = [
            shift_bbox(w.bbox(), xshift, yshift) for xshift, yshift, w in elements
        ]
        combined_bbox = combine_bboxes(bboxes)
        with self.moved(move=move, bbox=combined_bbox):
            for xshift, yshift, w in elements:
                with self.saved_context():
                    self.moveTo(xshift, yshift)
                    w.render(callback=self.show_cc, turtle=True)

    def first_front(self, move):
        w = WallBuilder(self)
        w.add(self.sx, 90, "e")
        w.add(self.front_length, 90, "F")
        w.add(reversed(self.sx), 90, self.ANGLED_POS)
        w.add(self.front_length, 90, "F")

        with self.moved(move=move, bbox=w.bbox(), label="first_front"):
            w.render(callback=self.show_cc, turtle=True)

            with self.saved_context():
                for x in self.sx[:-1]:
                    self.moveTo(x, 0)
                    self.fingerHolesAt(0, 0, self.front_length, 90)

    def front_pieces(self, move):

        # TODO: include self.angle for floor-front joint
        # (here and also in floor)

        for i, sh in enumerate(self.sh):
            for j, x in enumerate(self.sx):
                w = WallBuilder(self)
                a = sh * self.sin_a
                b = self.front_length - a
                print(f"{a = }")
                w.add(x, 90, "e")
                w.add(a, 0, ("f" if j == len(self.sx) - 1 else "e"))
                w.add(b, 90, "f")
                w.add(x, 90, self.ANGLED_POS)
                w.add(b, 0, ("f" if j == 0 else "e"))
                w.add(a, 90, "f")

                with self.moved(move=move, bbox=w.bbox(), label=f"front piece {i}/{j}"):
                    w.render(callback=self.show_cc, turtle=True)

    @property
    def sh_effective(self):
        return self.sh + [self.last_height()]

    def back(self, move):
        w = WallBuilder(self)
        print(self.last_height())
        w.add(self.sx, 90, "F")
        w.add(self.sh_effective, 90, "f")
        w.add(reversed(self.sx), 90, "e")
        w.add(reversed(self.sh_effective), 90, "f")

        with self.moved(move=move, bbox=w.bbox(), label="back"):
            # horizontal finger holes
            with self.saved_context():
                for h in self.sh:
                    self.moveTo(0, h)
                    with self.saved_context():
                        for x in self.sx:
                            self.fingerHolesAt(0, 0, x, 0)
                            self.moveTo(x, 0)

            # vertical finger holes
            with self.saved_context():
                for x in self.sx[:-1]:
                    self.moveTo(x, 0)
                    with self.saved_context():
                        for h in self.sh_effective:
                            self.fingerHolesAt(0, 0, h, 90)
                            self.moveTo(0, h)

            w.render(callback=self.show_cc, turtle=True)

    @contextlib.contextmanager
    def moved(self, move, bbox, label=None):
        minx, miny, maxx, maxy = bbox
        width = maxx - minx
        height = maxy - miny

        if self.move(width, height, move, before=True):
            return True
        self.moveTo(-minx, -miny)
        yield

        self.move(width, height, move, label=label)


@dataclasses.dataclass
class Instr:
    length: float
    angle: float
    edge: str


@dataclasses.dataclass
class WallBuilder:
    boxes: Boxes
    instr: list[Instr] = dataclasses.field(default_factory=list)
    current: np.array = dataclasses.field(default_factory=lambda: np.zeros(2))
    angle: float = 0

    def add(self, length: float | tuple[float] | list[float], angle: float, edge: str):
        if isinstance(length, (int, float)):
            self.current += np.array(
                (
                    length * cos(radians(self.angle)),
                    length * sin(radians(self.angle)),
                )
            )
            self.angle += angle
            self.instr.append(Instr(length=length, angle=angle, edge=edge))
            return

        if isinstance(length, Iterable):
            length = list(length)
            for l in length[:-1]:
                self.add(l, 0, edge)
            self.add(length[-1], angle, edge)
            return

        raise ValueError(f"Invalid length: {length}")

    def get_borders(self):
        borders = []
        for instr in self.instr:
            borders.extend((instr.length, instr.angle))
        return borders

    def get_edges(self):
        return [i.edge for i in self.instr]

    def total_width(self):
        minx, _, maxx, _ = self.bbox()
        return maxx - minx

    def bbox(self):
        edges = self.get_edges()
        edges = [self.boxes.edges[e] for e in edges]
        borders = self.boxes._closePolygon(self.get_borders())
        minx, miny, maxx, maxy = self.boxes._polygonWallExtend(borders, edges)
        return minx, miny, maxx, maxy

    def total_height(self):
        _, miny, _, maxy = self.bbox()
        return maxy - miny

    def render(self, move=None, callback=None, turtle=False, correct_corners=True):

        print("WallBuilder.render")
        for i, instr in enumerate(self.instr):
            msg = f"  {i=} length={instr.length}"
            if instr.angle:
                msg += f" angle={instr.angle}"
            print(msg)
        self.boxes.polygonWall(
            self.get_borders(),
            edge=self.get_edges(),
            correct_corners=correct_corners,
            callback=callback,
            move=move,
            turtle=turtle,
        )
