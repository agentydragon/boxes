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


# class HalfFingerEdge(FingerJointEdge):
#    def fingerLength(self, angle: float) -> tuple[float, float]:
#        length, recess = super().fingerLength(angle)
#        return length / 2, recess


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

    def shift(self, x, y):
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
        s = FingerJointSettings(
            relative=True,
            thickness=self.thickness,
            **self.edgesettings.get("FingerJoint", {}),
            angle=self.angle_deg,
        )
        self.edges[self.ANGLED_POS] = FingerJointEdge(self, s)
        self.edges[self.ANGLED_NEG] = FingerJointEdgeCounterPart(self, s)

        self.back(move="right")  # <- ok

        # self.floor(move="down") <- floor/back joint OK
        self.outer_side(move="right")

        # for i in range(len(self.sx) + 1):
        #    move = "right"
        #    if i == len(self.sx):
        #        move += " up"
        #    self.outer_side(move=move)

        # self.floor_pieces(move="up")
        # self.front_pieces(move="up")

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
        edge = "e"  # "f"
        w.add((self.floor_depth / 2, self.floor_depth / 2), self.angle_deg, "e")  # "F")
        w.add(self.front_length, 90, "e")  # "f")
        for i, sh in enumerate(self.sh):
            sh2 = sh + self.thickness
            if i == 0:
                sh2 += 0.5 * self.thickness
            w.add(sh2 * self.cos_a, -90, "e")
            w.add(sh2 * self.sin_a, 90, edge)
        w.add(
            float(w.current[0]) / self.sin_a,  # + (self.cos_a * self.thickness / 2),
            180 - self.angle_deg,
            "e",
        )
        w.surround(
            reversed(self.sh_effective), 0, "e", self.thickness, "e"  # "f",
        )  # <- side towards front

        with self.moved(move=move, bbox=w.bbox, label="outer_side"):
            # self.hole(w.current[0], w.current[1], 3)
            with self.saved_context():
                self.moveTo(0, self.thickness / 2)  # <--- fix this
                for i, sh in enumerate(self.sh):
                    self.moveTo(0, sh)
                    self.moveTo(0, self.thickness)
                    with self.saved_context():
                        self.fingerHolesAt(0, 0, self.floor_depth / 2, 0)
                        self.fingerHolesAt(
                            self.floor_depth / 2, 0, self.floor_depth / 2, 0
                        )
                        sh2 = sh + self.thickness
                        if i == 0:
                            sh2 += 2 * self.thickness
                        self.fingerHolesAt(
                            self.floor_depth,
                            0,
                            self.front_length,  # - self.sin_a * sh2,
                            self.angle_deg,
                        )

            w.render(callback=self.show_cc, turtle=True, correct_corners=False)

    def floor(self, move):
        w = WallBuilder(self)
        w.surround(
            self.sx, 90, self.ANGLED_NEG, self.thickness, "e"
        )  # <- side towards front
        d = self.floor_depth / 2
        w.add((d, d), 90, "f")
        w.surround(reversed(self.sx), 90, "f", self.thickness, "e")
        w.add((d, d), 0, "f")

        self.moveTo(-self.thickness, 0)
        with self.moved(move=move, bbox=w.bbox, label="floor"):

            with self.saved_context():
                self.moveTo(self.thickness * 1.5, 0)
                for x in self.sx[:-1]:
                    self.fingerHolesAt(x, 0, self.floor_depth, 90)
                    self.moveTo(x, 0)
                    self.moveTo(self.thickness, 0)
            w.render(callback=self.show_cc, turtle=True)

    def _floor_piece(self, x, i):
        w = WallBuilder(self)
        d = self.floor_depth / 2
        w.add(x, 90, "f")
        w.add(d, 0, ("f" if i == len(self.sx) - 1 else "e"))
        w.add(d, 90, "f")
        w.add(x, 90, self.ANGLED_NEG)  # <- side towards front
        w.add(d, 0, ("f" if i == 0 else "e"))
        w.add(d, 90, "f")
        return w

    def floor_pieces(self, move):
        elems = []
        dy = 0
        for j in range(len(self.sh)):
            dx = 0
            row = []
            for i, x in enumerate(self.sx):
                w = self._floor_piece(x, i)
                row.append(w.to_element(dx, dy))
                # sligthly overlap - we can do this
                dx += w.bbox.width

                # dx -= self.thickness  # <- fully tight
                # dx -= self.thickness * 0.7

            row_bbox = BBox.combine(b for _, b, _ in row)
            elems.extend(row)
            dy += row_bbox.height + self.spacing

        with self.moved(move=move, bbox=BBox.combine(b for _, b, _ in elems)):
            for (dx, dy), _, elem in elems:
                with self.saved_context():
                    self.moveTo(dx, dy)
                    elem()

    def first_front(self, move):
        w = WallBuilder(self)
        w.add(self.sx, 90, "e")
        w.add(self.front_length, 90, "F")
        w.add(reversed(self.sx), 90, self.ANGLED_POS)
        w.add(self.front_length, 90, "F")

        with self.moved(move=move, bbox=w.bbox, label="first_front"):
            w.render(callback=self.show_cc, turtle=True)

            with self.saved_context():
                for x in self.sx[:-1]:
                    self.moveTo(x, 0)
                    self.fingerHolesAt(0, 0, self.front_length, 90)

    def _front_piece(self, sh, x, j):
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
        return w

    def front_pieces(self, move):

        # TODO: include self.angle for floor-front joint
        # (here and also in floor)

        elems = []
        dy = 0
        for i, sh in enumerate(self.sh):
            dx = 0
            row = []
            for j, x in enumerate(self.sx):
                w = self._front_piece(sh, x, j)
                row.append(w.to_element(dx, dy))
                # sligthly overlap - we can do this
                dx += w.bbox.width - self.thickness * 0.7

            row_bbox = BBox.combine(b for _, b, _ in row)
            elems.extend(row)
            dy += row_bbox.height + self.spacing

        with self.moved(move=move, bbox=BBox.combine(b for _, b, _ in elems)):
            for (dx, dy), _, elem in elems:
                with self.saved_context():
                    self.moveTo(dx, dy)
                    elem()

    @property
    def sh_effective(self):
        return self.sh + [self.last_height()]

    def back(self, move):
        w = WallBuilder(self)

        # w.add(self.sx, 90, "F")
        w.surround(self.sx, 90, "F", self.thickness, "e")
        w.surround(self.sh_effective, 90, "F", self.thickness, "e")
        # w.add(reversed(self.sx), 90, "e")
        w.surround(reversed(self.sx), 90, "e", self.thickness, "e")
        w.surround(reversed(self.sh_effective), 90, "F", self.thickness, "e")

        with self.moved(move=move, bbox=w.bbox, label="back"):
            # horizontal finger holes
            with self.saved_context():
                self.moveTo(self.thickness, self.thickness / 2)
                for h in self.sh:
                    self.moveTo(0, self.thickness + h)
                    with self.saved_context():
                        for x in self.sx:
                            self.fingerHolesAt(0, 0, x, 0)
                            self.moveTo(x, 0)
                            self.moveTo(self.thickness, 0)

            # vertical finger holes
            with self.saved_context():
                self.moveTo(self.thickness / 2, self.thickness)
                for x in self.sx[:-1]:
                    self.moveTo(x + self.thickness, 0)
                    with self.saved_context():
                        for h in self.sh_effective:
                            self.fingerHolesAt(0, 0, h, 90)
                            self.moveTo(0, h)
                            self.moveTo(0, self.thickness)

            w.render(callback=self.show_cc, turtle=True)

    @contextlib.contextmanager
    def moved(self, move, bbox: BBox, label=None):
        assert isinstance(bbox, BBox)
        width = bbox.maxx - bbox.minx
        height = bbox.maxy - bbox.miny

        if self.move(width, height, move, before=True):
            return True
        self.moveTo(-bbox.minx, -bbox.miny)
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

    def surround(self, lengths, last_angle, positive_edge, gap_size, gap_edge):
        lengths = list(lengths)
        for l in lengths:
            self.add(gap_size, 0, gap_edge)
            self.add(l, 0, positive_edge)
        self.add(gap_size, last_angle, gap_edge)

    # def interleave(self, lengths, last_angle, positive_edge, gap_size, gap_edge):
    #    lengths = list(lengths)
    #    for l in lengths[:-1]:
    #        self.add(l, 0, positive_edge)
    #        self.add(gap_size, 0, gap_edge)
    #    self.add(lengths[-1], last_angle, positive_edge)

    @property
    def bbox(self) -> BBox:
        edges = self.get_edges()
        edges = [self.boxes.edges[e] for e in edges]
        borders = self.boxes._closePolygon(self.get_borders())
        minx, miny, maxx, maxy = self.boxes._polygonWallExtend(borders, edges)
        return BBox(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

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

    def to_element(self, dx, dy):
        return (
            (dx, dy),
            self.bbox.shift(dx, dy),
            lambda: self.render(callback=self.boxes.show_cc, turtle=True),
        )
