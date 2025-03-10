"""
scripts/boxes MailRack \
        --sh '50*3' \
        --sx '50*3' \
        --debug=True

"""

from __future__ import annotations

from math import cos, radians, sin, sqrt, tan

import numpy as np
from hamcrest import assert_that, close_to

from boxes import argparseSections, boolarg, restore
from boxes.edges import FingerJointEdge, FingerJointEdgeCounterPart, FingerJointSettings
from boxes.generators.raibase import (
    ALPHA_SIGN,
    Compound,
    RaiBase,
    Section,
    coord,
    inject_shortcuts,
)

PLAIN = "e"
FINGER = "f"
FINGER_COUNTER = "F"
ANGLED_POS = "b"
ANGLED_NEG = "B"
RIGHT_ARROW = "→"
DOWN_ARROW = "↓"


def mark(s):
    return DOWN_ARROW + s + DOWN_ARROW


def make_sections(xs, name, edge):
    return [Section(x, edge, text=f"{name}{i}") for i, x in enumerate(xs)]


class MailRack(RaiBase):
    def __init__(self):
        super().__init__()

        # sx: inner widths of sections

        self.buildArgParser("sh", "sx")
        self.add_float_arg("alpha_deg", 60)
        self.add_float_arg("side_angled_length", 100)
        self.add_float_arg("floor_depth", 30)  # 'd'

    @property
    def shortcuts(self):
        alpha_rad = radians(self.alpha_deg)
        f = self.finger_hole_width
        sh = self.sh
        d = self.floor_depth
        a = self.side_angled_length
        sin_a, cos_a = sin(alpha_rad), cos(alpha_rad)

        hat_length = (d + a * cos_a) / sin_a

        # right-angle triangles:
        # 1. front length, hat length, [diagonal]
        # 2. floor depth, (hat height + spacer), [diagonal]
        hat_height = sqrt(a**2 + hat_length**2 - d**2) - f

        sheff = sh + [hat_height]

        return dict(
            f=f,
            sx=self.sx,
            sh=sh,
            alpha_deg=self.alpha_deg,
            d=d,
            a=a,
            sin_a=sin_a,
            cos_a=cos_a,
            angle_rad=alpha_rad,
            hat_length=hat_length,
            hat_height=hat_height,
            sheff=sheff,
            gap=Section(f, PLAIN, text="f"),
        )

    def setup(self):
        s = self.make_angled_finger_joint_settings(self.alpha_deg)
        self.edges |= {
            ANGLED_POS: FingerJointEdge(self, s),
            ANGLED_NEG: FingerJointEdgeCounterPart(self, s),
        }

    def render(self):
        self.setup()
        self.back(move="right")  # <- ok
        # self.floor_pieces(move="right")
        self.side(move="right")

    @inject_shortcuts
    def side(
        self,
        a,
        alpha_deg,
        angle_rad,
        cos_a,
        d,
        f,
        gap,
        hat_height,
        hat_length,
        move,
        sh,
        sheff,
        sin_a,
    ):
        w = self.wall_builder("side")
        w.add(d, alpha_deg, FINGER_COUNTER, text=mark("floor=d"))

        zigzags: list[tuple[float, float]] = [
            tuple(coord(cos_a, sin_a) * (h + f)) for h in sh
        ]

        #### bottom cover options
        if False:
            # Plain
            section = Section(a, FINGER_COUNTER, text=mark("front=a"))
        else:
            # Same as covers on other levels
            _zig, zag = zigzags[1]
            section = Compound(
                [
                    Section(a - zag, PLAIN, text=mark("front counterzag")),
                    Section(zag, FINGER_COUNTER, text=mark("front zag")),
                ]
            )
        assert_that(section.length, close_to(a, delta=1e-3))

        w.add(section, 90)

        # Used to draw edges of finger hole area
        zigzag_corners = []

        # orthogonal, zig-zaggy side
        for zig, zag in zigzags:
            w.add(zig, -90, PLAIN, text=mark("zig"))
            zigzag_corners.append(w.position)
            w.add(zag, 90, FINGER_COUNTER, text=mark("zag"))

        # now go all the way from current x to x=0
        w.add(hat_length, 180 - alpha_deg, PLAIN, text=mark("topside"))

        # now all the way down.
        y_need = w.position[1] - sum(sh) - (len(sh) + 1) * f
        assert abs(y_need - hat_height) < 1e-3, f"Measured {y_need} != {hat_height}"
        w.add(
            reversed(
                Compound.intersperse(
                    gap,
                    make_sections(sheff, "sheff", FINGER),
                    start=True,
                    end=False,
                )
            ),
            90,
        )

        with w.moved(move=move):
            # Draw edges of finger hole area
            for zigzag_corner, (_zig, zag) in zip(zigzag_corners, zigzags):
                color = (0, 128, 128)
                with self.saved_context():
                    self.ctx.set_source_rgb(*color)
                    # lower stroke
                    with self.saved_context():
                        self.moveTo(zigzag_corner, 180 + alpha_deg)
                        self.edge(a - zag)
                        self.corner(-alpha_deg)
                        self.edge(d)
                        self.ctx.stroke()
                    # upper stroke
                    with self.saved_context():
                        self.moveTo(zigzag_corner, 90 + alpha_deg)
                        self.edge(f)
                        self.corner(90)
                        delta = -f * tan(angle_rad / 2)
                        print(f"{delta=}")
                        self.edge((a - zag) + delta)
                        self.corner(-alpha_deg)
                        self.edge(d + delta)
                        self.ctx.stroke()

    def circle(self, x=0, y=0, r=1):
        """Sets defaults."""
        super().circle(x=x, y=y, r=r)

    # @inject_shortcuts
    # def _floor_piece(self, sx, i):
    #    x = sx[i]
    #    d = self.floor_depth / 2
    #    return (
    #        self.wall_builder(f"floorpiece{i}")
    #        .add(x, 90, "f", text="x")
    #        .add(
    #            Compound(
    #                [
    #                    Section(d, (FINGER if i == len(sx) - 1 else PLAIN), text="d1"),
    #                    Section(d, FINGER, text="d2"),
    #                ]
    #            ),
    #            90,
    #        )
    #        .add(x, 90, ANGLED_NEG)
    #        .add(d, 0, (FINGER if i == 0 else PLAIN))
    #        .add(d, 90, FINGER)
    #    )

    # @inject_shortcuts
    # def floor_pieces(self, sx, sh, move):
    #    self.render_moved_elements(
    #        self.build_element_grid(
    #            nx=1,  # len(sx),
    #            ny=1,  # len(sh),
    #            element_factory=lambda xi, yi: self._floor_piece(xi),
    #        ),
    #        move=move,
    #    )

    @inject_shortcuts
    def back(self, f, sx, move, sheff, gap):
        xedges = Compound.intersperse(
            gap, make_sections(sx, "sx", FINGER_COUNTER), start=True, end=True
        )
        yedges = Compound.intersperse(
            gap,
            make_sections(sheff, "sheff", FINGER_COUNTER),
            start=True,
            end=False,
        )
        w = (
            self.wall_builder("back")
            .add(xedges, 90)
            .add(yedges, 90)
            .add(xedges.length, 90, PLAIN)
            .add(reversed(yedges), 90)
        )
        with w.moved(move=move):
            with self.saved_context():
                self.moveTo(f, f / 2)

                for iy, dy in enumerate(sheff):
                    with self.saved_context():
                        for ix, dx in enumerate(sx):
                            # if ix < len(sx) - 1:
                            if iy > 0:
                                self.fingerHolesAt(0, 0, dx, 0)

                            if ix > 0:
                                self.fingerHolesAt(-f / 2, f / 2, dy, 90)
                            self.moveTo(dx + f, 0)
                    self.moveTo(0, dy + f)
