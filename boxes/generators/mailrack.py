"""
scripts/boxes MailRack \
        --sh '50*3' \
        --sx '50*3' \
        --debug=True

"""

from __future__ import annotations

from math import cos, radians, sin, tan

import numpy as np

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


def undermark(s):
    return DOWN_ARROW + s + DOWN_ARROW


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
        return dict(
            f=self.finger_hole_width,
            sx=self.sx,
            sh=self.sh,
            alpha_deg=self.alpha_deg,
            alpha_rad=self.alpha_rad,
            d=self.floor_depth,
            a=self.side_angled_length,
            sin_a=sin(self.alpha_rad),
            cos_a=cos(self.alpha_rad),
            tan_a=tan(self.alpha_rad),
            angle_rad=self.alpha_rad,
        )

    def setup(self):
        s = self.make_angled_finger_joint_settings(self.alpha_deg)
        self.edges |= {
            ANGLED_POS: FingerJointEdge(self, s),
            ANGLED_NEG: FingerJointEdgeCounterPart(self, s),
        }

    def render(self):
        self.setup()
        # self.back(move="right")  # <- ok
        # self.floor_pieces(move="right")
        self.side(move="right")

    @inject_shortcuts
    def side(self, d, a, f, alpha_deg, move, sh, cos_a, sin_a, tan_a, angle_rad):

        # horizontal = self.separated(sx, xs_name="sx", edge=FINGER_COUNTER)
        # vertical = self.separated(sh, xs_name="sh", edge=FINGER_COUNTER)
        w = (
            self.wall_builder("side")
            .add(d, alpha_deg, FINGER, text=undermark("floor=d"))
            .add(a, 90, FINGER_COUNTER, text=undermark("front=a"))
            # .add(horizontal.length, 90, PLAIN)
            # .add(reversed(vertical), 90)
        )

        # draw some helpers
        # first_cusp = w.position

        # Used to draw edges of finger hole area
        zigzag_corners = []

        # orthogonal, zig-zaggy side
        for i, h in enumerate(sh):
            if i > 0:
                h += f
            zig, zag = coord(cos_a, sin_a) * h
            w.add(zig, -90, PLAIN, text=undermark("zig"))
            zigzag_corners.append((w.position, zag))
            w.add(
                zag,
                90,
                FINGER_COUNTER,
                text=undermark(f"zag"),
            )

        # now go all the way from current x to x=0
        distance = w.position[0] / sin_a
        w.add(distance, 180 - alpha_deg, PLAIN, text=undermark("topside"))

        # now all the way down.
        y_need = w.position[1] - sum(sh) - len(sh) * f
        eff_heights = sh + [y_need]
        w.add(
            Compound.intersperse(
                Section(f, PLAIN, text="f"),
                (Section(h, FINGER, text="sh") for h in reversed(eff_heights)),
                how="inner",
            ),
            0,
        )

        with w.moved(move=move):
            # Draw edges of finger hole area
            for zigzag_corner, zag in zigzag_corners:
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

            ### cusp point debug
            # for point in [
            #    first_cusp,
            #    # first_cusp + coord(0, f),
            # ]:
            #    color = (128, 128, 128)
            #    with self.saved_context():
            #        self.ctx.set_source_rgb(*color)

            #        self.moveTo(point, 90)
            #        self.circle(r=2)
            #        self.circle(r=3)
            #        # self.ctx.line_to(0, sh[0])
            #        self.moveTo(sh[0], 0)
            #        self.corner(90 + alpha_deg)
            #        self.edge(a)
            #        self.corner(-alpha_deg)
            #        self.edge(d)
            #        self.ctx.stroke()
            #        # self.ctx.moveTo(0, sh[0], 180 - alpha)
            #        # self.ctx.stroke()
            #        # self.ctx.line_to(0, sh[0])

    def circle(self, x=0, y=0, r=1):
        """Sets defaults."""
        super().circle(x=x, y=y, r=r)

    @property
    def alpha_rad(self):
        return radians(self.alpha_deg)

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

    @property
    def sh_effective(self):
        return self.sh + [self.last_height()]

    @inject_shortcuts
    def separated(self, f, xs, edge, xs_name) -> Compound:
        return Compound.intersperse(
            Section(f, PLAIN, text="f"),
            (Section(x, edge, text=f"{xs_name}{i}") for i, x in enumerate(xs)),
            how="outer",
        )

    @inject_shortcuts
    def back(self, f, sx, sh, move):
        horizontal = self.separated(sx, xs_name="sx", edge=FINGER_COUNTER)
        vertical = self.separated(sh, xs_name="sh", edge=FINGER_COUNTER)
        w = (
            self.wall_builder("back")
            .add(horizontal, 90)
            .add(vertical, 90)
            .add(horizontal.length, 90, PLAIN)
            .add(reversed(vertical), 90)
        )
        with w.moved(move=move):
            pass

        # with self.moved(move=move, bbox=w.bbox, label="back"):
        #    # horizontal finger holes
        #    with self.saved_context():
        #        self.moveTo(self.thickness, self.thickness / 2)
        #        for h in self.sh:
        #            self.moveTo(0, self.thickness + h)
        #            with self.saved_context():
        #                for x in self.sx:
        #                    self.fingerHolesAt(0, 0, x, 0)
        #                    self.moveTo(x, 0)
        #                    self.moveTo(self.thickness, 0)

        #    # vertical finger holes
        #    with self.saved_context():
        #        self.moveTo(self.thickness / 2, self.thickness)
        #        for x in self.sx[:-1]:
        #            self.moveTo(x + self.thickness, 0)
        #            with self.saved_context():
        #                for h in self.sh_effective:
        #                    self.fingerHolesAt(0, 0, h, 90)
        #                    self.moveTo(0, h)
        #                    self.moveTo(0, self.thickness)

        #    w.render(callback=self.show_cc, turtle=True)
