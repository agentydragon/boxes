"""
scripts/boxes TapeHolder2 \
        --debug=False \
        --preset 15leroy \
        --output=/home/agentydragon/tapeholder.svg \
        --FingerJoint_play=0.01
"""

from functools import lru_cache
from math import acos, degrees, pi, tan

import numpy as np
from tabulate import tabulate

from boxes import Color, holeCol, restore
from boxes.edges import BaseEdge, FingerJointSettings
from boxes.generators.raibase import (
    BBox,
    Edge,
    Element,
    Plain,
    RaiBase,
    Turn,
    inject_shortcuts,
)

INCH = 25.4


class DirectSlotEdge(BaseEdge):
    """Edge with 'depth' depth circular slot in the middle of its length.
    The slot is at an angle 'angle'."""

    description = "DirectSlotEdge"

    def __init__(self, boxes, depth, diameter):
        super().__init__(boxes, None)
        self.depth = depth
        self.diameter = diameter

    def __call__(self, length):
        depth = self.depth
        diameter = self.diameter
        l = length

        # with self.boxes.saved_context():
        #    self.boxes.moveTo(center, h)
        #    self.boxes.text("center", align="middle center", fontsize=7)
        #    self.boxes.hole(0, 0, 2)
        #    self.boxes.hole(0, 0, 1)

        self.edge((l - diameter) / 2)
        self.corner(90)
        self.edge(depth)

        self.corner(-180, diameter / 2)

        self.edge(depth)
        self.corner(90)
        self.edge((l - diameter) / 2)


class TapeHolder2(RaiBase):
    def _float_arg(self, name, **kwargs):
        self.argparser.add_argument(
            f"--{name}",
            action="store",
            type=float,
            **kwargs,
        )

    def __init__(self):
        super().__init__()

        self._float_arg("inner_width")
        self._float_arg("inner_diameter")
        self._float_arg("slot_diameter")
        self._float_arg("outer_diameter")
        self._float_arg("major_open_diameter")
        self._float_arg("top")
        self._float_arg("opening")
        self._float_arg("slot_length")
        self.argparser.add_argument(
            "--preset",
            action="store",
            type=str,
            default="",
        )
        self.addSettingsArgs(FingerJointSettings)

    def apply_presets(self):
        if not self.preset:
            return
        if self.preset == "test":
            rod_l = 40
            self.thickness = 3.6  # cardboard
            self.slot_diameter = 10 + 1
            self.inner_diameter = 40
            self.slot_length = 40
        elif self.preset == "test3":
            self.thickness = INCH / 8
            self.spool_hole_diameter = 20
            rod_l = 50
            self.rod_d = 5

            self.slot_diameter = 0.2 + self.rod_d  # 32 mm currently, 0.8 mm is play
            self.major_open_diameter = 20
            self.inner_diameter = 40
            self.slot_length = 30
            self.burn = 0.07  # yes for the 3mm plywood 25mm/s 45%
            self.outer_diameter = self.slot_diameter + 5
            self.top = 8
            self.opening = 1
        elif self.preset == "15leroy":
            # self.thickness = 3.8  # cardboard
            self.thickness = INCH / 8
            self.spool_hole_diameter = 40

            # thick rod:
            # rod_l = 107
            # rod_d = 32
            # d play was 0.6mm

            rod_l = 100.73
            self.rod_d = 15.78

            # add 2mm
            self.slot_diameter = 0.2 + self.rod_d  # 32 mm currently, 0.8 mm is play

            # diameter of area for spool.
            # actual measured diameter: 67mm
            self.major_open_diameter = 75
            self.inner_diameter = 95
            self.slot_length = 60
            self.burn = 0.07  # yes for the 3mm plywood 25mm/s 45%
            self.outer_diameter = self.slot_diameter + 10
            self.top = 15
            self.opening = 1
        else:
            raise Exception(f"Unknown preset: {self.preset}")
        rod_l_play = 0.6
        # inner width is the distance between the 2 inner A inserts.
        # rod includes their thickness.
        # rod can also move a bit to left/right = play
        self.inner_width = rod_l - rod_l_play - 2 * self.thickness
        assert (
            self.outer_diameter < self.spool_hole_diameter
        ), f"{self.outer_diameter} should be <40"
        self.spoolhead = self.outer_diameter - 2
        print(f"{self.inner_diameter = }")

    @property
    @lru_cache()
    def shortcuts(self):
        N = 8
        inner_diameter = self.inner_diameter
        angle_between_sides = pi * (N - 2) / N

        # dtheta = sin(angle_between_sides) * self.thickness
        dtheta = 0

        total_side = inner_diameter * tan(pi / N)
        fingered_side = total_side - dtheta

        print(f"{inner_diameter=} {fingered_side=} {dtheta=}")
        slot_length = self.slot_length
        slot_altitude = (
            inner_diameter + self.spool_hole_diameter - self.slot_diameter
        ) / 2
        print(f"{slot_altitude=}")
        backedge = (
            total_side
            - (2 * self.top + self.opening)
            # -> now we are at mid level of top side
            + slot_length
            + inner_diameter / 2
        )
        vars = {
            "n": N,
            "inner_diameter": inner_diameter,
            "inner_width": self.inner_width,
            "fingered_side": fingered_side,
            "angle_between_sides": degrees(angle_between_sides),
            "total_side": total_side,
            "slot_length": slot_length,
            "backedge": backedge,
            "dtheta": dtheta,
            "outer_diameter": self.outer_diameter,
            "slot_altitude": slot_altitude,
            # how high above center of the polygon is center of the slot
            "half_dt_gap": Plain(dtheta / 2),  # if dtheta else Turn(0),
            "thickness": self.thickness,
            "top": self.top,
            "opening": self.opening,
            "edge_turn": Turn(180 - degrees(angle_between_sides)),
        }
        print(tabulate(vars.items()))
        return vars

    @inject_shortcuts
    def front_top(self, thickness):
        w = self.wall_builder("front top").add(
            [
                Plain(self.inner_width + 4 * thickness),
                Turn(90),
                Edge(self.top, "B"),
                Turn(90),
            ]
            * 2
        )
        return Element.from_item(w, is_part="front top")

    @inject_shortcuts
    def front_down(self, thickness, inner_width):
        top_bottom = [Edge(self.top, "B"), Turn(90)]
        w = self.wall_builder("front down").add(
            Plain(thickness * 2),
            Edge(inner_width, "a"),
            Plain(thickness * 2),
            Turn(90),
            top_bottom,
            Plain(inner_width + 4 * thickness),
            Turn(90),
            top_bottom,
        )
        return Element.from_item(w, is_part="front down")

    @inject_shortcuts
    def inner_side_insert(
        self,
        fingered_side,
        inner_diameter,
        slot_altitude,
        half_dt_gap,
        top,
        opening,
        edge_turn,
    ):
        one_side = [half_dt_gap, Edge(fingered_side, "f"), half_dt_gap, edge_turn]
        w = self.wall_builder("inner_side_insert").add(
            # v- bottom, back right
            one_side,
            one_side,
        )

        w.add(half_dt_gap)  # back
        if False:
            # back
            w.add(Edge(top, "F"), Plain(opening + top))
        else:
            w.add(Edge(top + opening + top, "f"))  # back
        w.turn(90)

        # middle of the circle is (fingered_side/2, e/2).
        # X: that's correct.
        print(f"{w.position = }")
        # print(f"expected center: {(fingered_side/2, e/2)}")
        # print(f"resulting x: {w.position[0] - inner_diameter/2}")
        depth = w.position[1] - slot_altitude
        print(f"{depth = }")
        w.add(
            # v- top
            Edge(
                inner_diameter,
                DirectSlotEdge(
                    self,
                    depth=w.position[1] - slot_altitude,
                    diameter=self.slot_diameter,
                ),
            ),
            Turn(90),
            # v- front
            Edge(top, "f"),
            Plain(opening + top),
            # v- front left
            half_dt_gap,
            edge_turn,
            one_side,
        )
        return Element.from_item(w, is_part="inner side insert")

    @inject_shortcuts
    def side(
        self,
        left: bool,
        fingered_side,
        inner_diameter,
        slot_length,
        backedge,
        total_side,
        slot_altitude,
        half_dt_gap,
        edge_turn,
        top,
        opening,
    ):
        # w.close=False
        standard_side = [
            half_dt_gap,
            Edge(fingered_side, "f"),
            half_dt_gap,
            edge_turn,
        ]
        w = self.wall_builder("side").add(
            standard_side,  # <- bottom
            standard_side,  # <- back right
            half_dt_gap,  # <- back
        )
        # print(f"back right: {w.position=}")

        BACK_MODE = 2
        if BACK_MODE == 1:
            w.edge(top, "f").plain(opening + top + backedge)  # back
        elif BACK_MODE == 2:
            w.edge(top + opening + top, "f").edge(backedge, "f")  # back
        else:
            raise
        w.turn(90)
        # print(f"end of back: {w.position}")

        w.add(Edge(inner_diameter, "f"), Turn(90))  # top

        # print(f"start of front: {w.position}")
        # front
        w.add(
            Plain(backedge),
            Edge(top, "f"),
            Plain(opening),
            Edge(top, "f"),  # front
            half_dt_gap,
            edge_turn,  # <- front left
            standard_side,  # <- front left
        )
        # front
        print(f"x={inner_diameter/2=}")

        relative_x = total_side / 2
        print(f"relative X = {relative_x}")

        def additions():
            if left:
                self.slotHole(
                    # center of the polygon is: (total_side/2) above,
                    x=relative_x,
                    y=slot_altitude,
                    r=self.outer_diameter / 2,
                    dist=slot_length,
                )
            else:
                self.twoSlotHole(
                    x=relative_x,
                    y=slot_altitude,
                    r1=self.outer_diameter / 2,
                    r2=self.major_open_diameter / 2,
                    dist=slot_length,
                )

        return Element.from_item(w, is_part="side").add_render(additions)

    def _multimark(self, x=0, y=0):
        self.moveTo(x, y)
        for size in (1, 2, 3):
            self.hole(0, 0, size)

    @restore
    @holeCol
    def slotHole(self, x, y, r, dist):
        self.moveTo(x + r, y, -90)
        # self._multimark(x, y)
        self.corner(-180, r)
        self.edge(dist)
        self.corner(-180, r)
        self.edge(dist)

    @restore
    @holeCol
    def twoSlotHole(self, x, y, r1, r2, dist):
        # (x,y) is middle of bottom circle
        # dist is ... what?
        self.moveTo(x + r1, y, -90)
        # self._multimark(0, 0)

        self.corner(-180, r1)
        beta = acos(r1 / r2)
        extra = (r2**2 - r1**2) ** 0.5
        # print(f"{extra=}")
        self.edge(dist - extra)
        # print(f"{r1=} {r2=}")
        # print(f"{beta = :.2f}")
        self.corner(degrees(beta))
        corner_angle = -(180 + 2 * degrees(beta))
        # print(f"{corner_angle=}")
        self.corner(corner_angle, r2)
        self.corner(degrees(beta))
        # print(f"{dist-extra=}")
        self.edge(dist - extra)

    @inject_shortcuts
    def top_thing(self, inner_diameter, inner_width, thickness):
        w = self.wall_builder("top thing").add(
            # v- front side
            Plain(thickness),  # overhang for hanging
            Plain(inner_width + 4 * thickness),
            Turn(90),
            # v- finger hole edge on the right
            Edge(inner_diameter, "F"),
            Turn(90),
            # back side
            Edge(inner_width + thickness * 4, "f"),
            Plain(thickness),  # overhang for hanging
            Turn(90),
            Edge(inner_diameter, "f"),  # inner hang
            Turn(90),
        )

        def additions():
            self.fingerHolesAt(thickness * 0.5, 0, inner_diameter, 90)

        return Element.from_item(w, is_part="front down").add_render(additions)

    @inject_shortcuts
    def frontside(self, fingered_side, thickness, inner_width):
        # XXX: ARGGGHHHGGHHHH
        gap = Plain(thickness * 2)
        vertical = [Edge(fingered_side, "B"), Turn(90)]
        w = self.wall_builder("frontside").add(
            ###
            gap,
            Edge(inner_width, "a"),
            gap,
            Turn(90),
            ###
            vertical,
            ###
            gap,
            Edge(inner_width, "A"),
            gap,
            Turn(90),
            ###
            vertical,  # <- the side where there's fingers will get lengthened?
        )
        return Element.from_item(w, is_part="frontside")

    @inject_shortcuts
    def tab(self, inner_diameter, thickness):
        tab_height = thickness * 4
        vertical = [Plain(tab_height), Turn(90)]
        w = self.wall_builder("tab").add(
            ###
            Edge(inner_diameter, "F"),
            Turn(90),
            ###
            vertical,
            ###
            Plain(inner_diameter),
            Turn(90),
            ###
            vertical,
        )
        return Element.from_item(w, is_part="tab")

    def makecircle(self):
        def render():
            self.corner(360, radius=self.spoolhead / 2)

            # self.hole(0, self.spoolhead/2, self.rod_d)
            self.ctx.stroke()
            self.moveTo(0, self.spoolhead / 2 - self.rod_d / 2)
            self.corner(360, radius=self.rod_d / 2)
            self.set_source_color(Color.ETCHING)
            self.ctx.stroke()
            # self.rectangular_etching(d.centre_x, d.centre_y, d.photo_x, d.photo_y)

        return Element(
            position=np.array((0, 0), dtype=float),
            bbox=BBox(minx=0, miny=0, maxx=self.spoolhead, maxy=self.outer_diameter),
            render=[render],
            boxes=self,
            is_part="makecircle",
        )

    def open(self):
        self.apply_presets()
        super().open()  # <- call after - we change thickness

    # def calisquare(self):
    #     w = self.wall_builder("calisquare").add([Plain(10), Turn(90)] * 4)
    #     return Element.from_item(w, is_part="calisquare")

    @inject_shortcuts
    def backpiece(self, backedge, thickness, inner_width, opening, top):
        back = [
            Edge(top + opening + top, "B"),
            Edge(backedge, "F"),
            Plain(thickness),
        ]
        w = self.wall_builder("backpiece").add(
            # v- bottom
            Plain(thickness * 2),
            Edge(inner_width, "A"),
            Plain(thickness * 2),
            Turn(90),
            # v- back
            back,
            Turn(90),
            # v- top
            Edge(thickness * 4 + inner_width, "F"),
            Turn(90),
            # front?
            list(reversed(back)),
            Turn(-90),
        )
        return Element.from_item(w, is_part="backpiece")

    @inject_shortcuts
    def build(self, angle_between_sides, thickness):
        FingerJointSettings(
            thickness,
            **self.edgesettings.get("FingerJoint"),
            angle=180 - angle_between_sides,
        ).edgeObjects(self, chars="aA")
        FingerJointSettings(
            thickness,
            **(self.edgesettings.get("FingerJoint") | {"extra_length": 1}),
        ).edgeObjects(self, chars="bB")

        return self.ystack(
            # self.calisquare(),
            self.xstack(
                self.side(left=True),
                self.side(left=False),
                self.ystack(
                    self.backpiece(),
                    self.front_top(),
                    self.front_down(),
                ),
            ),
            self.xstack(self.inner_side_insert(), self.inner_side_insert()),
            self.xstack([self.frontside()] * 3),
            self.xstack(
                self.top_thing(),
                self.tab(),
                self.makecircle(),
                self.makecircle(),
                # TODO: add some outer circles here
                # TODO: top thing has broken finger holes
            ),
        )
