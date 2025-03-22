"""
scripts/boxes TapeHolder2 --debug=True --reference 0 --preset test
"""

from functools import lru_cache
import numpy as np
from hamcrest import assert_that, close_to
from math import cos,tan,sin, radians, pi, acos, degrees
from boxes.edges import BaseEdge, FingerJointSettings
from boxes.generators.raibase import RaiBase, inject_shortcuts, Element, BBox
from boxes import restore, holeCol

class DirectSlotEdge(BaseEdge):
    """Edge with 'depth' depth circular slot in the middle of its length.
    The slot is at an angle 'angle'."""
    description="DirectSlotEdge"
    def __init__(self, boxes, depth, diameter):
        super().__init__(boxes, None)
        self.depth = depth
        self.diameter = diameter

    def __call__(self, length, **kw):
        h = self.depth
        d = self.diameter
        l = length

        #with self.boxes.saved_context():
        #    self.boxes.moveTo(center, h)
        #    self.boxes.text("center", align="middle center", fontsize=7)
        #    self.boxes.hole(0, 0, 2)
        #    self.boxes.hole(0, 0, 1)

        self.boxes.edge((l - d)/2)
        self.boxes.corner(90)
        self.boxes.edge(h)

        self.boxes.corner(-180, d/2)

        self.boxes.edge(h)
        self.boxes.corner(90)
        self.boxes.edge((l - d)/2)


class TapeHolder2(RaiBase):
    def __init__(self):
        super().__init__()

        self.argparser.add_argument(
            "--inner_slot_extra_altitude",
            action="store",
            type=float,
            default=0,
        )
        self.argparser.add_argument(
            "--inner_width",
            action="store",
            type=float,
        )
        self.argparser.add_argument(
            "--inner_diameter",
            action="store",
            type=float,
        )
        self.argparser.add_argument(
            "--slot_diameter",
            action="store",
            type=float,
        )
        self.argparser.add_argument(
            "--outer_diameter",
            action="store",
            type=float,
        )
        self.argparser.add_argument(
            "--top",
            action="store",
            type=float,
        )
        self.argparser.add_argument(
            "--opening",
            action="store",
            type=float,
        )
        self.argparser.add_argument(
            "--slot_length",
            action="store",
            type=float,
        )
        self.argparser.add_argument(
            "--preset",
            action="store",
            type=str,
            default="",
        )

    def apply_presets(self):
        if not self.preset:
            return
        if self.preset == "test":
            rod_l = 40
            self.thickness = 3.6  # cardboard
            self.slot_diameter = 10 + 1
            self.inner_diameter = 40
            self.slot_length = 40
        elif self.preset == "15leroy":
            rod_l = 107 # 10.7 cm currently
            self.thickness = 25.4 / 8
            # self.thickness = 3.8  # cardboard
            self.slot_diameter = 0.8 + 32 # 32 mm currently, 0.8 mm is play
            self.inner_diameter = 86
            self.slot_length = 90
            self.burn = 0
            self.slot_extra_altitude = 6
            self.slot_top_offset = 20
        else:
            raise
        rod_l_play = 0.6
        # inner width is the distance between the 2 inner A inserts.
        # rod includes their thickness.
        # rod can also move a bit to left/right = play
        self.inner_width = rod_l - rod_l_play - 2 * self.thickness
        self.outer_diameter = self.slot_diameter + 10
        self.spoolhead = self.outer_diameter - 2
        self.top, self.opening = 15, 2
        print(f"{self.inner_diameter = }")

    @property
    @lru_cache()
    def shortcuts(self):
        N = 8
        inner_diameter = self.inner_diameter # inner_diameter = circle inside polygon
        angle_between_sides = pi * (N-2)/N
        dtheta = sin(angle_between_sides) * self.thickness
        #print(f"{dtheta=}")

        total_side = inner_diameter * tan(pi/N)

        fingered_side = total_side - dtheta
        inner_diameter = inner_diameter #/ cos(pi/N)
        print(f"{inner_diameter=} {fingered_side=} {dtheta=}")
        slot_length = self.slot_length
        backedge = total_side + slot_length - (2*self.top + self.opening)#  + slot_length
        vars = {
                'n': N,
            'inner_diameter': inner_diameter,
            'fingered_side': fingered_side,
            'angle_between_sides': degrees(angle_between_sides),
            'total_side': total_side,
            'slot_length': self.slot_length,
            'backedge': backedge,
            'dtheta': dtheta,
        }
        from tabulate import tabulate
        print(tabulate(vars.items()))
        return vars

    @inject_shortcuts
    def front_top(self, dtheta):
        w = self.wall_builder("front top")
        w.add(self.inner_width, 90, 'e')
        w.add(self.top, 90, 'b')
        w.add(self.inner_width, 90, 'e')
        w.add(self.top, 90, 'b')
        return Element.from_item(w).is_part()

    @inject_shortcuts
    def front_down(self):
        w = self.wall_builder("front down")
        w.add(self.inner_width, 90, 'a')
        w.add(self.top, 90, 'b')
        w.add(self.inner_width, 90, 'e')
        w.add(self.top, 90, 'b')
        return Element.from_item(w).is_part()

    @inject_shortcuts
    def back_down(self, dtheta):
        w = self.wall_builder("back down")
        w.add(self.inner_width, 90, 'A')

        w.add(dtheta/2, 0, 'e')
        w.add(self.top, 90, 'b')

        w.add(self.inner_width, 90, 'e')

        w.add(dtheta/2, 0, 'e')
        w.add(self.top, 90, 'b')
        return Element.from_item(w).is_part()

    @inject_shortcuts
    def inner_side_insert(self, fingered_side, inner_diameter, angle_between_sides, dtheta):
        w = self.wall_builder("inner_side_insert")
        w.close=False
        w.add(dtheta/2, 0, 'e') # bottom
        w.add(fingered_side, 0, 'F') # bottom
        w.add(dtheta/2, 180-angle_between_sides, 'e') # bottom

        w.add(dtheta/2, 0, 'e') # back right
        w.add(fingered_side, 0, 'F') # back right
        w.add(dtheta/2, 180-angle_between_sides, 'e') # back right

        w.add(dtheta/2, 0, 'e') # back
        w.add(self.top, 0, 'F') # back
        w.add(self.opening, 0, 'e') # back
        w.add(self.top, 90, 'e') # back

        # middle of the circle is (fingered_side/2, e/2).
        # X: that's correct.
        print(f"{w.position = }")
        #print(f"expected center: {(fingered_side/2, e/2)}")
        #print(f"resulting x: {w.position[0] - inner_diameter/2}")
        w.add(
            inner_diameter,
            90, 
            DirectSlotEdge(
                self,
                depth=(w.position[1] - inner_diameter/2) ,#- self.slot_extra_altitude,#10,
                diameter=self.slot_diameter,
            ),
        ) # <- top

        w.add(self.top, 0, 'F') # front
        w.add(self.opening, 0, 'e') # front
        w.add(self.top, 0, 'F') # front
        w.add(dtheta/2, 180-angle_between_sides, 'e') # <- front left

        w.add(dtheta/2, 0, 'e') # <- front left
        w.add(fingered_side, 0, 'F') # <- front left
        w.add(dtheta/2, 180-angle_between_sides, 'e') # <- front left
        return Element.from_item(w).is_part()

    @inject_shortcuts
    def side(self, fingered_side, inner_diameter, slot_length, backedge, angle_between_sides, dtheta, total_side):
        w = self.wall_builder("side")
        w.close=False

        w.add(dtheta/2, 0, 'e') # bottom
        w.add(fingered_side, 0, 'F') # bottom
        w.add(dtheta/2, 180-angle_between_sides, 'e') # bottom

        w.add(dtheta/2, 0, 'e') # back right
        w.add(fingered_side, 0, 'F') # back right
        w.add(dtheta/2, 180-angle_between_sides, 'e') # back right

        print(f"back right: {w.position=}")

        w.add(dtheta/2, 0, 'e') # back
        w.add(self.top, 0, 'F') # back
        w.add(self.opening, 0, 'e') # back
        w.add(self.top, 0, 'e') # back
        w.add(backedge, 90, 'e') # back
        print(f"end of back: {w.position}")

        w.add(inner_diameter, 90, 'f') # top

        print(f"start of front: {w.position}")
        w.add(backedge, 0, 'e') # front
        w.add(self.top, 0, 'F') # front
        w.add(self.opening, 0, 'e') # front
        w.add(self.top, 0, 'F') # front
        w.add(dtheta/2, 180-angle_between_sides, 'e') # <- front left

        w.add(dtheta/2, 0, 'e') # <- front left
        w.add(fingered_side, 0, 'F') # <- front left
        w.add(dtheta/2, 180-angle_between_sides, 'e') # <- front left
        # front
        print(f"x={inner_diameter/2=}")

        el = Element.from_item(w)
        relative_x = total_side/2
        print(f"relative X = {relative_x}")
        def additions():
            self.slotHole(
                # center of the polygon is: (total_side/2) above, 
                x=relative_x,
                y=self.inner_diameter / 2 + self.slot_extra_altitude,
                r=self.outer_diameter/2,
                dist=slot_length - self.slot_top_offset - self.slot_extra_altitude,
            )
        el.add_render(additions)
        return el.is_part()

    @restore
    @holeCol
    def slotHole(self, x, y, r, dist):
        self.moveTo(x + r, y, -90)
        #self.hole(0,0,3)
        #self.hole(0,0,2)
        #self.hole(0,0,1)
        self.corner(-180, r)
        self.edge(dist)
        self.corner(-180, r)
        self.edge(dist)

    @restore
    @holeCol
    def twoSlotHole(self, x, y, r1, r2, dist):
        # (x,y) is middle of bottom circle
        self.moveTo(x + r1, y, -90)
        self.hole(0, 0, 2)
        self.hole(0, 0, 3)

        self.corner(-180, r1)
        beta = acos(r1/r2)
        rd = r2-r1
        extra = (r2**2 - r1**2) ** 0.5
        print(f"{extra=}")
        self.edge(dist - extra)
        print(f"{r1=} {r2=}")
        print(f"{beta = :.2f}")
        self.corner(degrees(beta))
        corner_angle= -(180+2*degrees(beta))
        print(f"{corner_angle=}")
        self.corner(corner_angle, r2)
        self.corner(degrees(beta))
        print(f"{dist-extra=}")
        self.edge(dist-extra)

    @inject_shortcuts
    def top_thing(self, inner_diameter):
        w = self.wall_builder("front down")
        w.add(self.inner_width + self.thickness * 2, 90, 'e') # front side
        w.add(inner_diameter, 90, 'F') # finger hole edge on the right
        w.add(self.inner_width + self.thickness * 2, 0, 'e') # back side
        w.add(2 * self.thickness, 0, 'e') # space for left edge * 2
        w.add(self.thickness, 90, 'e') # overhang for hanging
        w.add(inner_diameter, 90, 'f') # inner hang
        w.add(self.thickness, 0, 'e') # overhang for hanging
        w.add(2 * self.thickness, 0, 'e') # space for left edge * 2

        el = Element.from_item(w)
        def additions():
            self.fingerHolesAt(
                -self.thickness*1.5,
                0,
                inner_diameter,
                90,
            )
        el.add_render(additions)
        return el.is_part()

    @inject_shortcuts
    def frontside(self, fingered_side, angle_between_sides):
        w = self.wall_builder("frontside")
        # XXX: ARGGGHHHGGHHHH
        w.add(self.inner_width, 90, 'a')
        w.add(fingered_side, 90, 'b')
        w.add(self.inner_width, 90, 'A')
        w.add(fingered_side, 90, 'b')
        return Element.from_item(w).is_part()

    @inject_shortcuts
    def tab(self, inner_diameter):
        tab_height = self.thickness * 4
        w = self.wall_builder("tab")
        w.add(inner_diameter, 90, 'F')
        w.add(tab_height, 90, 'e')
        w.add(inner_diameter, 90, 'e')
        w.add(tab_height, 90, 'e')
        return Element.from_item(w).is_part()

    def makecircle(self):
        def render():
            self.corner(360, radius=self.spoolhead/2)
        return Element(
                position=np.array((0, 0), dtype=float),
                bbox=BBox(minx=0, miny=0, maxx=self.spoolhead, maxy=self.outer_diameter),
                render=[render],
                boxes=self).is_part()

    def open(self):
        self.apply_presets()
        super().open() # <- call after - we change thickness
    
    @inject_shortcuts
    def render(self, angle_between_sides):
        FingerJointSettings(self.thickness, angle=angle_between_sides).edgeObjects(self, chars="aA")
        FingerJointSettings(self.thickness, extra_length=1).edgeObjects(self, chars="bB")

        self.ystack(
            self.side(),
            ####self.xstack(self.side(inner=False), self.side(inner=False)),
            self.xstack(self.inner_side_insert(), self.inner_side_insert()),
            self.front_top(),
            self.front_down(),
            self.back_down(),
            self.xstack([self.frontside()] * 3),
            self.top_thing(),
            self.tab(),
            self.xstack(self.makecircle(), self.makecircle())
        ).do_render()

