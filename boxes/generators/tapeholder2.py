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

INCH = 25.4

class DirectSlotEdge(BaseEdge):
    """Edge with 'depth' depth circular slot in the middle of its length.
    The slot is at an angle 'angle'."""
    description="DirectSlotEdge"
    def __init__(self, boxes, depth, diameter):
        super().__init__(boxes, None)
        self.depth = depth
        self.diameter = diameter

    def __call__(self, length, **kw):
        depth = self.depth
        diameter = self.diameter
        l = length

        #with self.boxes.saved_context():
        #    self.boxes.moveTo(center, h)
        #    self.boxes.text("center", align="middle center", fontsize=7)
        #    self.boxes.hole(0, 0, 2)
        #    self.boxes.hole(0, 0, 1)

        self.boxes.edge((l - diameter)/2)
        self.boxes.corner(90)
        self.boxes.edge(depth)

        self.boxes.corner(-180, diameter/2)

        self.boxes.edge(depth)
        self.boxes.corner(90)
        self.boxes.edge((l - diameter)/2)


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
        if self.preset == "test2":
            # self.thickness = 3.8  # cardboard
            self.thickness = INCH / 8
            self.spool_hole_diameter = 40
            rod_l = 30
            rod_d = 15.78
            self.slot_diameter = 0.6 + rod_d
            self.inner_diameter = 50 
            self.slot_length = 30
            self.burn = 0
        elif self.preset == "15leroy":
            # self.thickness = 3.8  # cardboard
            self.thickness = INCH / 8
            self.spool_hole_diameter = 40

            # thick rod:
            # rod_l = 107
            # rod_d = 32
            # d play was 0.6mm

            rod_l = 100.73
            rod_d = 15.78

            self.slot_diameter = 0.6 + rod_d # 32 mm currently, 0.8 mm is play

            # diameter of area for spool.
            # actual measured diameter: 67mm
            self.inner_diameter = 86 
            self.slot_length = 60
            self.burn = 0.07 # yes for the 3mm plywood 25mm/s 45%
        else:
            raise
        rod_l_play = 0.6
        # inner width is the distance between the 2 inner A inserts.
        # rod includes their thickness.
        # rod can also move a bit to left/right = play
        self.inner_width = rod_l - rod_l_play - 2 * self.thickness
        self.outer_diameter = self.slot_diameter + 10
        assert self.outer_diameter < self.spool_hole_diameter, f"{self.outer_diameter} should be <40"
        self.spoolhead = self.outer_diameter - 2
        self.top, self.opening = 15, 2
        print(f"{self.inner_diameter = }")

    @property
    @lru_cache()
    def shortcuts(self):
        N = 8
        inner_diameter = self.inner_diameter
        angle_between_sides = pi * (N-2)/N

        #dtheta = sin(angle_between_sides) * self.thickness
        dtheta=0

        total_side = inner_diameter * tan(pi/N)
        fingered_side = total_side - dtheta


        print(f"{inner_diameter=} {fingered_side=} {dtheta=}")
        slot_length = self.slot_length
        slot_altitude = inner_diameter / 2 + self.spool_hole_diameter / 2 - self.slot_diameter / 2
        backedge = total_side + slot_length + slot_altitude - (2*self.top + self.opening)
        vars = {
                'n': N,
            'inner_diameter': inner_diameter,
            'fingered_side': fingered_side,
            'angle_between_sides': degrees(angle_between_sides),
            'total_side': total_side,
            'slot_length': slot_length,
            'backedge': backedge,
            'dtheta': dtheta,
            'outer_diameter': self.outer_diameter,
            'slot_altitude': slot_altitude,
            # how high above center of the polygon is center of the slot
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
        return Element.from_item(w).is_part("front top").close_part("front top")

    @inject_shortcuts
    def front_down(self):
        w = self.wall_builder("front down")
        w.add(self.inner_width, 90, 'a')
        w.add(self.top, 90, 'b')
        w.add(self.inner_width, 90, 'e')
        w.add(self.top, 90, 'b')
        return Element.from_item(w).is_part("front down").close_part("front down")

    @inject_shortcuts
    def back_down(self, dtheta):
        w = self.wall_builder("back down")
        w.add(self.inner_width, 90, 'A')

        w.add(dtheta/2, 0, 'e')
        w.add(self.top, 90, 'b')

        w.add(self.inner_width, 90, 'e')

        w.add(dtheta/2, 0, 'e')
        w.add(self.top, 90, 'b')
        return Element.from_item(w).is_part("back down").close_part("back down")

    @inject_shortcuts
    def inner_side_insert(self, fingered_side, inner_diameter, angle_between_sides, dtheta, slot_altitude):
        w = self.wall_builder("inner_side_insert")
        #w.close=False
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
                depth=w.position[1] - slot_altitude,
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
        return Element.from_item(w).is_part("inner side insert").close_part("inner side insert")

    @inject_shortcuts
    def side(self, fingered_side, inner_diameter, slot_length, backedge, angle_between_sides, dtheta, total_side, slot_altitude):
        w = self.wall_builder("side")
        #w.close=False

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
                y=slot_altitude,
                r=self.outer_diameter/2,
                dist=slot_length,
            )
        el.add_render(additions)
        return el.is_part("side").close_part("side")

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
        # dist is ... what?
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
        return el.is_part("front down").close_part("front down")

    @inject_shortcuts
    def frontside(self, fingered_side, angle_between_sides):
        w = self.wall_builder("frontside")
        # XXX: ARGGGHHHGGHHHH
        w.add(self.inner_width, 90, 'a')
        w.add(fingered_side, 90, 'b')
        w.add(self.inner_width, 90, 'A')
        w.add(fingered_side, 90, 'b')
        return Element.from_item(w).is_part("frontside").close_part("frontside")

    @inject_shortcuts
    def tab(self, inner_diameter):
        tab_height = self.thickness * 4
        w = self.wall_builder("tab")
        w.add(inner_diameter, 90, 'F')
        w.add(tab_height, 90, 'e')
        w.add(inner_diameter, 90, 'e')
        w.add(tab_height, 90, 'e')
        return Element.from_item(w).is_part("tab").close_part("tab")

    def makecircle(self):
        def render():
            self.corner(360, radius=self.spoolhead/2)
        return Element(
                position=np.array((0, 0), dtype=float),
                bbox=BBox(minx=0, miny=0, maxx=self.spoolhead, maxy=self.outer_diameter),
                render=[render],
                boxes=self).is_part("makecircle").close_part("makecircle")

    def open(self):
        self.apply_presets()
        super().open() # <- call after - we change thickness
    
    @inject_shortcuts
    def render(self, angle_between_sides):
        FingerJointSettings(self.thickness, angle=angle_between_sides, **self.edgesettings.get("FingerJoint")).edgeObjects(self, chars="aA")
        FingerJointSettings(self.thickness,  **(self.edgesettings.get("FingerJoint") | {"extra_length": 1})).edgeObjects(self, chars="bB")


        self.ystack(
            self.xstack(self.side(inner=False), self.side(inner=False)),
            self.xstack(self.inner_side_insert(), self.inner_side_insert()),
            self.front_top(),
            self.front_down(),
            self.back_down(),
            self.xstack([self.frontside()] * 3),
            self.top_thing(),
            self.tab(),
            self.xstack(self.makecircle(), self.makecircle())
        ).do_render()

