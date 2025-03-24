"""
scripts/boxes TapeHolder --debug=True --reference 0
"""

import numpy as np
from hamcrest import assert_that, close_to
from math import cos,tan,sin, radians, pi
from boxes.edges import BaseEdge
from boxes.generators.raibase import RaiBase, inject_shortcuts, Element

class AngledSlotEdge(BaseEdge):
    """Edge with 'depth' depth circular slot in the middle of its length.
    The slot is at an angle 'angle'."""
    description="AngledSlotEdge"
    def __init__(self, boxes, angle, depth, diameter, center):
        super().__init__(boxes, None)
        self.angle = angle
        self.depth = depth
        self.diameter = diameter
        self.center = center

    def __call__(self, length, **kw):
        angle = self.angle
        h = self.depth
        d = self.diameter
        l = length
        center = self.center

        ang = radians(angle)
        sin_a,cos_a,tan_a = sin(ang),cos(ang),tan(ang)
        delta = d/cos_a

        print(f"expected depth of center: {h:.2f}")

        #a_mid, a_delta = cos_a * h, tan_a * (d/2)
        a_mid, a_delta = h/cos_a , tan_a * d/2
        B, C = a_mid + a_delta, a_mid - a_delta

        #center = np.array((l/2, h/2))
 
        A = (center - delta/2) - h * tan_a
        D = ((l - center) - delta/2) + h * tan_a
        #assert_that(A+D+delta, close_to(l, 0.001))

        with self.boxes.saved_context():
            self.boxes.moveTo(center, h)
            self.boxes.text("center", align="middle center", fontsize=7)
            self.boxes.hole(0, 0, 2)
            self.boxes.hole(0, 0, 1)

        self.boxes.edge(A)
        self.boxes.corner(90-angle)
        self.boxes.edge(B)

        self.boxes.corner(-180, d/2)

        self.boxes.edge(C)
        self.boxes.corner(90+angle)
        self.boxes.edge(D)



class TapeHolder(RaiBase):
    def __init__(self):
        super().__init__()

        rod_length_default = 107 # 10.7 cm currently
        thickness_default = 25.4/8
        rod_length_play = 1.5
        rod_diameter = 32 # 32 mm currently
        rod_diameter_play = 1.5

        self.argparser.add_argument(
            "--inner_width",
            action="store",
            type=float,
            default=rod_length_default - thickness_default - rod_length_play,
        )
        # needing: about 7cm
        # => need here about 86mm
        self.argparser.add_argument(
            "--inner_edge",
            action="store",
            type=float,
            default=86,  # TODO: this is a guess
        )
        self.argparser.add_argument(
            "--slot_diameter",
            action="store",
            type=float,
            default=rod_diameter + rod_diameter_play,
        )
        self.argparser.add_argument(
            "--outer_diameter",
            action="store",
            type=float,
            default=rod_diameter + rod_diameter_play + 3,
        )
        self.argparser.add_argument(
            "--alpha",
            action="store",
            type=float,
            default=5,
        )
        self.argparser.add_argument(
            "--top",
            action="store",
            type=float,
            default=15,
        )
        self.argparser.add_argument(
            "--opening",
            action="store",
            type=float,
            default=2,
        )

    @property
    def shortcuts(self):
        return {
            'e': self.inner_edge,
            'a': self.alpha,
            'tan_a': tan(self.alpha),
        }

    @inject_shortcuts
    def side(self, slot_diameter, e):
        w = self.wall_builder("side A")
        w.close=False

        # e = circumscribed circle diameter

        N = 6
        angle = ((N-2)*180)/N
        edge = e*sin(pi/N)
        print(f"{angle=} {edge=}")

        w.add(edge, 180-angle, 'f')
        w.add(edge, 270+angle, 'f')

        #w.add(edge, 180-angle, 'f')
        w.add(
            #edge,
            150,
            90,
            AngledSlotEdge(
                self,
                angle=-angle/2,
                depth=e/2,
                diameter=self.slot_diameter,
                center=0, #edge/2,
            ),
        )
        w.add(e, 90, 'f')
        w.add(150 - (2*self.top + self.opening), 0, 'e')
        w.add(self.top, 0, 'f')
        w.add(self.opening, 0, 'e')
        w.add(self.top, 270+angle, 'f')
        w.add(edge, 180-angle, 'f')
        # front
        ###w.add(edge - (self.top + self.opening), 180-angle, 'f')
        return Element.from_item(w)


    @inject_shortcuts
    def render(self, e, a, tan_a):
        a_inner = self.side(slot_diameter=self.slot_diameter)
        a_outer = self.side(slot_diameter=self.outer_diameter)
        self.ystack(a_inner, a_outer).do_render()

#        w = self.wall_builder("side A")
#        w.add(e, 90, 'f')
#        w.add(e, 90, 'f')
#        w.add(e, 90, AngledSlotEdge(self, angle=self.alpha, depth=e/2, diameter=self.slot_diameter, center=e/2))
#        # front
#        w.add(self.top, 0, 'f')
#        w.add(self.opening, 0, 'e')
#        w.add(e - (self.top + self.opening), 0, 'f')
#        side_a = Element.from_item(w)


#        w = self.wall_builder("side A")
#        w.add(e, 90, 'f')
#        w.add(e, 90, 'f')
#        w.add(e, 90, AngledSlotEdge(self, angle=self.alpha, depth=e/2, diameter=self.outer_diameter, center=e/2))
#        # front
#        w.add(self.top, 0, 'f')
#        w.add(self.opening, 0, 'e')
#        w.add(e - (self.top + self.opening), 0, 'f')
#        side_outer = Element.from_item(w)
#
#        self.ystack(side_a, side_outer).do_render()

        #w.add((
        ## ... todo ...

