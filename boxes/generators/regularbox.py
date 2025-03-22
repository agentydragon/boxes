# Copyright (C) 2013-2014 Florian Festi
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

from boxes import *
from boxes.generators.bayonetbox import BayonetBox


class RegularBox(BayonetBox):
    """Box with regular polygon as base"""

    description = """For short side walls that don't fit a connecting finger reduce *surroundingspaces* and *finger* in the Finger Joint Settings.

The lids needs to be glued. For the bayonet lid all outside rings attach to the bottom, all inside rings to the top.
"""

    ui_group = "Box"

    def __init__(self) -> None:
        Boxes.__init__(self)
        self.addSettingsArgs(edges.FingerJointSettings, surroundingspaces=1)
        self.buildArgParser("h", "outside")
        self.argparser.add_argument(
            "--radius_bottom",  action="store", type=float, default=50.0,
            help="inner radius of the box bottom (at the corners)")
        self.argparser.add_argument(
            "--n",  action="store", type=int, default=5,
            help="number of sides")
        self.argparser.add_argument(
            "--top",  action="store", type=str, default="none",
            choices=["none", "hole", "angled hole", "angled lid", "angled lid2", "round lid", "bayonet mount", "closed"],
            help="style of the top and lid")
        self.argparser.add_argument(
            "--alignment_pins",  action="store", type=float, default=1.0,
            help="diameter of the alignment pins for bayonet lid")
        self.argparser.add_argument(
            "--bottom",  action="store", type=str, default="closed",
            choices=["none", "closed", "hole", "angled hole", "angled lid", "angled lid2", "round lid"],
            help="style of the bottom and bottom lid")

        self.lugs=6

    def render(self):
        r0, h, n = self.radius_bottom, self.h, self.n
        t = self.thickness
        r0, sh0, side0  = self.regularPolygon(n, radius=r0)

        # length of side edges
        #l = (((side0-side0)/2)**2 + (sh0-sh0)**2 + h**2)**0.5
        l = ((r0-r1)**2 + h**2)**.5
        # angles of sides -90° aka half of top angle of the full pyramid sides
        a = math.degrees(0)
        # angle between sides (in boxes style change of travel)
        phi = 180 - 2 * math.degrees(
            math.asin(math.cos(math.pi/n) / math.cos(math.radians(a))))

        fingerJointSettings = copy.deepcopy(self.edges["f"].settings)
        fingerJointSettings.setValues(self.thickness, angle=phi)
        fingerJointSettings.edgeObjects(self, chars="gGH")



        def drawTop(r, sh, top_type, joint_type):
            if top_type == "closed":
                self.regularPolygonWall(corners=n, r=r, edges=joint_type[1], move="right")
            elif top_type == "angled lid":
                self.regularPolygonWall(corners=n, r=r, edges='e', move="right")
                self.regularPolygonWall(corners=n, r=r, edges='E', move="right")
            elif top_type in ("angled hole", "angled lid2"):
                self.regularPolygonWall(corners=n, r=r, edges=joint_type[1], move="right",
                                        callback=[lambda:self.regularPolygonAt(
                                            0, 0, n, h=sh-t)])
                if top_type == "angled lid2":
                    self.regularPolygonWall(corners=n, r=r, edges='E', move="right")
            elif top_type in ("hole", "round lid"):
                self.regularPolygonWall(corners=n, r=r, edges=joint_type[1], move="right",
                                        hole=(sh-t)*2)
            if top_type == "round lid":
                self.parts.disc(sh*2, move="right")
            if self.top == "bayonet mount":
                self.diameter = 2*sh
                self.parts.disc(sh*2-0.1*t, callback=self.lowerCB,
                                move="right")
                self.regularPolygonWall(corners=n, r=r, edges='F',
                                        callback=[self.upperCB], move="right")
                self.parts.disc(sh*2, move="right")


        with self.saved_context():
            drawTop(r0, sh0, self.bottom, "fF")

        self.regularPolygonWall(corners=n, r=r0, edges='F', move="up only")

        fingers_top = self.top in ("closed", "hole", "angled hole",
                                   "round lid", "angled lid2", "bayonet mount")
        fingers_bottom = self.bottom in ("closed", "hole", "angled hole",
                                         "round lid", "angled lid2")

        t_ = self.edges["f"].startwidth()
        bottom_edge = ('f' if fingers_bottom else 'e')
        top_edge = ('f' if fingers_top else 'e')

        if n % 2:
            e = bottom_edge + 'ege' + top_edge + 'eeGee'
            borders = [side0, 90-a, 0, 0, l, 0, 0, 90+a, side0,
                       90+a, 0, -90, t_, 90, l, 90, t_, -90, 0, 90-a]
            for i in range(n):
                self.polygonWall(borders, edge=e, correct_corners=False,
                                 move="right")
        else:
            borders0 = [side0, 90-a,
                        0, -90, t_, 90, l, 90, t_, -90, 0, 90+a,
                        side0, 90+a,
                        0, -90, t_, 90, l, 90, t_, -90, 0, 90-a]
            e0 = bottom_edge + 'eeGee' + top_edge + 'eeGee'

            borders1 = [side0, 90-a,
                        l, 90+a,
                        side0, 90+a,
                        l, 90-a]
            e1 = bottom_edge + 'g' + top_edge + 'g'
            for i in range(n//2):
                self.polygonWall(borders0, edge=e0, correct_corners=False,
                                 move="right")
                self.polygonWall(borders1, edge=e1, correct_corners=False,
                                 move="right")
