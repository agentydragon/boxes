"""
To run:
    scripts/boxes SkadisRoll --preset=demo


TODO:
    * Skadis reinforcers are incorrect (by 2cm when cut default)
    * Cut bottom spacer just as ring to reduce friction
"""

import logging
from boxes import Color
from math import sqrt, ceil
from boxes.edges import FingerJointSettings, FingerJointEdge, FingerJointEdgeCounterPart, MountingSettings, FingerHoleEdge
from boxes.edges import DoveTailSettings, DoveTailJoint, DoveTailJointCounterPart, FingerJointBase
from boxes.walledges import SkadisSettings
from boxes.fmt import (
    fmt_mm,
    fmt_mmxmm,
)
from boxes.generators.raibase import (
    Edge,
    Element,
    Plain,
    FINGER,
    mark,
    FINGER_HOLE_EDGE,
    FINGER_COUNTER,
    RaiBase,
    Close,
    Turn,
    coord,
    inject_shortcuts,
    BBox,
)

class SkadisRoll(RaiBase):
    def __init__(self) -> None:
        #logging.basicConfig(level=logging.INFO)
        super().__init__()
        self.add_arguments()

    def add_arguments(self):
        self.buildArgParser()
        # self.argparser.add_argument(
        #     "--x",
        #     action="store",
        #     type=float,
        #     help="Picture width in mm"
        # )
        # self.argparser.add_argument(
        #     "--y",
        #     action="store",
        #     type=float,
        #     help="Picture height in mm"
        # )
        self.argparser.add_argument(
            "--roll_d",
            action="store",
            type=float,
            help="Roll diameter",
        )
        self.roll_d: float
        self.argparser.add_argument(
            "--dowel_d",
            action="store",
            type=float,
            help="Dowel diameter",
        )
        self.dowel_d: float
        self.argparser.add_argument(
            "--roll_l",
            action="store",
            type=float,
            help="Roll length",
        )
        self.roll_l: float
        self.argparser.add_argument(
            "--inner_h",
            action="store",
            help="Inner height of the box",
            type=float,
        )
        self.inner_h: float
        self.argparser.add_argument(
            "--skadis_mounted_h",
            action="store",
            type=float,
            help="Part of height of walls mounted on Skadis",
        )
        self.skadis_mounted_h: float
        self.argparser.add_argument(
            "--guide_h",
            action="store",
            type=float,
            help="Height of guide walls on open sides",
        )
        self.guide_h: float
        self.argparser.add_argument(
            "--guide_opening",
            action="store",
            type=float,
            help="Opening in guide walls on open sides",
        )
        self.guide_opening: float
        self.argparser.add_argument(
            "--roof_d",
            action="store",
            type=float,
            help="Depth of the roof",
        )
        self.roof_d: float
        # self.argparser.add_argument(
        #     "--middle_t",
        #     action="store",
        #     type=float,
        #     help="Thickness of material for middle frame"
        # )
        # # self.argparser.add_argument(
        # #     "--middle_h",
        # #     action="store",
        # #     type=float,
        # #     help="Height of middle frame"
        # # )
        # # self.middle_h: float
        # self.argparser.add_argument(
        #     "--content_t",
        #     action="store",
        #     type=float,
        #     help="Combined thickness of the content (backing + picture + glass)",
        # )
        # #self.argparser.add_argument(
        # #    "--points_w",
        # #    action="store",
        # #    type=int,
        # #    help="Number of glazing points along the width",
        # #)
        # #self.argparser.add_argument(
        # #    "--points_h",
        # #    action="store",
        # #    type=int,
        # #    help="Number of glazing points along the height",
        # #)
        # #self.points_h: int

        # self.argparser.add_argument(
        #     "--dovetail_margin",
        #     action="store",
        #     type=float,
        #     help="Diagnoal distance to keep without dovetail; mm",
        # )
        # self.dovetail_margin: float
        # self.argparser.add_argument(
        #     "--front_middle_finger_margin",
        #     action="store",
        #     type=float,
        #     help="Horizontal distance to keep without finger joints; mm",
        # )
        # self.front_middle_finger_margin: float
        # self.addSettingsArgs(DoveTailSettings, size=2.0, depth=1.0)

        self.addSettingsArgs(SkadisSettings)


    @property
    def shortcuts(self):
        roll_d = self.roll_d
        opening = self.inner_h - self.guide_h
        roll_l = self.roll_l
        print(f"opening={fmt_mm(opening)}")
        if opening > roll_l:
            max_roof_depth = roll_d
            print(f"opening={fmt_mm(opening)} > roll_l={fmt_mm(roll_l)}, no max roof")
        else:
            max_roof_depth = roll_l - sqrt(roll_l ** 2 - opening ** 2)
        min_roof_depth = self.dowel_d / 2 + self.roll_d / 2
        print(f"roof depth min={fmt_mm(min_roof_depth)} max={fmt_mm(max_roof_depth)}")

        roof_d = self.roof_d
        assert (
            min_roof_depth <= roof_d <= max_roof_depth
        ), f"roof depth {fmt_mm(roof_d)} not in range {fmt_mm(min_roof_depth)} - {fmt_mm(max_roof_depth)}"

        reinforcer_l = self.skadis_mounted_h - 40 + 2 * self.thickness

        print(f"roll_d={fmt_mm(roll_d)}")
        return dict(
            reinforcer_l=reinforcer_l,
            dowel_d=self.dowel_d,
            roll_d=roll_d,
            roll_l=self.roll_l,
            inner_h=self.inner_h,
            skadis_mounted_h=self.skadis_mounted_h,
            guide_h=self.guide_h,
            guide_opening=self.guide_opening,
            roof_d=roof_d,
        )

    def setup(self):
        super().setup()

        s = SkadisSettings(
            self.thickness, True, **self.edgesettings.get("Skadis", {})
        )
        s.edgeObjects(self)
        self.wallHolesAt = self.edges["|"]


    def apply_preset(self):
        if self.preset == "test":
            self.burn = 0
            self.thickness = 3.175 # 1/8"
            # for Skadis match
            hole_aim = 60 # 20mm + 2*40mm
            skadis_hole = 5.0
            # self.roll_d = hole_aim - self.thickness + (skadis_hole - self.thickness) / 2
            self.roll_d = hole_aim - self.thickness + (skadis_hole - self.thickness) / 2

            self.roll_l = 30
            self.dowel_d = 15
            self.roof_d = 44
            self.skadis_mounted_h = 50
            self.inner_h = 90
            self.guide_h = 40
            self.guide_opening = 15
            # core ID: about 43 mm
        elif self.preset == "demo":
            self.burn = 0
            self.thickness = 5

            #self.roll_d = 126 # actual 123

            # for Skadis match
            hole_aim = 140 # 20mm + 3*40mm (3 Skadis spaces)
            skadis_hole = 5.0
            self.roll_d = hole_aim - self.thickness + (skadis_hole - self.thickness) / 2

            self.roll_l = 280
            self.dowel_d = 33
            self.roof_d = 88
            # self.skadis_mounted_h = 90  <---
            self.skadis_mounted_h = 70
            self.inner_h = 310
            self.guide_h = 30
            self.guide_opening = 15
            # core ID: about 43 mm
        else:
            assert self.preset == ""

    @inject_shortcuts
    def roof(self, roof_d: float, roll_d: float):
        solid = Element.from_item(
            self.wall_builder("roof").add(
                Plain(self.thickness * 2),
                Edge(roof_d, FINGER_HOLE_EDGE),
                Turn(90),
                Plain(roof_d + self.thickness * 2),
                Turn(90),
                Plain(roof_d + self.thickness * 2),
                Turn(90),
                Edge(roof_d, FINGER_HOLE_EDGE),
                Plain(self.thickness * 2),
            )
        )
        return Element.union(self, [solid, self.pocket().translate(coord(roll_d / 2 + self.thickness * 2, roll_d / 2 + self.thickness * 2))])


    @inject_shortcuts
    def wall(self, left: bool, roll_d, skadis_mounted_h, inner_h, guide_h, roof_d, reinforcer_l):
        corner_edge = FINGER if left else FINGER_COUNTER
        w = self.wall_builder("wall")
        w.add(Plain(self.thickness * 3)) # Skadis hole clearance
        w.add(Edge(roll_d, FINGER_HOLE_EDGE))

        if not left:
            w.add(Plain(self.thickness))

        w.add(
            Turn(90),
            # TODO: this probably won't be a Skadis edge?
            # Edge(skadis_mounted_h, corner_edge),
            # Edge(inner_h - skadis_mounted_h, corner_edge),
            Edge(inner_h + 2 * self.thickness, corner_edge),

            Turn(90),
        )
        if not left:
            w.add(Plain(self.thickness))

        w.add(
            Edge(roof_d, FINGER),
            Turn(90),
            Plain(inner_h - skadis_mounted_h),
            Turn(-90),
            Plain(roll_d - roof_d),
        )
        w.add(Plain(self.thickness * 3))
        w.add(
            Turn(90),
            #Plain(skadis_mounted_h - guide_h + 2 * self.thickness),
            #Edge(guide_h, FINGER_HOLE_EDGE),
            #Edge(guide_h, 'd'),
            Edge(skadis_mounted_h + 2 * self.thickness, 'd'),
        )
        element = Element.from_item(w)

        def extra():
            n_slots = ceil(roll_d // 40)

            for i in range(n_slots + 1):
                self.wallHolesAt(
                    self.thickness * 1.5 + (40 * i),
                    40,
                    reinforcer_l,
                    90,
                )

            for i in range(n_slots + 1):
                self.wallHolesAt(
                    self.thickness * 1.5 + (40 * i) + 20,
                    20,
                    reinforcer_l + 20,
                    90,
                )


        element.add_render(extra)

        return element


    def reinforcer(self, l: float):
        # TODO: must leave at least 1 thickness down for floor guide!
        reinforcer_d = 10
        reinforcer_hp = 5
        return Element.from_item(
            self.wall_builder("reinforcer").add(
                Plain(reinforcer_d),
                Turn(90),
                Plain(l + 2 * reinforcer_hp),
                Turn(90),
                Plain(reinforcer_d),
                Turn(90),
                Plain(reinforcer_hp),
                Turn(-90),
                Plain(self.thickness),
                Turn(90),
                Edge(l, 'b'),
                Plain(reinforcer_hp),
            )
        )



    @inject_shortcuts
    def floor(self, roll_d: float, guide_opening: float):
        return Element.from_item(
            self.wall_builder("floor").add(
                Edge(roll_d, FINGER),
                Turn(90),
                Edge(roll_d - guide_opening, FINGER),
                Plain(guide_opening),
                Turn(90),
                Plain(guide_opening),
                Edge(roll_d - guide_opening, FINGER),
                Turn(90),
                Edge(roll_d, FINGER),
            )
        )

    @inject_shortcuts
    def pocket(self, dowel_d: float):
        def render():
            # Starts centered.
            self.hole(0, 0, d=dowel_d)

        r = dowel_d / 2
        return Element(
            position=coord(0, 0),
            bbox=BBox(minx=-r, miny=-r, maxx=r, maxy=r),
            render=[render],
            boxes=self,
            is_part=None,
            color=None,
        )


    @inject_shortcuts
    def floor_guide(self, roll_d: float, dowel_d: float):
        solid = Element.from_item(
            self.wall_builder("floor guide").add(
                Plain(roll_d),
                Turn(90),
                Plain(roll_d),
                Turn(90),
                Plain(roll_d),
                #Edge(roll_d, FINGER),
                Turn(90),
                Plain(roll_d),
                #Edge(roll_d, FINGER),
            )
        )
        return Element.union(self, [solid, self.pocket().translate(coord(roll_d / 2, roll_d / 2))])

    @inject_shortcuts
    def guide_wall(self, roll_d: float, guide_h: float, guide_opening: float):
        return Element.from_item(
            self.wall_builder("guide wall").add(
                Edge(roll_d - guide_opening, FINGER_HOLE_EDGE),
                Turn(90),
                Plain(guide_h + 2 * self.thickness),
                Turn(90),
                Plain(roll_d - guide_opening - roll_d / 2),
                Turn(-90),
                Plain(self.skadis_mounted_h - guide_h),
                Turn(90),
                Plain(roll_d / 2),
                Turn(90),
                #Edge(guide_h, FINGER),
                # Edge(guide_h, 'd'),
                Edge(self.skadis_mounted_h + 2 * self.thickness, 'b'),
            )
        )
        # 'b' = wall joined edge

    #@inject_shortcuts
    #def front_frame(self, front_frame_border, front_frame_w, front_frame_h):
    #    # copied from split PhotoFrame.split front
    #    hypo = sqrt(2 * front_frame_border**2)
    #    dm = Plain(self.dovetail_margin)
    #    dove = hypo - 2 * self.dovetail_margin
    #    assert dove >= 0

    #    # COUNTER: depth of cutouts based on thickness of sides.
    #    outer_edge = FRONT_TO_MIDDLE_FINGER_COUNTER

    #    # d is dovetail joints
    #    top_dove = [dm, Edge(dove, FRONT_DOVETAIL), dm]
    #    top_bottom = Element.from_item(
    #        self.wall_builder("front frame top/bottom").add(
    #            Plain(self.front_middle_finger_margin),
    #            Edge(front_frame_w - (2 * self.front_middle_finger_margin), outer_edge),
    #            Plain(self.front_middle_finger_margin),
    #            Turn(90 + 45),
    #            *top_dove, Turn(90 - 45),
    #            Plain(self.window_w), Turn(90 - 45),
    #            *top_dove, Close()
    #        )
    #    )
    #    # D is dovetail joints counterpart
    #    side_dove = [dm, Edge(dove, FRONT_DOVETAIL_COUNTER), dm]
    #    side = Element.from_item(
    #        self.wall_builder("front frame left/right").add(
    #            Plain(self.front_middle_finger_margin),
    #            Edge(front_frame_h - (2 * self.front_middle_finger_margin), outer_edge),
    #            Plain(self.front_middle_finger_margin),
    #            Turn(90 + 45),
    #            *side_dove, Turn(90 - 45),
    #            Plain(self.window_h), Turn(90 - 45),
    #            *side_dove,
    #            Close()
    #        )
    #    )
    #    return self.ystack(
    #        top_bottom, top_bottom,
    #        side, side,
    #    )

    @inject_shortcuts
    def build(self, reinforcer_l):
        #print(f"Window: {fmt_mmxmm(self.window_w, self.window_h)}")

        return self.xstack(self.wall(True), self.wall(False), self.ystack(
            self.floor(),
            self.floor_guide(),
            self.guide_wall(),
            self.guide_wall(),
            self.reinforcer(reinforcer_l),
            self.reinforcer(reinforcer_l + 20),
            self.roof(),
            #self.xstack(self.glass(), self.backing()),
            #self.front_frame(),
            #self.middle_frame(),
        ))
