"""
To run:
    scripts/boxes GlazingFrame --preset=demo
"""

import logging
from boxes import Color
from math import sqrt, ceil
from boxes.edges import FingerJointSettings, FingerJointEdge, FingerJointEdgeCounterPart, MountingSettings
from boxes.edges import DoveTailSettings, DoveTailJoint, DoveTailJointCounterPart, FingerJointBase
from boxes.fmt import (
    fmt_mm,
    fmt_mmxmm,
)
from boxes.generators.raibase import (
    Edge,
    Element,
    Plain,
    mark,
    RaiBase,
    Close,
    Turn,
    coord,
    inject_shortcuts,
    BBox,
)

FRONT_DOVETAIL = 'a'
FRONT_DOVETAIL_COUNTER = 'A'
MIDDLE_MIDDLE_FINGER = 'b'
MIDDLE_MIDDLE_FINGER_COUNTER = 'B'
MIDDLE_TO_FRONT_FINGER = 'c'
FRONT_TO_MIDDLE_FINGER_COUNTER = 'C'

VGROOVE_COLOR = (128, 0, 128)
GLASS_COLOR = (0, 128, 128)
MIDDLE_FRAME_COLOR = (0, 0, 255)
PILOT_LINE_COLOR = (200, 200, 200)

class FingerJointEdgeCounterPartOverride(FingerJointEdgeCounterPart):
    def __init__(self, boxes, settings, finger_length_thickness_override: float):
        super().__init__(boxes, settings)
        self.finger_length_thickness_override = finger_length_thickness_override

    def fingerLength(self, angle: float) -> tuple[float, float]:
        # sharp corners
        if not (angle >= 90 or angle <= -90):
            raise NotImplementedError()
        return self.finger_length_thickness_override + self.settings.extra_length, 0.0  # type: ignore

class GlazingFrame(RaiBase):
    """
    Photo/picture frame that uses glazing points.

    This generator creates:
    - A rectangular front frame for holding an acrylic/glass panel and the thin picture.
    - A backing panel that fits behind the picture.
    - Laser-cut channels or notches for glazing points to lock the backing in place.

    Parameters:
      * x, y : size of the picture
      * glass_w, glass_h : size of the glass/acrylic
      * overlap : how much the frame overlaps the glass/acrylic
      * thickness : material thickness (in mm)
    """

    def __init__(self) -> None:
        logging.basicConfig(level=logging.INFO)
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
            "--window_w",
            action="store",
            type=float,
            help="Width of the opening through which the picture is visible",
        )
        self.window_w: float
        self.argparser.add_argument(
            "--window_h",
            action="store",
            type=float,
            help="Height of the opening through which the picture is visible",
        )
        self.window_h: float
        self.argparser.add_argument(
            "--front_frame_border",
            action="store",
            type=float,
            help="Width of front frame in mm"
        )
        self.front_frame_border: float
        self.argparser.add_argument(
            "--front_t",
            action="store",
            type=float,
            help="Thickness of front frame in mm"
        )
        self.front_t: float
        self.argparser.add_argument(
            "--middle_t",
            action="store",
            type=float,
            help="Thickness of material for middle frame"
        )
        # self.argparser.add_argument(
        #     "--middle_h",
        #     action="store",
        #     type=float,
        #     help="Height of middle frame"
        # )
        # self.middle_h: float
        self.argparser.add_argument(
            "--content_t",
            action="store",
            type=float,
            help="Combined thickness of the content (backing + picture + glass)",
        )
        #self.argparser.add_argument(
        #    "--points_w",
        #    action="store",
        #    type=int,
        #    help="Number of glazing points along the width",
        #)
        #self.argparser.add_argument(
        #    "--points_h",
        #    action="store",
        #    type=int,
        #    help="Number of glazing points along the height",
        #)
        #self.points_h: int

        self.argparser.add_argument(
            "--dovetail_margin",
            action="store",
            type=float,
            help="Diagnoal distance to keep without dovetail; mm",
        )
        self.dovetail_margin: float
        self.argparser.add_argument(
            "--front_middle_finger_margin",
            action="store",
            type=float,
            help="Horizontal distance to keep without finger joints; mm",
        )
        self.front_middle_finger_margin: float
        self.addSettingsArgs(DoveTailSettings, size=2.0, depth=1.0)

        self.middle_sand_correction = 0.72  # empirically to make 6mm birch look good


    @property
    def shortcuts(self):
        front_frame_w = self.window_w + 2 * self.front_frame_border
        front_frame_h = self.window_h + 2 * self.front_frame_border
        glazing_point_thickness = 3.2
        above_centerline = 0.5
        # actual glazing point wing thickness: 3.2 mm
        middle_h = self.content_t + glazing_point_thickness + above_centerline
        content_w, content_h = front_frame_w - 2 * self.middle_t, front_frame_h - 2 * self.middle_t
        assert content_w >= self.art_w
        assert content_h >= self.art_h
        return dict(
            front_frame_border=self.front_frame_border,
            middle_h=middle_h,
            front_frame_w=front_frame_w,
            front_frame_h=front_frame_h,
            content_w=content_w,
            content_h=content_h,
            above_centerline=above_centerline,
        )


    def setup(self):
        super().setup()

        # Dovetail in front: based on front thickness
        front_dovertail_settings = DoveTailSettings(
            thickness=self.front_t,
            relative=True,
            **self.edgesettings.get("DoveTail", {}),
        )
        self.edges[FRONT_DOVETAIL] = DoveTailJoint(self, front_dovertail_settings)
        self.edges[FRONT_DOVETAIL_COUNTER] = DoveTailJointCounterPart(self, front_dovertail_settings)

        # Middle<->middle fingers
        middle_middle_finger_settings = FingerJointSettings(
            thickness=self.middle_t,
            relative=True,
            **self.edgesettings.get("FingerJoint", {}),

            # add 0.2 mm for burn marks on fingers
            extra_length=(self.middle_sand_correction / self.middle_t),
        )
        self.edges[MIDDLE_MIDDLE_FINGER] = FingerJointEdge(self, middle_middle_finger_settings)
        self.edges[MIDDLE_MIDDLE_FINGER_COUNTER] = FingerJointEdgeCounterPart(self, middle_middle_finger_settings)

        # Recess in front frame to accommodate middle -> recess depth governed by thickness of middle, not front
        front_middle_settings = FingerJointSettings(
            thickness=self.front_t,
            relative=True,
            **self.edgesettings.get("FingerJoint", {}),
            surroundingspaces=0,  # We handle this ourselves.

            # add 0.2 mm for burn marks on fingers
            # TODO: make parametric
            extra_length=(self.middle_sand_correction / self.front_t),

            # xxx: default: space=2, finger=2
            # "finger" will be part of side frame appearing on front frame.
            # if smaller -> less sanding of burn marks
            space=self.front_middle_fingerjoint_space,
            finger=self.front_middle_fingerjoint_finger,
        )
        # Fingers from middle to front: pretend to use front thickness,
        # even if it's on middle material.
        self.edges[MIDDLE_TO_FRONT_FINGER] = FingerJointEdge(self, front_middle_settings)
        self.edges[FRONT_TO_MIDDLE_FINGER_COUNTER] = FingerJointEdgeCounterPartOverride(
            self,
            front_middle_settings,
            finger_length_thickness_override=self.middle_t,
        )


    def apply_preset(self):
        ACRYLITE_T = 2.94

        if self.preset == "gold-mongrelist":
            # "gold mongrelist" pic: 27.8 x 43.2 cm
            # minus 3mm on all sides for buffer
            self.art_w, self.art_h = 278, 432
            self.window_w, self.window_h = self.art_w - 6, self.art_h - 6

            thin_ply_t = 3.0

            # TODO: deeper groove

            self.front_frame_border = 19
            self.front_t = thin_ply_t
            self.middle_t = 6
            self.content_t = ACRYLITE_T + thin_ply_t

            # self.points_w = self.points_h = 3
            self.dovetail_margin_outer = 2.5
            self.dovetail_margin_inner = 0
            self.front_middle_finger_margin = 15

            self.front_middle_fingerjoint_space = 2
            self.front_middle_fingerjoint_finger = 2
        elif self.preset == "mongrelist-crying":
            # "mongrelist crying" pic: 22.7x30.5 cm
            self.window_w, self.window_h = 227, 305
            self.art_w, self.art_h = self.window_w, self.window_h

            thin_ply_t = 3.0

            self.front_frame_border = 22 # "10% of short edge" = "classic no-mat proportion"
            self.front_t = thin_ply_t  # 1/8"
            self.middle_t = 5.17
            self.content_t = ACRYLITE_T + thin_ply_t

            # self.points_w = self.points_h = 3
            self.dovetail_margin = 1
            self.front_middle_finger_margin = 15

            self.front_middle_fingerjoint_space = 2
            self.front_middle_fingerjoint_finger = 2
            #self.front_middle_fingerjoint_space = 12
            #self.front_middle_fingerjoint_finger = 6
        elif self.preset == "memento-mori":
            # "memento mori / memento vire" pics: 28x43 cm
            self.window_w, self.window_h = 280, 430
            self.art_w, self.art_h = self.window_w, self.window_h

            thin_ply_t = 3.0

            self.front_frame_border = 22
            self.front_t = thin_ply_t  # 1/8"
            self.middle_t = 6.34
            self.content_t = ACRYLITE_T + thin_ply_t

            #self.points_w = self.points_h = 3
            self.dovetail_margin = 1
            self.front_middle_finger_margin = 15

            self.front_middle_fingerjoint_space = 12
            self.front_middle_fingerjoint_finger = 6
        elif self.preset == "run":
            # run test: 20x20 mm window
            self.window_w, self.window_h = 30, 20
            self.art_w, self.art_h = self.window_w, self.window_h

            self.front_frame_border = 15
            self.front_t = 3.175  # 1/8"
            self.middle_t = 6.34

            thin_ply_t = 3.25

            self.content_t = ACRYLITE_T + thin_ply_t

            #self.points_w = self.points_h = 2
            self.dovetail_margin = 1
            #self.middle_h = 7.5
            self.front_middle_finger_margin = 10
            self.burn = 0

            # default
            self.front_middle_fingerjoint_space = 2
            self.front_middle_fingerjoint_finger = 2
        elif self.preset == "demo":
            self.burn = 0
            self.window_w, self.window_h = 90, 130
            self.art_w, self.art_h = self.window_w, self.window_h
            self.front_frame_border = 15
            self.front_t = 3.175  # 1/8"
            self.middle_t = 5
            self.content_t = 5
            # self.points_w, self.points_h = 3, 4
            self.dovetail_margin = 1.0
            #self.middle_h = 6
            self.front_middle_finger_margin = 7.5  # 2.0

            # default
            self.front_middle_fingerjoint_space = 2
            self.front_middle_fingerjoint_finger = 2
        else:
            assert self.preset == ""

    @inject_shortcuts
    def content_rectangle_path(self, content_w, content_h):
        return [
            Plain(content_w, text=mark("content_w")), Turn(90),
            Plain(content_h, text=mark("content_h")), Turn(90),
            Plain(content_w), Turn(90),
            Plain(content_h), Close()
        ]

    @inject_shortcuts
    def glass(self, content_w, content_h):
        text = f"glass {fmt_mmxmm(content_w, content_h)}"
        return Element.from_item(self.wall_builder(text).add(self.content_rectangle_path()), color=GLASS_COLOR)


    @inject_shortcuts
    def backing(self, content_w, content_h):
        text = "\n".join([
                "backing"
                f"content {fmt_mmxmm(content_w, content_h)}"
                f"art {fmt_mmxmm(self.art_w, self.art_h)}"
        ])
        backing = Element.from_item(self.wall_builder(text).add(self.content_rectangle_path()))

        w = self.wall_builder("backing_etching").add(
            Plain(self.art_w, text=mark("art_w")), Turn(90),
            Plain(self.art_h, text=mark("art_h")), Turn(90),
            Plain(self.art_w), Turn(90),
            Plain(self.art_h), Close()
        )
        delta = coord(
            (content_w - self.art_w) / 2,
            (content_h - self.art_h) / 2,
        )
        etching = Element.from_item(w, color=Color.ETCHING).translate(delta)
        return Element.union(self, [backing, etching])

    @inject_shortcuts
    def front_frame(self, front_frame_border, front_frame_w, front_frame_h):
        # copied from split PhotoFrame.split front
        hypo = sqrt(2 * front_frame_border**2)
        dm_a = Plain(self.dovetail_margin_outer)
        dm_b = Plain(self.dovetail_margin_inner)
        dove = hypo - self.dovetail_margin_outer - self.dovetail_margin_inner
        assert dove >= 0

        # COUNTER: depth of cutouts based on thickness of sides.
        outer_edge = FRONT_TO_MIDDLE_FINGER_COUNTER

        # d is dovetail joints
        top_dove = [dm_a, Edge(dove, FRONT_DOVETAIL), dm_b]
        top_bottom = Element.from_item(
            self.wall_builder("front frame top/bottom").add(
                Plain(self.front_middle_finger_margin),
                Edge(front_frame_w - (2 * self.front_middle_finger_margin), outer_edge),
                Plain(self.front_middle_finger_margin),
                Turn(90 + 45),
                *top_dove, Turn(90 - 45),
                Plain(self.window_w), Turn(90 - 45),
                *reversed(top_dove), Close()
            )
        )
        # D is dovetail joints counterpart
        side_dove = [dm_a, Edge(dove, FRONT_DOVETAIL_COUNTER), dm_b]
        side = Element.from_item(
            self.wall_builder("front frame left/right").add(
                Plain(self.front_middle_finger_margin),
                Edge(front_frame_h - (2 * self.front_middle_finger_margin), outer_edge),
                Plain(self.front_middle_finger_margin),
                Turn(90 + 45),
                *side_dove, Turn(90 - 45),
                Plain(self.window_h), Turn(90 - 45),
                *reversed(side_dove),
                Close()
            )
        )
        return self.ystack(
            top_bottom, top_bottom,
            side, side,
        )

    def pilot_line(self, size=10):
        def render():
            self.moveTo(-size/2, 0, 0)
            self.edge(size)

        return Element(
            position=coord(0, 0),
            bbox=BBox(minx=-size/2, maxx=size/2, miny=0, maxy=0),
            render=[render],
            boxes=self,
            is_part=None,
            color=PILOT_LINE_COLOR,
        )

    def v_groove(self, size=2):
        len = sqrt(2) * size

        def render():
            self.moveTo(-size, -size, 45)
            self.edge(len)
            self.corner(-90)
            self.edge(len)

        return Element(
            position=coord(0, 0),
            bbox=BBox(minx=-size, maxx=size, miny=0, maxy=size),
            render=[render],
            boxes=self,
            is_part=None,
            color=VGROOVE_COLOR,
        )

    # uniform
    # @inject_shortcuts
    # def make_grooves(self, length, count, above_centerline):
    #     return Element.union(self, [
    #         Element.union(self,
    #                       [self.v_groove(), self.pilot_line()])
    #         .translate(
    #             coord((length / (count + 1)) * i, above_centerline)
    #         )
    #         for i in range(1, count + 1)
    #     ])

    @inject_shortcuts
    def make_grooves(self, length, above_centerline,
                     edge_clearance=25, max_spacing=120):
        """
        Place glazing‑point grooves:
          ▸ fixed clearance `edge_clearance` from both ends (default 25 mm)
          ▸ no clear span > `max_spacing` (default 120 mm)
          ▸ points distributed evenly in the remaining span
        """

        usable = max(0, length - 2*edge_clearance)          # interior span
        n      = max(1, ceil(usable / max_spacing))    # points per side
        step   = usable / (n + 1)                           # even spacing

        return Element.union(self, [
            Element.union(self, [self.v_groove(), self.pilot_line()])
                   .translate(coord(edge_clearance + step*i, above_centerline))
            for i in range(1, n + 1)
        ])



    @inject_shortcuts
    def middle_frame(self, front_frame_w, front_frame_h, middle_h):
        # TODO:
        #  - those pieces are made from a different thickness!

        # FINGER: finger length based on thickness of front side.
        frame_edge = MIDDLE_TO_FRONT_FINGER

        # override fingerLength on fingerJointBase?

        assert self.front_middle_finger_margin >= self.middle_t, f"Need to keep enough free space without fingers (front_middle_finger_margin={fmt_mm(self.front_middle_finger_margin)}) for a full thickness (middle_t={fmt_mm(self.middle_t)})"
        r = Plain(self.front_middle_finger_margin - self.middle_t)
        top_bottom = Element.from_item(
            self.wall_builder("middle frame top/bottom").add(
                Plain(self.middle_t),
                r,
                Edge(front_frame_w - (2 * self.front_middle_finger_margin), frame_edge),
                r,
                Turn(90),
                Edge(middle_h, MIDDLE_MIDDLE_FINGER), Turn(90),
                Plain(front_frame_w - self.middle_t), Turn(90),
                Edge(middle_h, MIDDLE_MIDDLE_FINGER_COUNTER), #Close()
            ),
            color=MIDDLE_FRAME_COLOR,
        )
        groove_offset = coord(0, self.content_t)
        top_bottom = Element.union(self, [
            top_bottom,
            #self.make_grooves(front_frame_w, self.points_w).translate(groove_offset)
            self.make_grooves(front_frame_w).translate(groove_offset)
        ])
        side = Element.from_item(
            self.wall_builder("middle frame left/right").add(
                Plain(self.middle_t),
                r,
                Edge(front_frame_h - 2 * (self.front_middle_finger_margin), frame_edge),
                r,
                Turn(90),
                Edge(middle_h, MIDDLE_MIDDLE_FINGER), Turn(90),
                Plain(front_frame_h - self.middle_t), Turn(90),
                Edge(middle_h, MIDDLE_MIDDLE_FINGER_COUNTER),# Close()
            ),
            color=MIDDLE_FRAME_COLOR,
        )
        side = Element.union(self, [
            side,
            # self.make_grooves(front_frame_h, self.points_h).translate(groove_offset)
             self.make_grooves(front_frame_h).translate(groove_offset)
        ])

        #### check there's enough margin above content for glazing points
        ###remaining = self.middle_h - self.content_t

        #### by https://www.hardwareworld.com/p10k82f/Glazing-Push-Points
        #### they actually claim 0.375 mm
        ###assert remaining >= 1.0, f"Only {fmt_mm(remaining)} left for glazing point wings - make the middle frame higher"

        assert self.middle_t >= 3, f"Need enough middle thickness for 2.85 mm bite from glazing point"
        # TODO: chatgpt recommends <= 4mm

        return self.ystack(
            side,
            side,
            top_bottom,
            top_bottom,
        )

    @inject_shortcuts
    def build(self, content_w, content_h, middle_h):
        """
        Render the frame parts:
         - The front frame layer or 'picture window'.
         - The backing layer with edges or notches for glazing points.
         - Possibly extra geometry for channel edges or a separate middle layer.
        """

        print(f"Window: {fmt_mmxmm(self.window_w, self.window_h)}")
        print(f"Front frame: {fmt_mm(self.front_frame_border)} border around the window, {fmt_mm(self.front_t)} thick")
        print(f"Middle frame: {fmt_mm(middle_h)} deep, {fmt_mm(self.middle_t)} thick")
        print(f"Content: {fmt_mmxmm(content_w, content_h)}, {fmt_mm(self.content_t)} thick")

        # TODO: apply: front_t, middle_t, points_w, points_h


        # Render the backing
        return self.ystack(
            self.xstack(self.glass(), self.backing()),
            self.front_frame(),
            self.middle_frame(),
        )

        # TODO: front frame pieces
        # TODO: back frame pieces
