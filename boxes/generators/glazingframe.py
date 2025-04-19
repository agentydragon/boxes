"""
To run:
    scripts/boxes GlazingFrame --preset=demo
"""

from boxes import Boxes, Color
from math import sqrt
from boxes.edges import FingerJointSettings, FingerJointEdge, FingerJointEdgeCounterPart, MountingSettings
from boxes.edges import DoveTailSettings, DoveTailJoint, DoveTailJointCounterPart, FingerJointBase
from boxes.fmt import (
    fmt_mm,
    fmt_mmxmm,
)
from boxes.generators.raibase import (
    PLAIN,
    Edge,
    Element,
    Plain,
    mark,
    RaiBase,
    Close,
    Turn,
    FINGER,
    FINGER_COUNTER,
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
        import logging
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
            "--front_w",
            action="store",
            type=float,
            help="Width of front frame in mm"
        )
        self.front_w: float
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
        self.argparser.add_argument(
            "--middle_h",
            action="store",
            type=float,
            help="Height of middle frame"
        )
        self.middle_h: float
        self.argparser.add_argument(
            "--content_w",
            action="store",
            type=float,
            help="Width of content (backing / glass)",
        )
        self.content_w: float
        self.argparser.add_argument(
            "--content_h",
            action="store",
            type=float,
            help="Height of content (backing / glass)",
        )
        self.content_h: float
        self.argparser.add_argument(
            "--content_t",
            action="store",
            type=float,
            help="Combined thickness of the content (backing + picture + glass)",
        )
        self.argparser.add_argument(
            "--points_w",
            action="store",
            type=int,
            help="Number of glazing points along the width",
        )
        self.argparser.add_argument(
            "--points_h",
            action="store",
            type=int,
            help="Number of glazing points along the height",
        )
        self.points_h: int

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


    @property
    def shortcuts(self):
        return dict(
            front_w=self.front_w,
            middle_h=self.middle_h,
            front_frame_w=self.window_w + 2 * self.front_w,
            front_frame_h=self.window_h + 2 * self.front_w,
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
        )
        self.edges[MIDDLE_MIDDLE_FINGER] = FingerJointEdge(self, middle_middle_finger_settings)
        self.edges[MIDDLE_MIDDLE_FINGER_COUNTER] = FingerJointEdgeCounterPart(self, middle_middle_finger_settings)

        # Recess in front frame to accommodate middle -> recess depth governed by thickness of middle, not front
        front_middle_settings = FingerJointSettings(
            thickness=self.front_t,
            relative=True,
            **self.edgesettings.get("FingerJoint", {}),
            surroundingspaces=0,  # We handle this ourselves.
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
        if self.preset == "demo":
            self.window_w = 90
            self.window_h = 130
            self.front_w = 15
            self.front_t = 3.175  # 1/8"
            self.middle_t = 5
            self.content_w = 100
            self.content_h = 140
            self.content_t = 5
            self.points_w = 3
            self.points_h = 4
            self.dovetail_margin = 1.0
            self.middle_h = 6
            self.front_middle_finger_margin = 7.5  # 2.0
        else:
            assert self.preset == ""

    def content_rectangle_path(self):
        return [
            Plain(self.content_w, text=mark("content_w")), Turn(90),
            Plain(self.content_h, text=mark("content_h")), Turn(90),
            Plain(self.content_w), Turn(90),
            Plain(self.content_h), Close()
        ]

    def glass(self):
        text = f"glass {fmt_mmxmm(self.content_w, self.content_h)}"
        return Element.from_item(self.wall_builder(text).add(self.content_rectangle_path()))


    def backing(self):
        text = f"backing\ncontent {fmt_mmxmm(self.content_w, self.content_h)}\nwindow {fmt_mmxmm(self.window_w, self.window_h)}"
        backing = Element.from_item(self.wall_builder(text).add(self.content_rectangle_path()))

        w = self.wall_builder("backing_etching").add(
            Plain(self.window_w, text=mark("window_w")), Turn(90),
            Plain(self.window_h, text=mark("window_h")), Turn(90),
            Plain(self.window_w), Turn(90),
            Plain(self.window_h), Close()
        )
        delta = coord(
            (self.content_w - self.window_w) / 2,
            (self.content_h - self.window_h) / 2,
        )
        etching = Element.from_item(w, color=Color.ETCHING).translate(delta)
        return Element.union(self, [backing, etching])

    @inject_shortcuts
    def front_frame(self, front_w, front_frame_w, front_frame_h):
        # copied from split PhotoFrame.split front
        hypo = sqrt(2 * front_w**2)
        dm = Plain(self.dovetail_margin)
        dove = hypo - 2 * self.dovetail_margin
        assert dove >= 0

        # COUNTER: depth of cutouts based on thickness of sides.
        outer_edge = FRONT_TO_MIDDLE_FINGER_COUNTER

        # d is dovetail joints
        top_dove = [dm, Edge(dove, FRONT_DOVETAIL), dm]
        top_bottom = Element.from_item(
            self.wall_builder("front frame top/bottom").add(
                Plain(self.front_middle_finger_margin),
                Edge(front_frame_w - (2 * self.front_middle_finger_margin), outer_edge),
                Plain(self.front_middle_finger_margin),
                Turn(90 + 45),
                *top_dove, Turn(90 - 45),
                Plain(self.window_w), Turn(90 - 45),
                *top_dove, Close()
            )
        )
        # D is dovetail joints counterpart
        side_dove = [dm, Edge(dove, FRONT_DOVETAIL_COUNTER), dm]
        side = Element.from_item(
            self.wall_builder("front frame left/right").add(
                Plain(self.front_middle_finger_margin),
                Edge(front_frame_h - (2 * self.front_middle_finger_margin), outer_edge),
                Plain(self.front_middle_finger_margin),
                Turn(90 + 45),
                *side_dove, Turn(90 - 45),
                Plain(self.window_h), Turn(90 - 45),
                *side_dove,
                Close()
            )
        )
        return self.ystack(
            top_bottom, top_bottom,
            side, side,
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
            color=Color.ETCHING, # TODO
        )

    def make_grooves(self, length, count):
        return Element.union(self, [
            self.v_groove(2).translate(
                coord((length / (count + 1)) * i, 0)
            )
            for i in range(1, count + 1)
        ])

    @inject_shortcuts
    def middle_frame(self, front_frame_w, front_frame_h, middle_h):
        # TODO:
        #  - those pieces are made from a different thickness!

        # FINGER: finger length based on thickness of front side.
        frame_edge = MIDDLE_TO_FRONT_FINGER

        # override fingerLength on fingerJointBase?

        assert self.front_middle_finger_margin >= self.middle_t, f"Need to keep enough free space without fingers for a full thickness"
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
            )
        )
        groove_offset = coord(0, self.content_t)
        top_bottom = Element.union(self, [
            top_bottom,
            self.make_grooves(front_frame_w, self.points_w).translate(groove_offset)
        ])
        side = Element.from_item(
            self.wall_builder("middle frame sides").add(
                Plain(self.middle_t),
                r,
                Edge(front_frame_h - 2 * (self.front_middle_finger_margin), frame_edge),
                r,
                Turn(90),
                Edge(middle_h, MIDDLE_MIDDLE_FINGER), Turn(90),
                Plain(front_frame_h - self.middle_t), Turn(90),
                Edge(middle_h, MIDDLE_MIDDLE_FINGER_COUNTER),# Close()
            )
        )
        side = Element.union(self, [
            side,
            self.make_grooves(front_frame_h, self.points_h).translate(groove_offset)
        ])

        # check there's enough margin above content for glazing points
        remaining = self.middle_h - self.content_t

        # by https://www.hardwareworld.com/p10k82f/Glazing-Push-Points
        # they actually claim 0.375 mm
        assert remaining >= 1.0, f"Only {fmt_mm(remaining)} left for glazing point wings - make the middle frame higher"

        assert self.middle_t >= 3, f"Need enough middle thickness for 2.85 mm bite from glazing point"
        # TODO: chatgpt recommends <= 4mm

        return self.ystack(
            side,
            side,
            top_bottom,
            top_bottom,
        )

    def build(self):
        """
        Render the frame parts:
         - The front frame layer or 'picture window'.
         - The backing layer with edges or notches for glazing points.
         - Possibly extra geometry for channel edges or a separate middle layer.
        """

        print(f"Window: {fmt_mmxmm(self.window_w, self.window_h)}")
        print(f"Front frame: {fmt_mm(self.front_w)} around the window, {fmt_mm(self.front_t)} thick")
        print(f"Middle frame: {fmt_mm(self.middle_h)} deep, {fmt_mm(self.middle_t)} thick")
        print(f"Content: {fmt_mmxmm(self.content_w, self.content_h)}, {fmt_mm(self.content_t)} thick")

        # TODO: apply: front_t, middle_t, points_w, points_h


        # Render the backing
        return self.ystack(
            #self.xstack(self.glass(), self.backing()),
            self.backing(),
            #self.front_frame(),
            #self.middle_frame(),
        )

        # TODO: front frame pieces
        # TODO: back frame pieces
