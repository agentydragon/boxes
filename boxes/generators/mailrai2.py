"""

scripts/boxes MailRai

left & right

TODO: edges.CompoundEdge

"""

import contextlib
import dataclasses
import itertools
import math
import random

from boxes import BoolArg, Boxes, Color, argparseSections, boolarg, restore
from boxes.edges import (
    FingerHoles,
    FingerJointEdge,
    FingerJointEdgeCounterPart,
    FingerJointSettings,
)

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


@dataclasses.dataclass
class Instr:
    length: float
    angle: float
    edge: str


@dataclasses.dataclass
class WallBuilder:
    boxes: Boxes
    instr: list[Instr] = dataclasses.field(default_factory=list)

    def add(self, *, length: float, angle: float, edge: str):
        self.instr.append(Instr(length=length, angle=angle, edge=edge))

    def get_borders(self):
        borders = []
        for instr in self.instr:
            borders.extend((instr.length, instr.angle))
        return borders

    def get_edges(self):
        return [i.edge for i in self.instr]

    def total_width(self):
        minx, _, maxx, _ = self.bbox()
        return maxx - minx

    def bbox(self):
        edges = self.get_edges()
        edges = [self.boxes.edges[e] for e in edges]
        borders = self.boxes._closePolygon(self.get_borders())
        minx, miny, maxx, maxy = self.boxes._polygonWallExtend(borders, edges)
        return minx, miny, maxx, maxy

    def total_height(self):
        _, miny, _, maxy = self.bbox()
        return maxy - miny

    def render(self, move=None, callback=None, turtle=False):
        for i, instr in enumerate(self.instr):
            msg = f"  {i=} length={instr.length}"
            if instr.angle:
                msg += f" angle={instr.angle}"
            print(msg)
        self.boxes.polygonWall(
            self.get_borders(),
            edge=self.get_edges(),
            correct_corners=True,
            callback=callback,
            move=move,
            turtle=turtle,
        )

    def add_spaced_segment(self, lengths, inner_edge):
        for length in lengths:
            self.add(length=self.boxes.thickness, angle=0, edge="e")
            self.add(length=length, angle=0, edge=inner_edge)
        self.add(length=self.boxes.thickness, angle=90, edge="e")


class MailRai2(Boxes):
    def __init__(self):
        super().__init__()

        self.argparser.add_argument(
            "--section_heights",
            action="store",
            type=argparseSections,
            # default="40 50 60",  # 40, 50: approx OK
            default="20 30 40",  # 40, 50: approx OK
            help="Section heights, *inner*, from bottom to top.",
        )
        self.argparser.add_argument(
            "--section_widths",
            action="store",
            type=argparseSections,
            # default="50 60 70",  # approx OK
            default="30 40 50",  # approx OK
            help="Section widths, *inner*, left to right.",
        )
        self.argparser.add_argument(
            "--angle_degrees",
            action="store",
            type=float,
            default="60",
            help="Angle of fronts. 0 = horizontal, 90 = vertical.",
        )
        self.argparser.add_argument(
            "--front_length",
            action="store",
            type=float,
            # default="90",
            default="40",
            help="Front length, *inner*.",
        )
        self.argparser.add_argument(
            "--floor_inner",
            action="store",
            type=float,
            # default="30",
            default="20",
            help="Floor depth, inner",
        )
        self.section_heights: list[float]
        self.section_widths: list[float]

    @property
    def angle(self):
        return math.radians(self.angle_degrees)

    """
    polygonWall 'f' with (100, 90, 200, 90, 100, 90, 200, 90):
      *inner* width 10cm, *inner* height 20cm
    
    polygonWall 'F' with (100, 90, 200, 90, 100, 90, 200, 90):
      *outer* width 10cm, *outer* height 20cm
    
    on both 'f' and 'F', sequence of (25, 25, 25, 25) has same length as (100)

    edge:
        e = plain
    """

    @contextlib.contextmanager
    def moved(self, move, bbox, label=None):
        minx, miny, maxx, maxy = bbox
        width = maxx - minx
        height = maxy - miny

        if self.move(width, height, move, before=True):
            return True
        self.moveTo(-minx, -miny)
        yield

        self.move(width, height, move, label=label)

    def fingerholes(self, x, y, length, angle, *args, **kwargs):
        print("fingerholes")
        with self.saved_context():
            self.moveTo(x, y, angle)
            self.set_source_color(COLORS[random.randrange(len(COLORS))])
            self.ctx.line_to(length, 0)
            # self.ctx.rectangle(0, -2, length, 4)
            self.fingerHolesAt(0, 0, length, 0, *args, **kwargs)

    def show_cc(self, i):
        c = COLORS[i % len(COLORS)]
        self.ctx.set_source_rgb(*c)
        self.text(str(i), color=c, fontsize=5)
        self.circle(0, 0, r=2)

    @property
    def top_compartment_height(self):
        return math.sin(self.angle) * self.front_length

    def back(self, move=None):
        """
        Back:
          width:
            <T> section_width <T> section_width ... <T> section_width <T>
          height:
            <T> section_height <T> section_height ... <T> section_height  (no final thickness)
        """
        wall = WallBuilder(boxes=self)

        wall.add_spaced_segment(self.section_widths, "F")
        # TODO: this one should have spaced only under...
        wall.add_spaced_segment(self.section_heights, "F")
        wall.instr[-1].angle = 0
        wall.add(length=self.top_compartment_height, angle=90, edge="e")

        wall.add(
            length=sum(self.section_widths)
            + (len(self.section_widths) + 1) * self.thickness,
            angle=90,
            edge="e",
        )
        wall.add(length=self.top_compartment_height, angle=0, edge="e")
        wall.add_spaced_segment(reversed(self.section_heights), "F")

        # with self.saved_context():
        #    self.set_source_color((0, 255, 0))
        #    self.circle(0, 0, r=2)

        # with contextlib.nullcontext():
        with self.moved(move, wall.bbox(), label="back"):
            wall.render(callback=self.show_cc, turtle=True)

            self.set_source_color(Color.ANNOTATIONS)

            # TODO: those finger holes are weirdly positioned...

            # Horizontal left-to-right finger holes
            # fingerHolesAt is centered on the line.

            x_coords = self.x_starts
            y_coords = self.y_starts
            for (i, (x1, x2)), (j, (y1, y2)) in itertools.product(
                enumerate(zip(x_coords, x_coords[1:])),
                enumerate(zip(y_coords, y_coords[1:])),
            ):
                with self.saved_context():
                    self.moveTo(x1, y1)
                    if j > 0:
                        # settings = FingerJointSettings(
                        #    thickness=self.thickness,
                        #    relative=True,
                        #    **self.edgesettings.get("FingerJoint", {}),
                        #    angle=60,  # self.angle,
                        # )
                        # e = FingerHoles(self, settings)
                        # e(self.thickness / 2, 0, self.section_widths[i], 0)
                        self.fingerholes(
                            self.thickness / 2, 0, self.section_widths[i], 0
                        )
                    if i > 0:
                        self.fingerholes(
                            0,
                            self.thickness / 2,
                            (self.section_heights + [self.top_compartment_height])[j],
                            90,
                        )

            # TODO: assert settings of fingers .width == 1.0

            # settings = FingerJointSettings(angle = self.angle)
            # shift from left by width=70mm + self.thickness=3.0mm
            # horizontal distance from left edge of wall to left edge of finger holes: 7.366
            # middle of finger holes is at ~7.51 cm

            # if no "move" => will return to initial position at the end (= bottom left)
            # move "right" => will end in bottom right, etc.

    @property
    def cutout_a(self):
        return self.floor_inner - self.floor_cutout

    @property
    def cutout_b(self):
        return self.front_length - 10  # 50

    def side(self, outer: bool, move=None):
        label = "outer side piece" if outer else "inner vertical divider"
        wall = WallBuilder(boxes=self)
        wall.add(length=self.floor_inner, angle=self.angle_degrees, edge="F")
        wall.add(length=self.front_length, angle=90 - self.angle_degrees, edge="f")

        wall.add_spaced_segment(self.section_heights, "f")
        # if outer:
        #    wall.add_spaced_segment(self.section_heights, "f")
        # else:
        #    cutout_size = self.thickness / math.cos(self.angle)
        #    lengths = self.section_heights

        #    self.add(length=self.thickness, angle=0, edge="e")
        #    for length in lengths[:-1]:
        #        self.add(length=length, angle=0, edge="f")
        #        self.add(length=self.thickness, angle=0, edge="e")

        #    self.add(length=lengths[-1], angle=0, edge="f")
        #    self.add(length=self.thickness, angle=90, edge="e")

        wall.add(
            length=(math.cos(self.angle) * self.front_length) + self.floor_inner,
            angle=90,
            edge="e",
        )
        # going down
        wall.add_spaced_segment(
            reversed(self.section_heights + [self.top_compartment_height]), "f"
        )

        with self.moved(move, wall.bbox(), label=label):
            # finger holes for floors dn fronts
            with self.saved_context():
                wall.render(turtle=True, callback=self.show_cc)
            if outer:
                for height in self.y_starts[1:-1]:
                    with self.saved_context():
                        self.moveTo(0, height)
                        self.fingerholes(0, 0, self.floor_inner, 0)
                        self.moveTo(self.floor_inner, 0)
                        self.fingerholes(0, 0, self.front_length, self.angle_degrees)
            else:
                # go to cut out middle
                for i, y in list(enumerate(self.y_starts[1:-1])):
                    with self.saved_context():
                        depth_a = self.floor_inner - 10  # self.floor_inner / 2
                        depth_b = self.front_length  # 50
                        cutout_height = self.thickness * 1.1

                        delta = cutout_height * math.sin(
                            math.radians(45 - self.angle_degrees / 2)
                        )
                        a1 = self.cutout_a - delta
                        a2 = self.cutout_a + delta
                        b1 = self.cutout_b - delta
                        b2 = self.cutout_b + delta
                        print(f"{a1=:.2f} {a2=:.2f} {b1=:.2f} {b2=:.2f}")

                        self.moveTo(0, y - self.thickness)
                        # self.corner(-360, 5)
                        # self.corner(-360, 1)
                        self.moveTo(self.floor_inner - depth_a, 0, 0)
                        # self.corner(-360, 5)
                        # self.corner(-360, 1)
                        self.moveTo(0, -cutout_height / 2, 0)
                        self.edge(a2)
                        self.corner(self.angle_degrees)
                        self.edge(b2)
                        self.corner(90)
                        self.edge(cutout_height)
                        self.corner(90)
                        self.edge(b1)
                        self.corner(-self.angle_degrees)
                        self.edge(a1)
                        self.corner(90)
                        self.edge(cutout_height)

                        # for d in (dy, dx, dy, dx / 2.0 + r):
                        #    self.corner(-90, r)
                        #    self.edge(d - 2 * r)
                    # self.rectangularHole(0, y, 10, 10)
                    # with self.saved_context():
                    # self.moveTo(0, y, 0)
                    # self.text(f"L{i} {y=:.2f}", fontsize=5)
                    # self.edge(30)
                    # self.corner(90, 4)
                    # self.edge(30)
                    # self.corner(90, 4)
                    # self.edge(30)
                    # self.corner(90, 4)

    @property
    def x_starts(self):
        x_starts = [self.thickness / 2]
        for width in self.section_widths:
            x_starts.append(x_starts[-1] + width + self.thickness)
        return x_starts

    @property
    def y_starts(self):
        y_starts = [self.thickness / 2]
        for height in self.section_heights + [self.top_compartment_height]:
            y_starts.append(y_starts[-1] + height + self.thickness)
        return y_starts

    @property
    def floor_cutout(self):
        return 10

    def floor(self, move, first: bool):
        wall = WallBuilder(boxes=self)
        wall.add(length=self.floor_inner, angle=90, edge="f")
        # front
        wall.add_spaced_segment(self.section_widths, self.ANGLED_FINGER_JOINT_EDGE)

        wall.add(length=self.floor_inner, angle=90, edge="f")
        # back
        if first:
            wall.add_spaced_segment(reversed(self.section_widths), "f")
        else:
            lengths = list(reversed(self.section_widths))
            wall.add(length=self.thickness, angle=0, edge="e")
            for length in lengths[:-1]:
                wall.add(length=length, angle=90, edge="f")
                wall.add(length=self.floor_cutout, angle=-90, edge="e")
                wall.add(length=self.thickness, angle=-90, edge="e")
                wall.add(length=self.floor_cutout, angle=90, edge="e")

            wall.add(length=lengths[-1], angle=0, edge="f")
            wall.add(length=self.thickness, angle=90, edge="e")

        label = "bottom floor" if first else "interior floor"
        with self.moved(move, wall.bbox(), label=label):
            self.text("zero", fontsize=5)
            # finger holes for dividers
            if first:
                for i, x in list(enumerate(self.x_starts))[1:-1]:
                    with self.saved_context():
                        self.text(f"section {i}", fontsize=5, x=0, y=x)
                        self.fingerholes(0, x, self.floor_inner, 0)
            wall.render(turtle=True, callback=self.show_cc)

    ANGLED_FINGER_JOINT_EDGE = "a"
    ANGLED_FINGER_JOINT_EDGE_COUNTERPART = "A"

    def front(self, move, first: bool):
        wall = WallBuilder(boxes=self)
        # back
        wall.add_spaced_segment(
            self.section_widths, self.ANGLED_FINGER_JOINT_EDGE_COUNTERPART
        )

        wall.add(length=self.front_length, angle=90, edge="F")
        # top / front
        if first:
            wall.add_spaced_segment(reversed(self.section_widths), "e")
        else:
            lengths = list(reversed(self.section_widths))
            wall.add(length=self.thickness, angle=0, edge="e")
            for length in lengths[:-1]:
                wall.add(length=length, angle=90, edge="e")
                wall.add(length=self.floor_cutout, angle=-90, edge="e")
                wall.add(length=self.thickness, angle=-90, edge="e")
                wall.add(length=self.floor_cutout, angle=90, edge="e")
                # wall.add(length=self.thickness, angle=0, edge="e")

            wall.add(
                length=lengths[-1],
                angle=0,
                edge="e",
            )
            wall.add(length=self.thickness, angle=90, edge="e")

        wall.add(length=self.front_length, angle=90, edge="F")

        label = "first front" if first else "interior front"
        with self.moved(move, wall.bbox(), label=label):
            wall.render(turtle=True, callback=self.show_cc)

    def render(self):
        self.ctx.set_line_width(0.3)

        settings = FingerJointSettings(
            thickness=self.thickness,
            relative=True,
            **self.edgesettings.get("FingerJoint", {}),
            angle=60,  # self.angle,
        )
        self.edges[self.ANGLED_FINGER_JOINT_EDGE] = FingerJointEdge(self, settings)
        self.edges[self.ANGLED_FINGER_JOINT_EDGE_COUNTERPART] = (
            FingerJointEdgeCounterPart(self, settings)
        )

        # self.polygonWall(
        #     [25, 0, 25, 0, 25, 0, 25, 90, 200, 90, 100, 90, 200, 90],
        #     edge="f",
        #     correct_corners=True,
        #     callback=None,
        # )

        # self.polygonWall(
        #    [self.section_width * 100, 90, 200, 90, 100, 90, 200, 90],
        #    edge="FFeF",
        #    correct_corners=True,
        #    callback=None,
        # )

        # self.back(move="right")
        # self.side(move="right", outer=True)
        self.side(move="right", outer=False)
        # self.floor(move="right", first=True)
        self.floor(move="right", first=False)
        self.front(move="right", first=True)
        self.front(move="right", first=False)
