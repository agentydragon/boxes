"""
scripts/boxes MailRack --debug=True --reference 0

"""

from __future__ import annotations

import logging
from math import cos, radians, sin, sqrt, tan

from hamcrest import assert_that, close_to

from boxes import argparseSections, boolarg, restore
from boxes.edges import FingerJointEdge, FingerJointEdgeCounterPart, FingerJointSettings
from boxes.generators.raibase import (
    ALPHA_SIGN,
    DEG_SIGN,
    Compound,
    Element,
    RaiBase,
    Section,
    SkippingFingerJoint,
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
SKIP_EVEN = "a"
SKIP_ODD = "A"
SKIP_ODD_REVERSE = "@"


def mark(s):
    return DOWN_ARROW + s + DOWN_ARROW


def make_sections(xs, name, edge):
    return [Section(x, edge, text=f"{name}{i}") for i, x in enumerate(xs)]


class MailRack(RaiBase):
    def __init__(self):
        super().__init__()

        # sx: inner widths of sections

        self.buildArgParser(sh="40*2", sx="40*2")
        self.add_float_arg("alpha_deg", 60)
        self.add_float_arg("side_angled_length", 60)
        self.add_float_arg("floor_depth", 30)  # 'd'

    @property
    def shortcuts(self):
        alpha_rad = radians(self.alpha_deg)
        f = self.finger_hole_width
        sh = self.sh
        d = self.floor_depth
        a = self.side_angled_length
        sin_a, cos_a = sin(alpha_rad), cos(alpha_rad)

        hat_length = (d + a * cos_a) / sin_a

        # right-angle triangles:
        # 1. front length, hat length, [diagonal]
        # 2. floor depth, (hat height + spacer), [diagonal]
        hat_height = sqrt(a**2 + hat_length**2 - d**2)

        sheff = sh + [hat_height]

        zigzags: list[tuple[float, float]] = [
            tuple(coord(cos_a, sin_a) * (h + f)) for h in sh
        ]
        return dict(
            f=f,
            sx=self.sx,
            sh=sh,
            alpha_deg=self.alpha_deg,
            d=d,
            a=a,
            sin_a=sin_a,
            cos_a=cos_a,
            angle_rad=alpha_rad,
            hat_length=hat_length,
            hat_height=hat_height,
            sheff=sheff,
            gap=Section(f, PLAIN, text="f"),
            zigzags=zigzags,
        )

    def setup(self):

        # logging.getLogger("SkippingFingerJoint").setLevel(logging.DEBUG)
        # logging.basicConfig(level=logging.DEBUG)

        s = self.make_angled_finger_joint_settings(self.alpha_deg)
        self.edges[ANGLED_POS] = FingerJointEdge(self, s)
        self.edges[ANGLED_NEG] = FingerJointEdgeCounterPart(self, s)

        s = self.make_standard_finger_joint_settings()
        self.edges[SKIP_EVEN] = SkippingFingerJoint(
            self, s, idx_predicate=lambda i: i % 2 == 0
        )
        self.edges[SKIP_ODD] = SkippingFingerJoint(
            self, s, idx_predicate=lambda i: i % 2 == 1
        )
        self.edges[SKIP_ODD_REVERSE] = SkippingFingerJoint(
            self, s, idx_predicate=lambda i: i % 2 == 1, reversed=True
        )

    def render(self):
        self.setup()

        # To render all, need:
        #   - `back`
        #   - `midfloors`
        #   - `bottom``
        #   - `all_fronts`
        #   - len(sx)+1 copies of `side`

        back = self.back()
        back.do_render()

        # midfloors = self.midfloors()
        # midfloors.translate(
        #    coord(self.thickness, -midfloors.bbox.height)
        # ).do_render()

        bottom = self.bottom_floor()
        bottom = bottom.translate(
            coord(
                0, -bottom.bbox.height
            )  # -(midfloor_pieces.bbox.height + bottom.bbox.height))
        )
        bottom.do_render()

        # front = self.front()
        # front = front.translate(coord(0, -front.bbox.height - bottom.bbox.height - 10))
        # front.do_render()

        all_fronts = self.all_fronts()
        all_fronts = all_fronts.translate(
            coord(0, -all_fronts.bbox.height - bottom.bbox.height - 10)
        )
        all_fronts.do_render()

        side: Element = self.side()
        # side = side.mirror().translate(coord(-3, self.thickness))
        side = side.translate(coord(back.bbox.width + 3, self.thickness))
        side.do_render()

        # self.floor_pieces(move="right")

    @inject_shortcuts
    def side(
        self,
        a,
        alpha_deg,
        angle_rad,
        d,
        f,
        gap,
        hat_length,
        sh,
        sheff,
        zigzags,
    ) -> Element:
        w = self.wall_builder("side")
        w.add(d, alpha_deg, FINGER, text=mark("floor=d"))

        #### bottom cover options
        if False:
            # Plain
            section = Section(a, FINGER_COUNTER, text=mark("front=a"))
        else:
            # Same as covers on other levels
            _zig, zag = zigzags[0]
            section = Compound(
                [
                    Section(a - zag, PLAIN, text=mark("front counterzag")),
                    Section(zag, FINGER, text=mark("front zag")),
                ]
            )
        assert_that(section.length, close_to(a, delta=1e-3))

        w.add(section, 90)

        # Used to draw edges of finger hole area
        zigzag_corners = []

        # orthogonal, zig-zaggy side
        for zig, zag in zigzags:
            w.add(zig, -90, PLAIN, text=mark("zig"))
            zigzag_corners.append(w.position)
            w.add(zag, 90, FINGER, text=mark("zag"))

        # now go all the way from current x to x=0
        w.add(hat_length, 180 - alpha_deg, PLAIN, text=mark("topside"))

        # now all the way down.
        w.add(
            reversed(
                Compound.intersperse(
                    gap,
                    make_sections(sheff, "sheff", FINGER),
                    start=False,
                    end=False,
                )
            ),
            90,
        )

        def internals():
            for zigzag_corner, (_zig, zag) in zip(zigzag_corners, zigzags):
                with self.saved_context():
                    if self.debug:
                        color = (0, 128, 128)
                        self.ctx.set_source_rgb(*color)
                    # upper stroke
                    with self.saved_context():
                        self.moveTo(zigzag_corner, 180 + alpha_deg)
                        self.edge(a - zag)
                        self.corner(-alpha_deg)
                        self.edge(d)
                        self.ctx.stroke()
                    # lower stroke
                    with self.saved_context():
                        self.moveTo(zigzag_corner, 270 + alpha_deg)
                        self.edge(f)
                        self.corner(-90)
                        delta = f * tan(angle_rad / 2)
                        print(f"{delta=}")
                        self.edge((a - zag) + delta)
                        self.corner(-alpha_deg)
                        self.edge(d + delta)
                        self.ctx.stroke()

            with self.saved_context():
                self.moveTo(0, -f / 2)
                for h, (_zig, zag) in zip(sh, zigzags):
                    self.moveTo(0, h + f)
                    self.fingerHolesAt(0, 0, d, 0)

        element = Element.from_item(w)
        element.add_render(internals)
        return element

    @inject_shortcuts
    def side_working_a(
        self,
        a,
        alpha_deg,
        angle_rad,
        cos_a,
        d,
        f,
        gap,
        hat_length,
        sh,
        sheff,
        sin_a,
    ) -> Element:
        w = self.wall_builder("side")
        w.add(d, alpha_deg, FINGER_COUNTER, text=mark("floor=d"))

        zigzags: list[tuple[float, float]] = [
            tuple(coord(cos_a, sin_a) * (h + f)) for h in sh
        ]

        #### bottom cover options
        if False:
            # Plain
            section = Section(a, FINGER_COUNTER, text=mark("front=a"))
        else:
            # Same as covers on other levels
            _zig, zag = zigzags[1]
            section = Compound(
                [
                    Section(a - zag, PLAIN, text=mark("front counterzag")),
                    # VSection(a - zag, FINGER_COUNTER, text=mark("front counterzag")),
                    Section(zag, FINGER_COUNTER, text=mark("front zag")),
                ]
            )
        assert_that(section.length, close_to(a, delta=1e-3))

        w.add(section, 90)

        # Used to draw edges of finger hole area
        zigzag_corners = []

        # orthogonal, zig-zaggy side
        for zig, zag in zigzags:
            w.add(zig, -90, PLAIN, text=mark("zig"))
            zigzag_corners.append(w.position)
            w.add(zag, 90, FINGER_COUNTER, text=mark("zag"))

        # now go all the way from current x to x=0
        w.add(hat_length, 180 - alpha_deg, PLAIN, text=mark("topside"))

        # now all the way down.
        w.add(
            reversed(
                Compound.intersperse(
                    gap,
                    make_sections(sheff, "sheff", FINGER),
                    start=True,
                    end=False,
                )
            ),
            90,
        )

        def internals():
            for zigzag_corner, (_zig, zag) in zip(zigzag_corners, zigzags):
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

            with self.saved_context():
                self.moveTo(0, f / 2)
                for h, (_zig, zag) in zip(sh, zigzags):
                    self.moveTo(0, h + f)
                    # self.moveTo(0, h)
                    # self.fingerHolesAt(0, 0, a - zag, 90)
                    with self.saved_context():
                        delta = f / 2 * tan(angle_rad / 2)
                        self.fingerHolesAt(0, 0, d - delta, 0)

                        # Eh, finger holes aren't worth it...
                        #
                        # self.moveTo(d - delta, 0, alpha_deg)
                        # self.fingerHolesAt(0, 0, a - zag, 0)

        element = Element.from_item(w)
        element.add_render(internals)
        return element

    @inject_shortcuts
    def all_fronts(self, sheff) -> Element:
        return self.build_element_grid(
            nx=1,
            ny=len(sheff),
            element_factory=self.front,
        )

    @inject_shortcuts
    def front(self, xi, yi, gap, sx, a, zigzags, f) -> Element:
        assert xi == 0
        # NOTE: this is the *first one* - it uses size of first drawer's zig/zag
        # horizontal: same sections as on X on back
        mouth_edges = Compound.intersperse(
            gap, make_sections(sx, "mouth", PLAIN), start=True, end=True
        )
        xedges = Compound.intersperse(
            gap, make_sections(sx, "sx angled neg", ANGLED_NEG), start=True, end=True
        )
        # xxx: reuse zig on level 0
        if yi == 0:
            _zig, zag = zigzags[0]
        else:
            _zig, zag = zigzags[yi - 1]
        side = Compound(
            [
                Section(zag, FINGER_COUNTER, text=mark("front zag")),
                Section(a - zag, PLAIN, text=mark("front counterzag")),
            ]
        )
        w = (
            self.wall_builder(f"front{yi}")
            .add(mouth_edges, 90)
            .add(side, 90)
            .add(xedges, 90)
            .add(reversed(side), 90)
        )

        def internals():
            # Vertical finger holes for bottom floor
            with self.saved_context():
                self.moveTo(f / 2, 0)
                for x in sx[:-1]:
                    self.moveTo(x + f, 0)
                    self.fingerHolesAt(0, 0, zag, 90)

        element = Element.from_item(w)
        element.add_render(internals)
        return element

    @inject_shortcuts
    def bottom_floor(self, sx, gap, d, f) -> Element:
        w = self.wall_builder("bottom_floor")
        xedges = Compound.intersperse(
            gap, make_sections(sx, "sx to front", ANGLED_POS), start=True, end=True
        )
        w.add(xedges, 90)
        w.add(d, 90, FINGER_COUNTER)
        xedges2 = Compound.intersperse(
            gap, make_sections(sx, "sx", FINGER), start=True, end=True
        )
        w.add(xedges2, 90)
        w.add(d, 90, FINGER_COUNTER)

        def internals():
            # Vertical finger holes for bottom floor
            with self.saved_context():
                self.moveTo(f / 2, 0)
                for x in sx[:-1]:
                    self.moveTo(x + f, 0)
                    self.fingerHolesAt(0, 0, d, 90)

        element = Element.from_item(w)
        element.add_render(internals)
        return element

    @inject_shortcuts
    def midfloor(self, x_idx, y_idx, sx, d, alpha_deg) -> Element:
        x = sx[x_idx]
        w = self.wall_builder(f"midfloor{x_idx}/{y_idx}")
        w.add(x, 90, ANGLED_POS, text=mark(f"front {alpha_deg}{DEG_SIGN}"))
        w.add(d, 90, (SKIP_EVEN if x_idx < len(sx) - 1 else FINGER))
        w.add(x, 90, FINGER)
        w.add(d, 90, (SKIP_ODD_REVERSE if x_idx > 0 else FINGER))
        return Element.from_item(w)

    @inject_shortcuts
    def midfloors(self, sx, sh, f):
        return self.build_element_grid(
            nx=len(sx),
            ny=len(sh),
            element_factory=self.midfloor,
            xspacing=-f,
        )

    def circle(self, x=0, y=0, r=1):
        """Sets defaults."""
        super().circle(x=x, y=y, r=r)

    @inject_shortcuts
    def back(self, f, sx, sheff, gap) -> Element:
        xedges = Compound.intersperse(
            gap, make_sections(sx, "sx", FINGER_COUNTER), start=True, end=True
        )
        yedges = Compound.intersperse(
            gap,
            make_sections(sheff, "sheff", FINGER_COUNTER),
            start=True,
            end=False,
        )
        w = (
            self.wall_builder("back")
            .add(xedges, 90)
            .add(yedges, 90)
            .add(xedges.length, 90, PLAIN)
            .add(reversed(yedges), 90)
        )

        def internals():
            with self.saved_context():
                self.moveTo(f, f / 2)
                for iy, dy in enumerate(sheff):
                    with self.saved_context():
                        for ix, dx in enumerate(sx):
                            # if ix < len(sx) - 1:
                            if iy > 0:
                                self.fingerHolesAt(0, 0, dx, 0)

                            if ix > 0:
                                self.fingerHolesAt(-f / 2, f / 2, dy, 90)
                            self.moveTo(dx + f, 0)
                    self.moveTo(0, dy + f)

        element = Element.from_item(w)
        element.add_render(internals)
        return element
