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

        # debug values:
        #
        # self.buildArgParser(sh="40*2", sx="40*2")
        # self.add_float_arg("alpha_deg", 60)
        # self.add_float_arg("side_angled_length", 60)
        # self.add_float_arg("floor_depth", 30)
        # self.add_str_arg("middle_style", "pockets")

        self.buildArgParser(sh="120*2", sx="240*3")
        self.add_float_arg("alpha_deg", 60)
        self.add_float_arg("side_angled_length", 150)
        self.add_float_arg("floor_depth", 50)
        self.add_str_arg("middle_style", "pockets")

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
            middle_style=self.middle_style,
        )

    def setup(self):
        assert self.middle_style in ("pockets", "fingerholes")

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

        self.ctx.set_line_width(1)

        # To render all, need:
        #   - `back`
        #   - `midfloors`
        #   - `bottom``
        #   - `all_fronts`
        #   - len(sx)+1 copies of `side`

        y = 0

        def _stack_y(element):
            nonlocal y
            y -= element.bbox.height + self.spacing
            element.translate(coord(0, y)).do_render()

        back = self.back()
        _stack_y(back)
        _stack_y(
            self.ystack(
                self.midfloors().translate(coord(-self.thickness, 0)),
                self.all_fronts(),
                self.bottom_floor(),
            )
        )

        # divide sides to left/right
        all_sides = [self.side() for _ in range(len(self.sx) + 1)]
        split = len(all_sides) // 2
        on_left, on_right = all_sides[:split], all_sides[split:]

        right = self.ystack(on_right).translate(
            coord(back.bbox.width + 2 * self.spacing, 0)
        )
        left = self.ystack(on_left).mirror().translate(coord(-2 * self.spacing, 0))
        e = Element.union(self, [left, right]).translate(coord(0, self.thickness))
        e.translate(coord(0, -e.bbox.height)).do_render()

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
        middle_style,
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
        if middle_style == "fingerholes":
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
        elif middle_style == "pockets":
            s = list(reversed(sheff))
            w.add(s[0], 90, FINGER)
            for h in s[1:]:
                w.add(d, -90, PLAIN)
                w.add(f, -90, PLAIN)
                w.add(d, 90, PLAIN)
                w.add(h, 90, FINGER)

        def internals():
            if self.debug:
                for zigzag_corner, (_zig, zag) in zip(zigzag_corners, zigzags):
                    with self.saved_context():
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

            if middle_style == "fingerholes":
                with self.saved_context():
                    self.moveTo(0, -f / 2)
                    for h, (_zig, zag) in zip(sh, zigzags):
                        self.moveTo(0, h + f)
                        self.fingerHolesAt(0, 0, d, 0)

        return Element.from_item(w).add_render(internals).close_part()

    @inject_shortcuts
    def all_fronts(self, sheff) -> Element:
        return self.build_element_grid(ny=len(sheff), factory=self.front)

    @inject_shortcuts
    def front(self, xi, yi, gap, sx, a, zigzags, f) -> Element:
        assert xi == 0
        # NOTE: this is the *first one* - it uses size of first drawer's zig/zag
        # horizontal: same sections as on X on back
        mouth_edges = Compound.intersperse(
            gap, make_sections(sx, "mouth", PLAIN), start=True, end=True
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
        w = self.wall_builder(f"front{yi}").add(mouth_edges, 90).add(side, 90)

        # add with pockets
        w.add(gap, 0)
        for x in sx[:-1]:
            w.add(sx[0], 90, ANGLED_NEG)
            w.add(a - zag, -90, PLAIN)
            w.add(f, -90, PLAIN)
            w.add(a - zag, 90, PLAIN)
        w.add(sx[-1], 0, ANGLED_NEG)
        w.add(gap, 90)

        w.add(reversed(side), 90)

        def internals():
            # Vertical finger holes for bottom floor
            self.moveTo(f / 2, 0)
            for x in sx[:-1]:
                self.moveTo(x + f, 0)
                self.fingerHolesAt(0, 0, zag, 90)

        return Element.from_item(w).add_render(internals).close_part()

    @inject_shortcuts
    def bottom_floor(self, sx, gap, d, f) -> Element:
        w = self.wall_builder("bottom floor")
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
            self.moveTo(f / 2, 0)
            for x in sx[:-1]:
                self.moveTo(x + f, 0)
                self.fingerHolesAt(0, 0, d, 90)

        return Element.from_item(w).add_render(internals).close_part()

    @inject_shortcuts
    def midfloor_pockets(self, sx, d, gap) -> Element:
        # TODO: very similar to bottom floor
        w = self.wall_builder("midfloor(pockets)")

        def _half(edge_type, name):
            xedges = Compound.intersperse(
                gap, make_sections(sx, name, edge_type), start=True, end=True
            )
            w.add(gap, 0)
            w.add(xedges, 0)
            w.add(gap, 90)  # <- added gap to extend to both sides
            w.add(d, 90, PLAIN)

        _half(ANGLED_POS, "sx to front")
        _half(FINGER, "sx to back")
        return Element.from_item(w).close_part()

    @inject_shortcuts
    def midfloor_fingerholes(self, x_idx, sx, d, alpha_deg) -> Element:
        x = sx[x_idx]
        w = self.wall_builder(f"midfloor(fingerholes) {x_idx}")
        w.add(x, 90, ANGLED_POS, text=mark(f"front {alpha_deg}{DEG_SIGN}"))
        w.add(d, 90, (SKIP_EVEN if x_idx < len(sx) - 1 else FINGER))
        w.add(x, 90, FINGER)
        w.add(d, 90, (SKIP_ODD_REVERSE if x_idx > 0 else FINGER))
        return Element.from_item(w).close_part()

    @inject_shortcuts
    def midfloors(self, sx, sh, f, middle_style) -> Element:
        if middle_style == "fingerholes":
            make_row = lambda: self.xstack(
                self.midfloor_fingerholes(i) for i in range(len(sx))
            )
        if middle_style == "pockets":
            make_row = self.midfloor_pockets

        return self.ystack(make_row() for _ in range(len(sh)))

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
            self.moveTo(f, f / 2)
            for iy, dy in enumerate(sheff):
                with self.saved_context():
                    for ix, dx in enumerate(sx):
                        if iy > 0:
                            self.fingerHolesAt(0, 0, dx, 0)

                        if ix > 0:
                            self.fingerHolesAt(-f / 2, f / 2, dy, 90)
                        self.moveTo(dx + f, 0)
                self.moveTo(0, dy + f)

        return Element.from_item(w).add_render(internals).close_part()
