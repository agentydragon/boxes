"""
scripts/boxes MailRack --debug=True --reference 0

scripts/boxes MailRack --Mounting_style='mounting tab'

"Tab": round thingies sticking up with slots for screws

scripts/boxes MailRack --Mounting_style='straight edge, extended'

scripts/boxes MailRack --Mounting_num=3

scripts/boxes MailRack --preset=15leroy

mounting screw:
    3.45mm shaft diameter
    5.48mm head diameter

scripts/boxes MailRack --preset=test-noplate

scripts/boxes MailRack --preset=test-noplate --format=lbrn2 --output=/home/agentydragon/test-noplate.lbrn2

scripts/boxes MailRack --preset=15leroy
scripts/boxes MailRack --preset=15leroy --format=lbrn2 --output=/home/agentydragon/15leroy.lbrn2

"""

from __future__ import annotations

import logging
from math import cos, radians, sin, sqrt, tan

from hamcrest import assert_that, close_to

from boxes import argparseSections, boolarg, holeCol, restore
from boxes.edges import (
    FingerJointEdge,
    FingerJointEdgeCounterPart,
    FingerJointSettings,
    MountingSettings,
)
from boxes.generators.raibase import (
    ALPHA_SIGN,
    PLAIN,
    DEG_SIGN,
    Compound,
    Element,
    RaiBase,
    Section,
    SkippingFingerJoint,
    coord,
    fmt,
    fmt_mm,
    inject_shortcuts,
)

FINGER = "f"
FINGER_COUNTER = "F"
ANGLED_POS = "b"
ANGLED_NEG = "B"
RIGHT_ARROW = "→"
DOWN_ARROW = "↓"
SKIP_EVEN = "a"
SKIP_ODD = "A"
SKIP_ODD_REVERSE = "@"

MIDDLE_POCKETS = "pockets"
MIDDLE_POCKETS_IN_FINGERS_OUT = "pockets_in_fingers_out"
MIDDLE_FINGERHOLES = "fingerholes"


def mark(s):
    return DOWN_ARROW + s + DOWN_ARROW

def make_sections(xs, name, edge):
    return [Section(x, edge, text=f"{name}{i}") for i, x in enumerate(xs)]


# class MailRack(RaiBase):
class MailRack(RaiBase):
    def __init__(self):
        super().__init__()

        # debug values:
        #
        # self.buildArgParser(sh="40*2", sx="40*2")
        # self.add_float_arg("alpha_deg", 60)
        # self.add_float_arg("side_angled_length", 60)
        # self.add_float_arg("floor_depth", 30)
        # self.add_str_arg("middle_style",MIDDLE_POCKETS)

        self.buildArgParser(
            sh="180*2", sx="240*3"
        )  # TODO: some cominations aren't compatible; document
        self.add_float_arg("alpha_deg", 60)
        self.add_float_arg("side_angled_length", 170)
        self.add_float_arg("floor_depth", 50)
        self.add_float_arg("cutout_height", 10)
        self.add_float_arg("d_from_bottom", 30)
        self.argparser.add_argument(
            "--n_cutouts",
            action="store",
            type=int,
            default=2,
        )
        self.argparser.add_argument(
            "--middle_style",
            action="store",
            type=str,
            help="How to handle middle floors. Pockets: make cutouts for middle floors in sides & left/right separators. Fingerjoints: split each middle floor into one piece for each horizontal split. Attach them into finger holes. pockets_in_fingers_out: pockets in the middle, finger holes in the sides.",
            choices=[MIDDLE_POCKETS, MIDDLE_FINGERHOLES, MIDDLE_POCKETS_IN_FINGERS_OUT],
            default=MIDDLE_POCKETS_IN_FINGERS_OUT,
        )
        self.argparser.add_argument(
            "--top_style",
            action="store",
            type=str,
            help="What to do about the top (open) shelf. Continuous: extend diagonal line of opening until touching back. Symmetric (only for equally tall shelves): make opening 'zig' as large as in all other shelves, then cut off horizontally.",
            choices=["continuous", "symmetric"],
            default="symmetric",
        )
        self.argparser.add_argument(
            "--plate_slot_width",
            action="store",
            type=float,
            default=20,  # <- for nameplate
            help="Width of plate slots, set to 0 to disable.",
        )
        self.argparser.add_argument(
            "--plate_slot_depth",
            action="store",
            type=float,
            default=3,
            help="todo",
        )
        self.argparser.add_argument(
            "--plate_slot_distance",
            action="store",
            type=float,
            default=60,
            help="Distance of nameplate slots.",
        )
        self.addSettingsArgs(MountingSettings)
        self.argparser.add_argument(
            "--preset",
            action="store",
            type=str,
            default="",
        )

    @property
    def shortcuts(self):
        alpha_rad = radians(self.alpha_deg)
        f = self.finger_hole_width
        sh = self.sh
        d = self.floor_depth
        a = self.side_angled_length
        sin_a, cos_a = sin(alpha_rad), cos(alpha_rad)
        top_style = self.top_style

        zigzags: list[tuple[float, float]] = [
            tuple(coord(cos_a, sin_a) * (h + f)) for h in sh
        ]

        if top_style == "continuous":
            self.hat_length = (d + a * cos_a) / sin_a
            # right-angle triangles:
            # 1. front length, hat length, [diagonal]
            # 2. floor depth, (hat height + spacer), [diagonal]
            hat_height = sqrt(a**2 + self.hat_length**2 - d**2)
        if top_style == "symmetric":
            zig = zigzags[0][0]
            self.hat_horizontal = d + a * cos_a - zig * sin_a
            hat_height = zig * cos_a + a * sin_a

        sheff = sh + [hat_height]
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
            hat_height=hat_height,
            sheff=sheff,
            gap=Section(f, PLAIN, text="f"),
            zigzags=zigzags,
            middle_style=self.middle_style,
            top_style=self.top_style,
        )

    def setup(self):
        assert self.top_style in ("continuous", "symmetric")
        if self.top_style == "symmetric":
            assert len(set(self.sh)) == 1

        # logging.getLogger("SkippingFingerJoint").setLevel(logging.DEBUG)
        # logging.basicConfig(level=logging.DEBUG)

        s = self.make_angled_finger_joint_settings(self.alpha_deg)
        self.edges[ANGLED_POS] = FingerJointEdge(self, s)
        self.edges[ANGLED_NEG] = FingerJointEdgeCounterPart(self, s)
        # self.angled_finger_holes = FingerHoles(s)

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

    def open(self):
        self.apply_preset()
        super().open()

    def apply_preset(self):
        if not self.preset:
            return
        me = self.edgesettings["Mounting"]

        self.reference = 0
        self.thickness = 3.175 # 1/8 inch
        self.top_style = "symmetric"
        self.middle_style = MIDDLE_POCKETS_IN_FINGERS_OUT
        if self.preset == "15leroy":
            self.sh = [160] * 2
            self.sx = [240] * 3
            self.alpha_deg = 60
            self.side_angled_length = 170
            self.floor_depth = 50

            self.plate_slot_width = 20
            self.plate_slot_distance = 60
            self.plate_slot_depth = self.thickness
            self.d_from_bottom = 20 
            self.cutout_height = 15 
            self.cutout_width = 70

            me["d_shaft"] = 3.6 # 3.45 plus margin
            me["d_head"] = 6.5 # 5.48 plus margin
            me["num"] = 2 #len(self.sx)
        elif self.preset == "test-noplate":
            self.d_from_bottom = 50
            self.cutout_height = 5
            self.sh = [40] * 2
            self.sx = [30] * 3
            self.alpha_deg = 60
            self.side_angled_length = 50
            self.floor_depth = 30

            self.plate_slot_width = 0
            self.plate_slot_distance = 0
            self.plate_slot_depth = 0

        else:
            raise ValueError()


    @inject_shortcuts
    def render(self, sheff):
        self.setup()
        if self.cutout_width == "equal":
            # TODO: check stuff etc.
            space_plus_drawer = (self.sx[0] + self.f) / self.n_cutouts
            self.cutout_width = space_plus_drawer / 2

        print(f"Cutout: {self.cutout_width} x {fmt_mm(self.cutout_height)}")

        # self.ctx.set_line_width(1)

        ################## full render

        # To render all, need:
        #   - `back`
        #   - `midfloors`
        #   - `bottom``
        #   - `all_fronts`
        #   - len(sx)+1 copies of `side`

        elems = []

        back = self.back()
        print(f"width {fmt_mm(back.width)}, height {fmt_mm(back.height)}")
        back = back.translate(coord(0, -back.height))
        elems.append(back)

        stack = self.ystack(
            self.midfloors(),
            *(self.front(yi) for yi in range(len(sheff))),
            self.bottom_floor(),
        )
        stack = stack.translate(coord(0, -stack.height - self.spacing - back.height))
        elems.append(stack)

        # divide sides to left/right
        all_sides = [self.side(is_inner=False) for _ in range(2)] + [self.side(is_inner=True) for _ in range(len(self.sx) - 1)]
        print(f"depth: {fmt_mm(all_sides[0].width)}")
        split = len(all_sides) // 2
        sides = Element.union(self, [
            self.ystack(all_sides[split:]).mirror().translate(coord(-3 * self.spacing, 0)),
            self.ystack(all_sides[:split]).translate(coord(back.width + 3 * self.spacing, 0)),
        ])
        elems.append(sides.translate(coord(0, -sides.height)))

        full = Element.union(self, elems)
        print(f"Cut material size: {fmt_mm(full.width)} x {fmt_mm(full.height)}")
        full.do_render()

    @inject_shortcuts
    def side(
        self,
        a,
        alpha_deg,
        angle_rad,
        d,
        f,
        gap,
        sh,
        sheff,
        zigzags,
        middle_style,
        top_style,
        cos_a,
        sin_a,
        is_inner: bool,
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

        if top_style == "continuous":
            # now go all the way from current x to x=0
            w.add(self.hat_length, 180 - alpha_deg, PLAIN, text=mark("topside"))
        if top_style == "symmetric":
            # all zigs & zags are the same because this only works with equally-sized
            # shelves
            w.add(zig, 90 - alpha_deg, PLAIN, text=mark("topside"))
            w.add(
                d + a * cos_a - zig * sin_a, 90, PLAIN, text=mark("topside horizontal")
            )

        # now all the way down.
        s = list(reversed(sheff))
        have_slot = (
            middle_style == MIDDLE_POCKETS or
            (middle_style == MIDDLE_POCKETS_IN_FINGERS_OUT and is_inner)
        )
        if have_slot:
            w.add(s[0], 90, FINGER)
            for h in s[1:]:
                w.slot(depth=d, length=f)
                w.add(h, 90, FINGER)
        else:
            w.add(
                Compound.intersperse(
                    gap,
                    make_sections(s, "sheff", FINGER),
                    start=False,
                    end=False,
                ),
                90,
            )

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

            if (
                middle_style == MIDDLE_FINGERHOLES or
                (middle_style == MIDDLE_POCKETS_IN_FINGERS_OUT and not is_inner)
            ):
                with self.saved_context():
                    self.moveTo(0, -f / 2)
                    for h, (_zig, zag) in zip(sh, zigzags):
                        self.moveTo(0, h + f)
                        self.fingerHolesAt(0, 0, d, 0)

        return Element.from_item(w).add_render(internals).close_part()

    @inject_shortcuts
    def front(self, yi, gap, sx, a, zigzags, f) -> Element:
        # expected width of front section
        expected_width = (f*(len(sx)+1) + sum(sx))

        # xxx: reuse zig on level 0
        if yi == 0:
            _zig, zag = zigzags[0]
        else:
            _zig, zag = zigzags[yi - 1]

        w = self.wall_builder(f"front{yi}")


        ###### front edge ######
        pos_start = w.position
        print(f"{pos_start=}")

        if self.plate_slot_width:
            swidth, sdist, sdepth = self.plate_slot_width, self.plate_slot_distance, self.plate_slot_depth
            for x in sx:
                w.add(gap, 0)
                leftover = x - 2 * swidth - sdist
                w.add(leftover / 2, 90, PLAIN)
                w.slot(sdepth, swidth)
                w.add(sdist, 90, PLAIN)
                w.slot(sdepth, swidth)
                w.add(leftover / 2, 0, PLAIN)
            w.add(gap, 90)

        else:
            mouth_edges = Compound.intersperse(
                gap, make_sections(sx, "mouth", PLAIN), start=True, end=True
            )
            w.add(mouth_edges, 90)

        pos_end = w.position
        ###### end front edge ######
        print(f"=> {pos_end=}")
        assert_that(float((pos_end - pos_start)[0]), close_to(
            f*(len(sx)+1) + sum(sx), 0.01
        ))

        # TODO: dedupe w/ rev side
        if yi == 0:
            w.add(zag, 0, FINGER_COUNTER, text=mark("front zag"))
        else:
            w.add(zag, 90, FINGER_COUNTER, text=mark("front zag"))
            w.add(f, -90, PLAIN)

        w.add(a - zag, 90, PLAIN, text=mark("front counterzag"))

        if yi == 0:
            w.add(gap, 0)

        for x in sx[:-1]:
            # Slot only on drawers above level 1
            if yi != 0:
                w.add(x, 90, ANGLED_NEG)
                w.slot(depth=(a-zag), length=f)
            else:
                w.add(x, 0, ANGLED_NEG)
                w.add(gap, 0)

        if yi == 0:
            w.add(sx[-1], 0, ANGLED_NEG)
            w.add(gap, 90)
        else:
            w.add(sx[-1], 90, ANGLED_NEG)


        if yi == 0:
            w.add(a - zag, 0, PLAIN, text=mark("front counterzag"))
            w.add(zag, 0, FINGER_COUNTER, text=mark("front zag"))
        else:
            w.add(a - zag, -90, PLAIN, text=mark("front counterzag"))
            w.add(f, 90, PLAIN)
            w.add(zag, 90, FINGER_COUNTER, text=mark("front zag"))

        def internals():
            # Vertical finger holes for bottom floor
            with self.saved_context():
                self.moveTo(f / 2, 0)
                for x in sx[:-1]:
                    self.moveTo(x + f, 0)
                    self.fingerHolesAt(0, 0, zag, 90)

            # golden ratio is 1.61803398875

            with self.saved_context():
                # self.moveTo(f / 2, 0)
                assert len(set(sx)) == 1, "cutouts only ok if all equal"

                space_plus_drawer = (sx[0] + f) / self.n_cutouts

                space = space_plus_drawer - self.cutout_width
                self.moveTo( space / 2 + f / 2, 0)
                # if rendered now, the cutout's top edge would be aligned with drawer's top edge.

                self.moveTo(0, -self.cutout_height + zag - self.d_from_bottom)
                

                for _ in range(len(sx) * self.n_cutouts):
                    self.front_cutout(self.cutout_width, self.cutout_height)
                    self.moveTo(self.cutout_width, 0)
                    self.moveTo(space, 0)

        return Element.from_item(w).add_render(internals).close_part()

    @holeCol
    def front_cutout(self, cutout_width, cutout_height):
        self.moveTo(cutout_height / 2, 0)

        self.edge(cutout_width - cutout_height)
        self.corner(180, cutout_height / 2)
        self.edge(cutout_width - cutout_height)
        self.corner(180, cutout_height / 2)

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
    def midfloor_fullwidth(self, sx, d, gap) -> Element:
        # TODO: very similar to bottom floor
        w = self.wall_builder(f"midfloor({self.middle_style})")

        def _half(edge_type, name):
            middle = Compound.intersperse(
                gap, make_sections(sx, name, edge_type), start=False, end=False
            )

            if self.middle_style == MIDDLE_POCKETS:
                w.add(gap, 0)
                w.add(middle, 0)
                w.add(gap, 90)
                w.add(d, 90, PLAIN)
            elif self.middle_style == MIDDLE_POCKETS_IN_FINGERS_OUT:
                w.add(middle, 90)
                w.add(d, 90, FINGER) # PLAIN)

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
    def midfloors_level(self, yi):
        if self.middle_style == MIDDLE_FINGERHOLES:
            return self.xstack(
                self.midfloor_fingerholes(i) for i in range(len(sx))
            )
        elif self.middle_style in (MIDDLE_POCKETS, MIDDLE_POCKETS_IN_FINGERS_OUT):
            return self.midfloor_fullwidth()
        else:
            raise ValueError(f"{self.middle_style=}")

    @inject_shortcuts
    def midfloors(self, sh) -> Element:
        return self.ystack(self.midfloors_level(yi) for yi in range(len(sh)))

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
            .add(xedges.length, 90, "G")  # <- mounting edge
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
