"""
scripts/boxes NamePlate --reference 0
scripts/boxes NamePlate --format=lbrn2 --output=/home/agentydragon/nameplate.lbrn2

"""

from __future__ import annotations
import itertools

from boxes.edges import (
    FingerJointEdge,
    FingerJointEdgeCounterPart,
    FingerJointSettings,
)
from boxes.generators.raibase import (
    PLAIN,
    Element,
    inject_shortcuts,
    RaiBase,
)

DOWN_ARROW = "↓"

def mark(s):
    return DOWN_ARROW + s + DOWN_ARROW


class NamePlate(RaiBase):
    def __init__(self):
        super().__init__()

        self.argparser.add_argument(
            "--slot_width",
            action="store",
            type=float,
            default=20,
            help="Width of plate slots, set to 0 to disable.",
        )
        self.argparser.add_argument(
            "--slot_distance",
            action="store",
            type=float,
            default=60,
            help="Distance of nameplate slots.",
        )
        DEFAULT_WIDTH = 150
        self.argparser.add_argument(
            "--front_width", action="store", type=float, default=DEFAULT_WIDTH,
        )
        from scipy.constants import golden
        self.argparser.add_argument(
            "--front_height", action="store", type=float, default=DEFAULT_WIDTH / golden,
        )
        self.argparser.add_argument(
            "--back_height", action="store", type=float, default=10
        )
        # width is automatically between plates

    @property
    def shortcuts(self):
        return dict(
        )

    def setup(self):
        s = FingerJointSettings(
            relative=True,
            thickness=self.thickness,
            **self.edgesettings.get("FingerJoint", {}),
            surroundingspaces=0,
        )
        # directly joint
        self.edges['a'] = FingerJointEdge(self, s)
        self.edges['A'] = FingerJointEdgeCounterPart(self, s)

        pass

    def plate(self, width, height):
        w = self.wall_builder("front")
        # bottom
        x = (width - self.slot_distance - 2 * self.slot_width) / 2
        print(f"{x = }")

        w.add(width , 90, PLAIN, text=mark("front bottom")),

        w.add(height, 90, PLAIN, text=mark("front side A")),
        w.add(x, 0, PLAIN),
        w.add(self.slot_width, 0, 'A', text=mark("slot")),

        w.add(self.slot_distance, 0, PLAIN)

        w.add(self.slot_width, 0, 'A', text=mark("slot")),
        w.add(x, 90, PLAIN),
        w.add(height, 90, PLAIN, text=mark("front side B")),
        
        w.add(width, 90, PLAIN, text=mark("front top")),

        return Element.from_item(w).close_part()

    def front(self):
        return self.plate(self.front_width, self.front_height)

    def back(self):
        back_width = self.slot_distance + 2 * self.slot_width + 2 * self.thickness
        return self.plate(back_width, self.back_height)

    def connector(self):
        w = self.wall_builder("connector")

        t = self.thickness + 0.3 # add a bit of tolerance

        w.add(self.slot_width, 90, 'a')
        w.add(t, 90, PLAIN)
        w.add(self.slot_width, 90, 'a')
        w.add(t, 90, PLAIN)

        return Element.from_item(w).close_part()


    #def bridge(self):
    #    w = self.wall_builder("bridge")

    #    t = self.thickness + 0.3 # add a bit of tolerance

    #    w.add(self.slot_width, 0, 'a')
    #    w.add(self.slot_distance, 0, PLAIN)
    #    w.add(self.slot_width, 90, 'a')
    #    w.add(t, 90, PLAIN)

    #    w.add(self.slot_width, 0, 'a')
    #    w.add(self.slot_distance, 0, PLAIN)
    #    w.add(self.slot_width, 90, 'a')
    #    w.add(t, 90, PLAIN)

    #    return Element.from_item(w).close_part()


    @inject_shortcuts
    def render(self):
        self.setup()
        self.ystack(self.front(), self.back(), self.connector(), self.connector()).do_render()
