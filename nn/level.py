import math
import numpy as np
from time import sleep
from sets import Set

actorChars = {
  "@": "Player",
  "o": "Coin",
  "!": "Lava",
  "|": "Rope",
  "+": "RopeWall",
  "G": "Goal",
}

world_chars = {
  "x": "Wall",
  "|": "Rope",
  "+": "RopeWall",
  " ": None,
  ".": None,

}

def surrounding_coords(x, y):
    # top, right, bottom, left
    return (x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)

def bounding_coords(x, y):
    #        N,         NE,             E,         SE,              S,          SW,            W,           NW
    return (x, y - 1), (x + 1, y - 1), (x + 1, y), (x + 1, y + 1), (x, y + 1), (x - 1, y + 1), (x - 1, y), (x - 1, y - 1)

class Actor:
    def __init__(self, x, y, ch, l):
        self.pos = (x, y)
        self.surrounding = surrounding_coords(x, y)
        self.ch = ch
        self.type = actorChars[ch]
        self.level = l
        self.facing = "left"
        self.can_collide = True
        self.distance_from_goal = 0
        self.seen = Set()

        # What do we see?
        self.get_sight()

    def get_surrounding_ch(self):
        return [self.level.get_ch(*x) for x in surrounding_coords(*self.pos)]

    def get_bounding_ch(self):
        return [self.level.get_ch(*x) for x in bounding_coords(*self.pos)]

    def raycast(self, angle, record=False):
        ch = ' '
        distance = 0.0
        delta = 0.5 * (1.0 / math.cos(math.pi / 4.0))
        x = self.pos[0]
        y = self.pos[1]
        upper_bound = (distance + self.pos[0] < self.level.width) and (distance + self.pos[1] < self.level.height)
        lower_bound = (distance + self.pos[0] > 0) and (distance + self.pos[1] > 0)
        while ch == ' ' and upper_bound and lower_bound:
            distance += delta
            x = distance * math.cos(angle) + self.pos[0] + 0.5
            y = -(distance * math.sin(angle)) + self.pos[1] + 0.5
            ch = self.level.get_ch(int(x), int(y))
            if record:
                self.seen.add((ord(ch), int(x), int(y)))
        return (ch, distance, x, y);

    def get_sight(self, coords=False, norm=False):
        eyes = []
        eye_count = 200
        for eye in xrange(0, eye_count):
            ch, distance, x, y = self.raycast(eye * (math.pi / (eye_count / 2)), coords)
            if not coords:
                eyes += [ord(ch), distance]
            elif norm:
                # Our max ord range, stopping at 'o'
                max_ord_size = 255.0
                distance = distance / math.sqrt(len(self.level.original[0]) ** 2 + len(self.level.original) ** 2)
                eyes += [ord(ch) / max_ord_size, distance]
            else:
                eyes.append((ord(ch), int(x), int(y)))
        return np.array(eyes) if not coords ^ norm else eyes

    def get_surrounding_sight(self):

        return [
            (self.level.get_ch(*x), x[0], x[1])
                for x in surrounding_coords(*self.pos)
        ]

    def get_resulting_ch(self, action):
        """
          Grabs the character the agent would move to given an action
        """
        return self.get_surrounding_ch()[action]

    def distance_from(self, x, y):
        gx, gy = self.pos

        # Calculate and return the distance between the two points
        return math.sqrt( (gx - x) ** 2 + (gy - y) ** 2)

    def has_seen(self, eye):
        # print self.seen
        # print eye
        # HACK: not?
        return not len([x for x in self.seen if x == eye]) == 1

    def has_explored(self):
        return len(self.seen & Set(self.get_sight(coords=True)))

    # Move up: 0, Right: 1, Down: 2, Left: 3
    def _move(self, t):
        direction = self.surrounding[t]
        # See if anything is occupying the next slot
        occupant = self.level.get_ch(*direction)
        if not world_chars.get(occupant):
            # Set our current position to nothing
            self.level.set_ch(*self.pos, value=" ")
            self.level.set_ch(*direction, value=self.ch)
            self.pos = direction
            self.surrounding = surrounding_coords(*direction)
            # self.seen = self.seen + self.get_sight(coords=True)

class Level:
    def __init__(self, level):
        self.original = level
        self.width = len(level[0])
        self.height = len(level)
        self.grid = [];
        self.actors = [];
        self.parsed = self.parse_level(level)

    # Return the goal actor
    def goal(self):
        return [x for x in self.actors if x.type == "Goal"][0]

    def distance_from_goal(self, x, y):
        g = self.goal();
        gx, gy = g.pos

        # Calculate and return the distance between the two points
        return math.sqrt( (gx - x) ** 2 + (gy - y) ** 2)

    def get_ch(self, x, y):
        return self.original[y][x]

    def set_ch(self, x, y,value=" "):
        o_r = self.original[y]
        n_r = []
        for i, z in enumerate(o_r):
            if (i == x):
                n_r.append(value)
            else:
                n_r.append(z)

        self.original[y] = "".join(n_r)

    def parse_level(self, plan):
        grid_line = []
        for y in range(self.height):
            line = plan[y]
            for x in range(self.width):
                ch = line[x]
                actor = actorChars.get(ch);
                if actor is not None:
                    self.actors.append(Actor(x, y, ch, self))

                field_type = world_chars.get(ch)
                grid_line.append(field_type)

        self.grid.append(grid_line)
        return self.grid
