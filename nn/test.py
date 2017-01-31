from prettyprint import pp

from net import AgentController
from visualization import Display
import subprocess as sp
from sets import Set
import numpy as np
import time
from collections import deque
import random
from random import choice

def flip_up(level):
    return level[::-1]

def flip_left(level):
    return [x[::-1] for x in level]

def rotate(level):
    return flip_up(flip_left(level))

def random_translation(level):
    translation_map = [
        flip_up,
        flip_left,
        rotate,
    ]
    return choice(translation_map)(level)

desired_map = [
  "  xxxxxxxxxxxxxxxxxxxx  ",
  "  x                  x  ",
  "  x                  x  ",
  "  x       o       G  x  ",
  "  x    o         xxxxx  ",
  "  x    x             x  ",
  "  x    x  o          x  ",
  "  x    xxxxxxx       x  ",
  "  x                  x  ",
  "  x                  x  ",
  "  xxxx         o o   x  ",
  "  xx@        xxxxx   x  ",
  "  xxxxxxx            x  ",
  "        x!!!!!!!!!!!!x  ",
  "        xxxxxxxxxxxxxx  ",
  "                        "
]


desired_map2 = [
  "  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  ",
  "  x                                                            x  ",
  "  x                                                            x  ",
  "  x                                        o                   x  ",
  "  x                                    o                       x  ",
  "  x                                                            x  ",
  "  x                                              o             x  ",
  "  x                                                            x  ",
  "  x                                                 o          x  ",
  "  x                       o    o        o                      x  ",
  "  x                         xxxxxxxxxxxxxxxxxxx                x  ",
  "  x                         x         !                o       x  ",
  "  x                         x         1                        x  ",
  "  x                         x        !!                   G    x  ",
  "  xxx   xxxxx    xxxx o xxxxx                                  x  ",
  "  x                         x                                  x  ",
  "  x                         x                                  x  ",
  "  x                         x                                  x  ",
  "  x            o            x                                  x  ",
  "  x            o            x                                  x  ",
  "  x                         x                       o          x  ",
  "  x            o            x       o                          x  ",
  "  x                         x                                  x  ",
  "  x                         x                                  x  ",
  "  x                                                            x  ",
  "  x                                xxxxxxxxxxxxxxxxx    xxxxxxxx  ",
  "  x                                                            x  ",
  "  xxxxxxxxxxxxxxxxx                                            x  ",
  "  x               x                                            x  ",
  "  x               x       o                                    x  ",
  "  x               x                        !!!                 x  ",
  "  x                                                            x  ",
  "  x               o       o                                    x  ",
  "  x                                                            x  ",
  "  x     o o o     x                                            x  ",
  "  x    !!!!!o     x                                            x  ",
  "  x     o!!!o     x                                            x  ",
  "  x      !!!o     x       o      o   o   o    o    o           x  ",
  "  x     o o o     x                                            x  ",
  "  x                                                            x  ",
  "  x               o                                o           x  ",
  "  x       o       x                                            x  ",
  "  x               x                                            x  ",
  "  x!!!!           x                                o           x  ",
  "  xxxxxxxo xxxxxxxx                                            x  ",
  "  x                                                            x  ",
  "  x                           xxxx   xxxxxxxxxxx   o   xxxxxxxxx  ",
  "  x       o         o         x                                x  ",
  "  x                           x                                x  ",
  "  x                           x                                x  ",
  "  x                           x                                x  ",
  "  xxxxxxx o xxxxxx     xxxxxxxx      o                         x  ",
  "  x              x               o   xx                 o   o  x  ",
  "  x              x           o                        xxxxxxxxxx  ",
  "  x       o      o        o                                    x  ",
  "  x @            x       xxxxx                            o    x  ",
  "  xxxxxxxxxxxx!!!x                                             x  ",
  "  xxxxxxxxxxxxxxxx!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!x  ",
  "  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  ",
]


from level import Level

# The brain of our player
agent = AgentController()

level = Level(desired_map[:], agent)


completed_goal_count = 0

# Find our player
player = [x for x in level.actors if x.type == "Player"][0]

# How far are we from the goal? (G)
distance_from_goal = lambda: level.distance_from_goal(*player.pos)
last_action = None
# record the last position
last_distance = distance_from_goal()
count = 0
done_count = 0
global_count = 0
display = Display(desired_map[:])
previously_seen = Set()
coin_count = 0
move_history = deque(maxlen=15)
zero_state = np.zeros((1, agent.dqn.model.hidden_size))
state_in = (zero_state, np.copy(zero_state))
state_out = state_in
while True:
    observation = player.get_sight(coords=True, norm=True)
    # pp(observation)

    # Ask the agent what we should do with this state
    action, state_out = agent.get_action(observation, state_in)

    # store our last move

    # Did we touch the goal?
    touched_goal = player.get_resulting_ch(action) == "G"
    touched_lava = player.get_resulting_ch(action) == "!"
    touched_coin = player.get_resulting_ch(action) == "o"
    touched_wall = player.get_resulting_ch(action) == "x"

    old_distance = distance_from_goal()
    last_seen = len(player.seen)

    # apply the action to our game:
    player._move(action)

    backtrack_amt = -1 if action < 4 else -0.3
    reward = backtrack_amt if (player.pos in move_history) else 0
    # reward = 0
    move_history.append(player.pos)

    keymap = agent.code_from_int(action)

    new_observation = player.get_sight(coords=True, norm=True)

    s_ch = [x[0] for x in player.get_sight(coords=True)]

    new_distance = distance_from_goal()

    goalbound_reward = 0

    if (completed_goal_count < 10):
        goalbound_reward = 1 if old_distance > new_distance else -0.1 if old_distance < new_distance else 0

    reward = reward + goalbound_reward

    done = False
    weigh = lambda weight, i, chs: sum([weight for x in i if x in chs])

    # We do like seeing empty spaces, coins, and the goal however
    # optimal_space_reward_sum = weigh(1, s_ch, ["G"])
    # coin_seen_reward = weigh(len(player.seen) / 8, s_ch, ["o"])
    player.has_explored()

    currently_seen = len(player.seen)
    reward += 1 if last_seen < currently_seen else 0

    # percentage_explored = len(player.seen) * 1.5
    # reward = reward + optimal_space_reward_sum + coin_seen_reward

    if touched_lava:
        reward -= 5
        done = True

    if touched_wall:
        # reward -= (reward * 4)
        reward = -3
        done = True

    if touched_goal:
        reward += 6
        done = True
        completed_goal_count += 1

    if count == 1000:
        done = True
        reward = reward / 2

    if touched_coin:
        reward += 1.5

    # Reward or punish the agent for it's good/bad work
    agent.reward_action(observation, new_observation, action, state_in, state_out, reward, done=done)
    state_in = state_out
    # print reward
    if done:
        done_count += 1
        state_in = (zero_state, np.copy(zero_state))
        # Reset our map
        level = Level(desired_map[:], agent)

        if touched_goal and completed_goal_count > 10:
            print "#TRIGGERED"
            # Reset our map
            level = Level(random_translation(desired_map[:]), agent)
            completed_goal_count = 5

        move_history = deque(maxlen=15)

        # Find our player
        player = [x for x in level.actors if x.type == "Player"][0]

        # Add our reward sum to our memory
        agent.buf.append(agent.reward_sum)

        # Reset
        distance_from_goal = lambda: level.distance_from_goal(*player.pos)
        player.seen.clear()
        count = 0
        coin_count = 0


    # Display:
    # Maintain a copy of our level for display purposes
    l = Level(level.original[:], agent)

    # If the show fog button was hit in the display, iterate through our seen items
    # : = seen over n steps, . = seen over current step
    if display.show_fog:
        seen_coords = [(x[1], x[2]) for x in player.seen]
        for y_i, y in enumerate(l.original):
            for x_i, x in enumerate(y):
                if (x_i, y_i) not in seen_coords:
                    l.set_ch(x_i, y_i, ":")

        seen_coords = [(x[1], x[2]) for x in previously_seen]
        for y_i, y in enumerate(l.original):
            for x_i, x in enumerate(y):
                if (x_i, y_i) not in seen_coords:
                    l.set_ch(x_i, y_i, ".")



    previously_seen = previously_seen | player.seen

    # Update our pygame display
    display.episode = count
    display.current_reward = agent.reward_sum
    display.mean_reward = np.mean(agent.buf)
    display.action = keymap
    display.player = player
    display.update(l.original)
    agent.reward_sum = 0
    count += 1

    if (global_count % 30) == 0:
        print done_count

    global_count += 1

    # char_mult = len(l.original[0])
    # print("#" * char_mult)
    # # print seen_ch
    # # print "# Last distance: %s #" % last_distance
    # print "# Current distance: %s #" % new_distance
    # print "# Key pressed:%s #" % keymap
    # print player.seen
    # print("#" * char_mult)
    # print
    # pp(level.original)
    # print("#" * char_mult)
    # print("# `.` = already visited #")
    # print("#" * char_mult)
    # pp(l.original)
    # print
    # print
    # print "Episode #" + str(count)
    # print "Reward for this episode: ", agent.reward_sum
    # print "Average reward for last 1000 episodes: ", np.mean(agent.buf)
    # last_distance = distance_from_goal()
    # time.sleep(1)


    # x.run_observation(np.zeros((720, )))


# from prettyprint import pp
# dgoal = lambda : str(level.distance_from_goal(*player.pos))
#
# print("New distance from goal: " + dgoal())
# print
# pp("\n".join(level.original))

# print "\n".join(level.original)
# import ipdb; ipdb.set_trace()
