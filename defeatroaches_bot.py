# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scripted agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features, units
import sklearn

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY
_PLAYER_ALLY = features.PlayerRelative.ALLY

_hp = 2


FUNCTIONS = actions.FUNCTIONS

def _xy_locs(mask):
    """Mask should be a set of bools from comparison with a feature layer."""

    y, x = mask.nonzero()
    return list(zip(x, y))


def WhoHasAggro(prev_hp, curr_hp):

    is_low = 20
    for i in range(len(curr_hp)):
        if curr_hp[i] < is_low:
            if curr_hp[i] - prev_hp[i] != 0:
                return i

    return None

class DefeatRoaches(base_agent.BaseAgent):
    """An agent specifically for solving the DefeatRoaches map."""

    def __init___(self, *args, **kwags):

        super(DefeatRoaches, self).__init__(*args, **kwargs)
        self.step_delay = 0
        self.units_hp = []

    def reset(self):

        super(DefeatRoaches, self).reset()
        self.step_delay = 0
        self.units_hp = []

    def step(self, obs):

        super(DefeatRoaches, self).step(obs)
        
        if self.step_delay ==  50:
            # move selected marine
            self.step_delay -= 1
            player_relative = obs.observation.feature_screen.player_relative
            
            roaches = _xy_locs(player_relative == _PLAYER_ENEMY)
            marines = _xy_locs(player_relative == _PLAYER_ALLY)

            mm = marines.mean()
            rm = roaches.mean()
            new_pos = MoveAway(mm, rm)

            return FUNCTIONS.Scan_Move_screen('now', new_pos)

        elif self.step_delay > 0:
            # select army after sufficient time

            self.step_delay -= 1
            return FUNCTIONS.no_op()
       else:

            # step_delay is over 
            return FUNCTIONS.select_army('select')

        if FUNCTIONS.Attack_screen.id in obs.observation.available_actions:

            selected_units = obs.observation['multi_select'] 

            if len(self.units_hp) > 0:
                # check which marines has aggro and move them
                # the unit that is losing hp has aggro
                unit_w_aggro = WhoHasAggro(self.units_hp, selected_units)
                if unit_w_aggro:

                    # que work for aggro unit
                    coord = FindMoveLocation(obs)
                    self.step_delay = 50
                    return FUNCTIONS.select_unit('select', unit_w_aggro)
            else:

                # stores all the hp from all the units selected
                self.units_hp = [unit[_hp] for unit in selected_units]
            
            player_relative = obs.observation.feature_screen.player_relative
            roaches = _xy_locs(player_relative == _PLAYER_ENEMY)
            if not roaches:
                return FUNCTIONS.no_op()

            # Find the roach with max y coord.
            target = roaches[numpy.argmax(numpy.array(roaches)[:, 1])]
            return FUNCTIONS.Attack_screen("now", target)

        if FUNCTIONS.select_army.id in obs.observation.available_actions:

            return FUNCTIONS.select_army("select")


        return FUNCTIONS.no_op()
