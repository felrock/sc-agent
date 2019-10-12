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


FUNCTIONS = actions.FUNCTIONS

units = []


def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""

  y, x = mask.nonzero()
  return list(zip(x, y))

def moveMarineAwayFromRoaches(roaches, marine):

    new_pos = (0,0)

    roach_avg_x = roaches.sum(key=lambda x:x[0])/len(roaches)
    roach_avg_v = roaches.sum(key=lambda x:x[1])/len(roaches)
    
    new_pos = (marine[0]-roach_avg_x[0], marine[1]-roach_avg_x[1])

    return new_pos

def findMarineWithLowHpAndBeingAttack(marines):
    pass


class DefeatRoaches(base_agent.BaseAgent):
    """An agent specifically for solving the DefeatRoaches map."""

    def step(self, obs):
        super(DefeatRoaches, self).step(obs)
        if FUNCTIONS.Attack_screen.id in obs.observation.available_actions:

            _TERRAN_COMMANDCENTER = 18
            _UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
            unit_type = obs.observation['screen'][_UNIT_TYPE]
            cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero() 

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

