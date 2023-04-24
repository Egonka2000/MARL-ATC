from gymnasium.spaces import Discrete, Box, Dict
import numpy as np
from bluesky import stack, traf
from bluesky.tools.aero import ft
from bluesky.tools import geo
import bluesky as bs
import random
from bluesky.core.walltime import Timer
import math
from pettingzoo.utils import ParallelEnv

class AtcGymEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}
    def __init__(self, train_mode):
        self.train_mode = train_mode
        self.train_started = False
        self.positions = self.load_positions()
        self.routeDistances()
        self.max_ac = 3
        self.times = [20, 25, 30]
        self.spawn_queue = random.choices(self.times, k=self.positions.shape[0])
        self.active_ac = 0
        self.total_ac = 0
        self.ac_routes = np.zeros(self.max_ac)
        self.update_timer = 0
        self.success_counter = 0
        self.collision_counter = 0
        self.intruders = 2
        self.done = False
        self.possible_agents = ["{}".format(r) for r in range(self.max_ac)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self.agents_prev_alt = {agent: 0 for agent in self.possible_agents}
        self.prev_v_separation = {agent: 0 for agent in self.possible_agents}

        self.action_spaces = {agent: Discrete(3) for agent in self.possible_agents}

        low  = np.array([0, -13000, 0, -10, 0], dtype=np.float32) 
        high = np.array([self.max_d, 36300, 360,  10, self.max_d], dtype=np.float32)

        self.observation_spaces = Dict({
            agent: Dict({
                'agent': Box(low, high, dtype=np.float32),
                'intruder-1': Box(low, high, dtype=np.float32),
                'intruder-2': Box(low, high, dtype=np.float32)
            }) for agent in self.possible_agents
        })

        #distance_to_goal, actual FL, previous FL, route ID, distance_to_agent
        self._states = {
            agent: {
                'agent':  np.array([self.max_d, 28000, 180, 0, 0 ], dtype=np.float32),
                'intruder-1':  np.array([self.max_d, 28000, 180, 0, self.max_d], dtype=np.float32),
                'intruder-2':  np.array([self.max_d, 28000, 180, 0, self.max_d], dtype=np.float32)
        } for agent in self.possible_agents
        }
        
        stack.stack('OP')
        stack.stack('FF')
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]

    def step(self, action_dict):
        self.delete_if_terminated()
        self.truncations = self.terminations
        self.done = (len(traf.id) == 0 or any(self.terminations.values()))
        if not self.done:
            for _idx, action in action_dict.items():
                idx = int(_idx)
                agent_acid = traf.id[idx]
                agent_alt = int(traf.alt[idx] / ft)
                if action == 0:
                    stack.stack('ALT {}, {}'.format(agent_acid, agent_alt - 200))
                if action == 2:
                    stack.stack('ALT {}, {}'.format(agent_acid, agent_alt + 200))
                if action == 1:
                    stack.stack('ALT {}, {}'.format(agent_acid, agent_alt))
                distance, v_separation, nearest_ac_idx = self.nearest_ac(self.get_distance_matrix_ac(), idx)
                nearest_ac_alt = int(traf.alt[int(nearest_ac_idx)] / ft)

                if agent_alt < 20000:
                    if agent_alt < 5000:
                        self.rewards[_idx] = -20
                    else:
                        self.rewards[_idx] = (math.log2(agent_alt / 10000) - 1) * 10
                elif(agent_alt > 33000):
                    self.rewards[_idx] = 1 - (math.pow(math.e, (agent_alt / 10000)) / 5)
                elif distance > 12:
                    if agent_alt == self.agents_prev_alt[_idx]:
                        #print("Distance > 12 és {} a magasság stagnál".format(agent_acid))
                        self.rewards[_idx] = 1
                    else:
                        self.rewards[_idx] = -1
                elif distance > 3:
                    if v_separation <= 5000 and v_separation >= 2000:
                        if int(agent_alt) == int(self.agents_prev_alt[_idx]):
                            #print("Distance > 3 és {} a magasság stagnál".format(agent_acid))
                            self.rewards[_idx] = 30
                        else:
                            #print("Distance > 3 és {} a magasság nem stagnál".format(agent_acid))
                            self.rewards[_idx] = 5
                    elif agent_alt > nearest_ac_alt and agent_alt > self.agents_prev_alt[_idx] and nearest_ac_alt <= self.agents_prev_alt[nearest_ac_idx] and v_separation < 3500:
                        print("Distance > 3 és az {} távolodik a hozzá legközelebbi {} felfelé".format(agent_acid, nearest_ac_idx))
                        self.rewards[_idx] = (math.pow(math.e, (v_separation / 1000)) / math.e)**0.3
                    elif agent_alt < nearest_ac_alt and agent_alt < self.agents_prev_alt[_idx] and nearest_ac_alt >= self.agents_prev_alt[nearest_ac_idx] and v_separation < 3500:
                        print("Distance > 3 és az {} távolodik a hozzá legközelebbi {} lefelé".format(agent_acid, nearest_ac_idx))
                        self.rewards[_idx] = (math.pow(math.e, (v_separation / 1000)) / math.e)**0.3
                    elif(v_separation > 5000):
                        self.rewards[_idx] = 1 - (math.pow(math.e, (v_separation / 1000)) / math.e)**0.3
                    else:
                        self.rewards[_idx] = -5

                self.agents_prev_alt[_idx] = agent_alt
                self.prev_v_separation[_idx] = v_separation

        self.infos = {agent: {} for agent in self.possible_agents}

        if self.train_mode:    
            Timer.update_timers()
            bs.sim.update()

        if not self.done:
            self.get_agents_and_nearest_ac_intruders_states(self.get_distance_matrix_ac())
            
        return self._states, self.rewards, self.terminations, self.truncations, self.infos
    
    def reset(self, seed=None, options=None):
        self.active_ac = 0
        self.total_ac = 0
        self.ac_routes = np.zeros(self.max_ac)
        self.update_timer = 0
        self.spawn_queue = random.choices(self.times, k=self.positions.shape[0])
        self.done = False
        self.agents = self.possible_agents[:]
        self.prev_v_separation = {agent: 0 for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.success_counter = 0
        self.collision_counter = 0

        self.spawn_ac_with_delay()
        if self.train_mode:
            bs.sim.update()
        self.get_agents_and_nearest_ac_intruders_states(self.get_distance_matrix_ac())
        return self._states
    
    def routeDistances(self):
        self.route_distances = []

        for pos in self.positions:
            olat, olon, _, glat, glon = pos
            _, distance = geo.qdrdist(olat, olon, glat, glon)
            self.route_distances.append(distance)
        
        self.max_d = max(self.route_distances) 

    def distance_intruder(self, agent_idx, intruder_idx):
        _, dist = geo.qdrdist(traf.lat[agent_idx], traf.lon[agent_idx], traf.lat[intruder_idx], traf.lon[intruder_idx])
        return dist

    def load_positions(self):
        try:
            positions = np.load('routes/case_study_a_route.npy')
        except:
            positions = np.array([[41.1, -95.0246, 79.5023, 41.449, -90.508], [41.4, -95.0246, 90, 41.449, -90.508], [41.7, -95.0246, 97.2483, 41.449, -90.508]])
            np.save("routes/case_study_a_route.npy", positions)
            positions = np.load('routes/case_study_a_route.npy')
        return positions

    def spawn_ac_with_delay(self):
        if self.total_ac < self.max_ac:
            if self.total_ac == 0:
                for i in range(len(self.positions)):
                    self.spawn_ac(self.total_ac, self.positions[i])
                    self.ac_routes[self.total_ac] = i
                    
                    self.total_ac += 1
                    self.active_ac += 1
            else:
                for k in range(len(self.spawn_queue)):
                    if self.update_timer == self.spawn_queue[k]:
                        self.spawn_ac(self.total_ac, self.positions[k])
                        self.ac_routes[self.total_ac] = k

                        self.total_ac += 1
                        self.active_ac += 1

                        self.spawn_queue[k] = self.update_timer + random.choices(self.times, k=1)[0]
                    if self.total_ac == self.max_ac:
                        break

    def spawn_ac(self, _idx, ac_details):
        lat, lon, hdg, glat, glon = ac_details
        speed = 251#np.random.randint(251, 340)
        alt = np.random.randint(26000, 28000)
        self.agents_prev_alt[str(_idx)] = int(alt)

        stack.stack('CRE SWAN{}, A320, {}, {}, {}, {}, {}'.format(_idx, lat, lon, hdg, alt, speed))
        stack.stack('SWAN{} ADDWPT {}'.format(_idx, "PEA"))
        stack.stack('SWAN{} AFTER {} ADDWPT {}'.format(_idx, "PEA", "KMLI"))
        stack.stack('PAN PEA')      

    def delete_if_terminated(self):
        for i in range(self.max_ac):
            terminal, terminal_type = self.check_that_ac_should_terminated(i)
            if(terminal):
                if terminal_type == 1:
                        self.collision_counter += 1
                        print("Total Coll: {}".format(self.collision_counter))
                else:
                    self.success_counter += 1
                    print("Total Success: {}".format(self.success_counter))
                for i in range(len(traf.id)):                
                    stack.stack('DEL {}'.format(traf.id[i]))
                    self.active_ac -= 1
                    self.terminations[str(i)] = True
                break

    def check_that_ac_should_terminated(self, _idx):
            # If the ac is terminal
            terminal = False
            # The type of terminal that the ac is
            # 0 = not
            # 1 = collision
            # 2 = goal reached
            terminal_type = 0

            distance, v_separation, _ = self.nearest_ac(self.get_distance_matrix_ac(), _idx)
            goal_d = self.dist_goal(_idx)

            if distance <= 1 and v_separation < 2000:
                terminal = True
                terminal_type = 1
            elif goal_d < 5 and terminal == False:
                terminal = True
                terminal_type = 2
            elif traf.ap.route[_idx].wplon[-1] < traf.lon[_idx] and terminal == False:
                terminal = True
                terminal_type = 2

            return terminal, terminal_type

    #Nautical Miles
    def get_distance_matrix_ac(self):
        size = traf.lat.shape[0]
        distances = geo.latlondist_matrix(np.repeat(traf.lat, size), np.repeat(traf.lon, size), np.tile(traf.lat, size), np.tile(traf.lon, size)).reshape(size, size)
        return distances

    #Nautical Miles
    def nearest_ac(self, dist_matrix, _idx):
        row = dist_matrix[:,_idx]
        close = 10e+25
        alt_separations = 0
        nearest_ac_idx = _idx
        
        for i, dist in enumerate(row):
            if i != _idx and dist < close:
                close = dist
                this_alt = traf.alt[_idx]
                close_alt = traf.alt[i]
                nearest_ac_idx = str(i)
                alt_separations = abs(this_alt - close_alt) / ft

        return close, alt_separations, nearest_ac_idx

    def get_agents_and_nearest_ac_intruders_states(self, dist_matrix):
        for _idx in range(self.max_ac):
            row = dist_matrix[:,_idx]
            sorted_idx = np.array(np.argsort(row, axis=0))
            intruder_count = 0
            for i, idx in enumerate(sorted_idx):
                if idx == _idx:
                    self._states["{}".format(_idx)]["agent"] = self._get_state(idx[0], _idx)
                else:
                    self._states["{}".format(_idx)]["intruder-{}".format(i)] = self._get_state(idx[0], _idx)
                    intruder_count += 1
                    if intruder_count == self.intruders:
                        break
    
    def _get_state(self, _idx, idx):
        if(idx == _idx):
            return np.array(
                [
                    self.dist_goal(idx),
                    int(traf.alt[idx] / ft),
                    traf.hdg[idx],
                    traf.vs[idx],
                    0
                ], dtype=np.float32
            )
        else:
            return np.array(
                [
                    self.dist_goal(_idx),
                    int(traf.alt[_idx] / ft),
                    traf.hdg[_idx],
                    traf.vs[_idx],
                    self.distance_intruder(_idx, idx)
                ], dtype=np.float32
            )

    #Nautical Miles
    def dist_goal(self, _idx):
        olat = traf.lat[_idx]
        olon = traf.lon[_idx]
        ilat,ilon = traf.ap.route[_idx].wplat[-1],traf.ap.route[_idx].wplon[-1]
        
        _, dist = geo.qdrdist(olat,olon,ilat,ilon)
        return dist