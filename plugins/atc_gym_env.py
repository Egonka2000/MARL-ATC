from gymnasium.spaces import Discrete, Box, Dict, Tuple
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
        self.intruders = 1
        self.possible_alts = [25000, 28000, 31000]
        self.done = False
        self.possible_agents = ["{}".format(r) for r in range(self.max_ac)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self.agents_prev_action = {agent: 1 for agent in self.possible_agents}

        self.action_spaces = {agent: Discrete(3) for agent in self.possible_agents}

        low  = np.array([0, -6000, -360, -10], dtype=np.float32) 
        high = np.array([self.max_d, 6000, 360,  10], dtype=np.float32)

        self.observation_spaces = Dict({
            agent: Dict({
                'intruder-traf': Box(low, high, dtype=np.float32), 
                'intruder-act':  Discrete(3)
            }) for agent in self.possible_agents
        })

        self.reset_states()
        
        stack.stack('OP')
        stack.stack('FF')

    def reset_states(self):
        #distance, v_separation, hdg_diff, vspeed, previous action
        self._states = {
            agent: {
                'intruder-traf': np.array([self.max_d, 0, 180, 0], dtype=np.float32),
                'intruder-act': 0                   
            } for agent in self.possible_agents
        }

    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]

    def step(self, action_dict):
        terminal, terminal_type, agent_term, nearest_agent_term = self.delete_if_terminated()
        self.truncations = self.terminations
        self.done = (len(traf.id) == 0 or any(self.terminations.values()))
        if not self.done:
            for agent, action in action_dict.items():
                ac_idx = int(agent)                        
                distance, v_separation, hdg_diff, nearest_agent = self.nearest_ac(self.get_distance_matrix_ac(), ac_idx)
                nearest_ac_vs  = traf.vs[int(nearest_agent)]

                if distance > 10:
                    self.rewards[agent] = 0
                elif abs(v_separation) > 2000:
                    if action == self.agents_prev_action[agent]:
                        self.rewards[agent] = 5
                    else:
                        self.rewards[agent] = 0
                elif (action == action_dict[nearest_agent]):
                    self.rewards[agent] = -1                     
                elif v_separation > 0 and action > action_dict[nearest_agent]:
                    self.rewards[agent] = 0.9*(v_separation/2000)
                elif v_separation < 0 and action < action_dict[nearest_agent]:
                    self.rewards[agent] = 0.9*(v_separation/2000)
                else:
                    self.rewards[agent] = -1 + 0.9*(v_separation / 2000)    

                self.agents_prev_action[agent] = action

                self._states[agent]["intruder-traf"] = np.array([distance, v_separation, hdg_diff, nearest_ac_vs], dtype=np.float32)
                self._states[agent]["intruder-act"]  = action_dict[nearest_agent]                 

                stack.stack('ALT {}, {}'.format(traf.id[ac_idx], self.possible_alts[action]))

                if self.train_mode:
                    Timer.update_timers()
                    bs.sim.update() 
                
        if terminal:            
            if terminal_type == 1:
                self.rewards[agent_term] = -1000
                self.rewards[nearest_agent_term] = -1000                
            if terminal_type == 2:
                self.rewards = {agent: 1000 for agent in self.possible_agents}

        self.infos = {agent: {} for agent in self.possible_agents}
                       
        return self._states, self.rewards, self.terminations, self.truncations, self.infos
    
    def reset(self, seed=None, options=None):
        self.active_ac = 0
        self.total_ac = 0
        self.ac_routes = np.zeros(self.max_ac)
        self.update_timer = 0
        self.spawn_queue = random.choices(self.times, k=self.positions.shape[0])
        self.done = False
        self.agents = self.possible_agents[:]
        self.agents_prev_action = {agent: 0 for agent in self.possible_agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self.spawn_ac_with_delay()
        
        if len(traf.id) != 0:
            self.reset_states()

        if self.train_mode:
            bs.sim.update()    

        return self._states
    
    def routeDistances(self):
        self.route_distances = []

        for pos in self.positions:
            olat, olon, _, glat, glon = pos
            _, distance = geo.qdrdist(olat, olon, glat, glon)
            self.route_distances.append(distance)
        
        self.max_d = max(self.route_distances)

    def load_positions(self):
        try:
            positions = np.load('routes/case_study_a_route.npy')
        except:
            positions = np.array([[41.1, -95.0246, 79.5023, 41.449, -90.508], [41.7, -95.0246, 97.2483, 41.449, -90.508]])
            np.save("routes/case_study_b_route.npy", positions)
            positions = np.load('routes/case_study_b_route.npy')
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
        
        stack.stack('CRE SWAN{}, A320, {}, {}, {}, {}, {}'.format(_idx, lat, lon, hdg, alt, speed))
        stack.stack('SWAN{} ADDWPT {}'.format(_idx, "PEA"))
        stack.stack('SWAN{} AFTER {} ADDWPT {}'.format(_idx, "PEA", "KMLI"))
        stack.stack('PAN PEA')      

    def delete_if_terminated(self):
        for i in range(self.max_ac):
            agent_term = str(i)

            terminal, terminal_type, nearest_agent_term = self.check_that_ac_should_terminated(i)
            
            if(terminal):
                if terminal_type == 1:
                    self.collision_counter += 1                    
                else:
                    self.success_counter += 1

                print("Total Success: {} Total Coll: {}".format(self.success_counter, self.collision_counter))

                for i in range(len(traf.id)):                
                    stack.stack('DEL {}'.format(traf.id[i]))
                    self.active_ac -= 1
                    self.terminations[str(i)] = True
                
                if self.train_mode:
                    Timer.update_timers()
                    bs.sim.update() 
                break
        return terminal, terminal_type, agent_term, nearest_agent_term

    def check_that_ac_should_terminated(self, _idx):
            # If the ac is terminal
            terminal = False
            # The type of terminal that the ac is
            # 0 = not
            # 1 = collision
            # 2 = goal reached
            terminal_type = 0

            distance, v_separation, _, nearest_agent = self.nearest_ac(self.get_distance_matrix_ac(), _idx)
            goal_d = self.dist_goal(_idx)

            if distance <= 1 and abs(v_separation) < 1500:
                terminal = True
                terminal_type = 1
            elif goal_d < 5 and terminal == False:
                terminal = True
                terminal_type = 2
            elif traf.ap.route[_idx].wplon[-1] < traf.lon[_idx] and terminal == False:
                terminal = True
                terminal_type = 2

            return terminal, terminal_type, nearest_agent

    #Nautical Miles
    def get_distance_matrix_ac(self):
        size = traf.lat.shape[0]
        distances = geo.latlondist_matrix(np.repeat(traf.lat, size), np.repeat(traf.lon, size), np.tile(traf.lat, size), np.tile(traf.lon, size)).reshape(size, size)
        return distances

    #Nautical Miles
    def nearest_ac(self, dist_matrix, _idx):
        row = dist_matrix[:,_idx]
        nearest_dist = 10e+25
        v_separation = 0
        
        for i, dist in enumerate(row):
            if i != _idx and dist < nearest_dist:
                nearest_dist  = dist
                
                this_alt      = traf.alt[_idx]
                nearest_alt   = traf.alt[i]

                this_hdg      = traf.hdg[_idx]
                nearest_hdg   = traf.hdg[i]

                nearest_agent = str(i)

                v_separation  = (this_alt - nearest_alt) / ft
                hdg_diff      = this_hdg - nearest_hdg 

        return float(nearest_dist), v_separation, hdg_diff, nearest_agent

    #Nautical Miles
    def dist_goal(self, _idx):
        olat = traf.lat[_idx]
        olon = traf.lon[_idx]
        ilat,ilon = traf.ap.route[_idx].wplat[-1],traf.ap.route[_idx].wplon[-1]
        
        _, dist = geo.qdrdist(olat,olon,ilat,ilon)
        return dist