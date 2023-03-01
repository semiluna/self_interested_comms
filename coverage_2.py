from matplotlib import colors
import colorsys
import copy
import numpy as np
from enum import Enum

import gymnasium as gym
from gymnasium import spaces

DEFAULT_OPTIONS = {
    'world_shape': [24, 24],
    'state_size': 48,
    'collapse_state': False,
    'termination_no_new_coverage': 10,
    'max_episode_len': -1,
    "min_coverable_area_fraction": 0.6,
    "map_mode": "random",
    "n_agents": [5],
    "disabled_teams_step": [False],
    "disabled_teams_comms": [False],
    'communication_range': 8.0,
    'one_agent_per_cell': False,
    'ensure_connectivity': True,
    'reward_type': 'semi_cooperative',
    #"operation_mode": 'all', # greedy_only, coop_only, don't default for now
    'episode_termination': 'early',
    'agent_observability_radius': None,
}

class Dir(Enum):
    RIGHT  = 0
    LEFT   = 1
    UP     = 2
    DOWN   = 3

X = 1
Y = 0

ROW = 0
COL = 1

class WorldMap():
    def __init__(self, random_state, shape, min_coverable_area_fraction):
        self.shape = tuple(shape)
        self.min_coverable_area_fraction = min_coverable_area_fraction
        self.reset(random_state)

    def reset(self, random_state, mode="random"):
        self.coverage = np.zeros(self.shape, dtype=np.int)
        if mode == "random":
            if self.min_coverable_area_fraction == 1.0:
                self.map = np.zeros(self.shape, dtype=np.uint8)
            else:
                self.map = np.ones(self.shape, dtype=np.uint8)
                p = np.array([random_state.randint(0, self.shape[c]) for c in [Y, X]])
                while self.get_coverable_area_faction() < self.min_coverable_area_fraction:
                    d_p = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]][random_state.randint(0, 4)])
                    p_new = np.clip(p + d_p, [0,0], np.array(self.shape)-1)
                    self.map[min(p[Y],p_new[Y]):max(p[Y],p_new[Y])+1, min(p[X],p_new[X]):max(p[X],p_new[X])+1] = 0
                    p = p_new

    def get_coverable_area_faction(self):
        coverable_area = ~(self.map > 0)
        return np.sum(coverable_area)/(self.map.shape[X]*self.map.shape[Y])

    def get_coverable_area(self):
        coverable_area = ~(self.map>0)
        return np.sum(coverable_area)

    def get_covered_area(self):
        coverable_area = ~(self.map>0)
        return np.sum((self.coverage > 0) & coverable_area)

    def get_coverage_fraction(self):
        coverable_area = ~(self.map>0)
        covered_area = (self.coverage > 0) & coverable_area
        return np.sum(covered_area)/np.sum(coverable_area)

class Action(Enum):
    NOP         = 0
    MOVE_RIGHT  = 1
    MOVE_LEFT   = 2
    MOVE_UP     = 3
    MOVE_DOWN   = 4

class Robot():
    def __init__(self, index, random_state, world, observability_radius):
        self.index = index
        self.world = world
        self.radius = observability_radius
        self.pose = np.array([-1, -1])
        self.reset(random_state)
    
    def reset(self, random_state, pose_mean=np.array([0, 0]), pose_var=1):
        def random_pos(var):
            return np.array([
                int(np.clip(random_state.normal(loc=pose_mean[c], scale=var), 0, self.world.map.shape[c]-1))
            for c in [ROW, COL]])

        current_pose_var = pose_var
        self.pose = random_pos(current_pose_var)
        self.prev_pose = self.pose.copy()
        while self.world.map.map[self.pose[Y], self.pose[X]] == 1 or (self.world.is_occupied(self.pose, self) and self.one_agent_per_cell):
            self.pose = random_pos(current_pose_var)
            current_pose_var += 0.1

        self.coverage = np.zeros(self.world.map.shape, dtype=np.bool)
        self.state = None
        self.no_new_coverage_steps = 0
        self.reward = 0
    
    def step(self, action):
        action = Action(action)

        delta_pose = {
            Action.MOVE_RIGHT:  [ 0,  1],
            Action.MOVE_LEFT:   [ 0, -1],
            Action.MOVE_UP:     [-1,  0],
            Action.MOVE_DOWN:   [ 1,  0],
            Action.NOP:         [ 0,  0]
        }[action]

        is_valid_pose = lambda p: all([p[c] >= 0 and p[c] < self.world.map.shape[c] for c in [Y, X]])
        is_obstacle = lambda p: self.world.map.map[p[Y]][p[X]] == 1

        self.prev_pose = self.pose.copy()
        desired_pos = self.pose + delta_pose
        # NOTE: multiple agents can be on the same position
        if is_valid_pose(desired_pos) and (not is_obstacle(desired_pos)):
            self.pose = desired_pos

        if self.world.map.coverage[self.pose[Y], self.pose[X]] == 0:
            self.world.map.coverage[self.pose[Y], self.pose[X]] = self.index
            self.reward = 1
            self.no_new_coverage_steps = 0
        else:
            self.reward = 0
            self.no_new_coverage_steps += 1

        self.coverage[self.pose[Y], self.pose[X]] = True
    
    def update_state(self):
        coverage = self.coverage.copy().astype(np.int8)

        # Get local information about obstacles, anything outside is an `obstacle`
        local_world = self.local_frame(self.world.map.map, fill=1)
        # Get local information on my previous coverage
        local_coverage = self.local_frame(coverage, fill=0)
        
        # Get neighbours within field-of-vision
        local_robots = np.zeros(self.world.map.shape, dtype=np.uint8)
        for agent in self.world.agents.values():
            if agent is not self and np.sum((agent.pose - self.pose)**2) < self.radius**2:
                local_robots[agent.pose[ROW], agent.pose[COL]] = 2
        local_robots[self.pose[ROW], self.pose[COL]] = 1
        local_robots = self.local_frame(local_robots, fill=0)
        
        self.state = np.stack([local_world, local_coverage, local_robots], axis=-1).astype(np.uint8)
    
    def local_frame(self, grid, fill=0):
        local_grid = np.full(self.world.map.shape, fill_value=fill)
        MAX_R, MAX_C = self.world.map.shape
        center_row = MAX_R // 2
        center_col = MAX_C // 2
        for r in range(0, MAX_R):
            for c in range(0, MAX_C):
                dr = r - self.pose[ROW]
                dc = c - self.pose[COL]
                if dr**2 + dc**2 < self.radius**2:
                    local_grid[center_row + dr, center_col + dc] = grid[r, c]
        return local_grid


class CoverageEnv(gym.Env):
    def __init__(self, env_config):
        self.seed()

        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(env_config)

        self.fig = None
        self.map_colormap = colors.ListedColormap(['white', 'black', 'gray'])

        self.n_agents = self.cfg['n_agents']
        self.agent_colors = [self.generate_color(a_id) for a_id in range(self.n_agents)]

        map_size = self.cfg['world_shape']
        self.observation_space = spaces.Dict({
            'agents': spaces.Tuple((
                spaces.Dict({
                'map': spaces.Box(0, np.inf, shape=(map_size[ROW], map_size[COL], 3)),
                'pos': spaces.Box(low=np.array([0, 0]), high=np.array([map_size[ROW], map_size[COL]]), dtype=np.int8)
                }),
            ) * self.n_agents),
            # global adjacency matrix
            'gso': spaces.Box(-np.inf, np.inf, shape=(self.n_agents, self.n_agents)),
            # TODO: might need to add global state? anyway
        })
        self.action_space = spaces.Tuple((spaces.Discrete(5),) * self.n_agents)

        self.map = WorldMap(self.world_random_state, 
                            map_size, 
                            self.cfg['min_coverage_area_fraction'])
        
        self.agents = []
        for idx in range(self.n_agents):
            self.agents.append(
                Robot(
                    idx,
                    self.agent_random_state,
                    self,
                    self.cfg['agent_observability_radius'],
                )
            )
        
        self.reset()
    
    def reset(self):
        self.dones = [False] * self.n_agents
        self.timestep = 0
        self.map.reset(self.world_random_state, self.cfg['map_mode'])
    
        def random_pos_seed():
            rnd = self.agent_random_state
            if self.cfg['map_mode'] == "random":
                return np.array([rnd.integers(0, self.map.shape[c]) for c in [Y, X]])

        pose_seed = None
        if not self.cfg['map_mode'] == "random" or pose_seed is None:
            # shared pose_seed if random map mode
            pose_seed = random_pos_seed()
            while self.map.map[pose_seed[Y], pose_seed[X]] == 1:
                pose_seed = random_pos_seed()
            for agent in self.agents:
                agent.reset(self.agent_random_state, pose_mean=pose_seed, pose_var=1)

        return self.step([Action.NOP]*self.cfg['n_agents'])[0]

    def compute_gso(self):
        all_agents = self.agents
        dists = np.zeros((len(all_agents), len(all_agents)))
        
        for agent_y in range(len(all_agents)):
            for agent_x in range(agent_y):
                dst = np.sum(np.array(all_agents[agent_x][0].pose - all_agents[agent_y][0].pose)**2)
                dists[agent_y, agent_x] = dst
                dists[agent_x, agent_y] = dst

        current_dist = self.cfg['communication_range']
        A = dists < (current_dist**2)
        active_row = ~np.array([a[1] for a in all_agents])
        if self.cfg['ensure_connectivity']:
            def is_connected(m):
                def walk_dfs(m, index):
                    for i in range(len(m)):
                        if m[index][i]:
                            m[index][i] = False
                            walk_dfs(m, i)

                m_c = m.copy()
                walk_dfs(m_c, 0)
                return not np.any(m_c.flatten())

            # set done teams as generally connected since they should not be included by increasing connectivity
            while not is_connected(A[active_row][:, active_row]):
                current_dist *= 1.1
                A = (dists < current_dist**2)

        # Mask out done agents
        A = (A).astype(np.int)

        # normalization: refer https://github.com/QingbiaoLi/GraphNets/blob/master/Flocking/Utils/dataTools.py#L601
        np.fill_diagonal(A, 0)
        deg = np.sum(A, axis = 1) # nNodes (degree vector)
        D = np.diag(deg)
        Dp = np.diag(np.nan_to_num(np.power(deg, -1/2)))
        L = A # D-A
        gso = Dp @ L @ Dp
        return gso

    def step(self, actions):
        self.timestep += 1
        
        for idx, agent in enumerate(self.agents):
            agent.step(actions[idx])

        states, rewards = [], []
        for idx, agent in enumerate(self.agents):
            state, reward, done, _ = agent.update_state()
            states.append(state)
            rewards.append(reward)
            self.dones[idx] = done

        pose_map = np.zeros(self.map.shape, dtype=np.uint8)
        for agent in self.agents:
            pose_map[agent.pose[ROW], agent.pose[COL]] = 1
        
        global_state = np.stack([self.map.map, self.map.coverage > 0, pose_map], axis=-1)
        world_done = self.timestep == self.cfg['max_episode_len'] or \
                        self.map.get_coverage_fraction() == 1.0  
        
        if self.cfg['episode_termination'] == 'early':
            robots_done = any(self.dones)
        else:
            raise NotImplementedError("Unknown termination mode", self.cfg['episode termination'])
        
        done = world_done or robots_done

        state = {
            'agents': tuple([{
                'map': states[a_id],
                'pos': self.agents[a_id].pose,
            } for a_id in range(len(self.agents))]),
            'gso': self.compute_gso(),
            'state': global_state,
        }

        info = {
            'current_global_coverage': self.map.get_coverage_fraction(),
            'coverable_area': self.map.get_coverable_area(),
            'rewards': rewards,
        }

        return state, sum(rewards), done, info

    # ======================= RENDER MODE ============================
    def generate_color(self, agent_id):
            # Scale the agent ID to the range [0, 1]
            normalized_id = agent_id / self.cfg['n_agents']
            
            # Use the HSV color space to generate a color
            hue = normalized_id
            saturation = 0.8
            value = 0.8
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            
            # Convert the RGB color to a hex string
            hex_color = '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
            
            return hex_color