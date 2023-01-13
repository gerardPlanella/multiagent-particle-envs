import numpy as np
import math
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario
from multiagent.utils import overlaps, toroidal_distance

class Scenario(BaseScenario):
    def make_world(self, config, size=6.0, n_preds=7, pred_vel=1.2, prey_vel=1.0, sensor_range=4.5, discrete=True):
        world = World()
        # set any world properties
        world.env_key = config.env
        world.n_steps = 500
        world.mode = config.mode
        world.torus = True
        world.dim_c = 2
        world.size = size
        world.origin = np.array([world.size/2, world.size/2])
        world.use_sensor_range = config.use_sensor_range
        if world.use_sensor_range:
            world.init_thresh = config.init_range_thresh

        num_good_agents = 1
        self.n_preds = num_adversaries = n_preds
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent {}'.format(i)
            agent.id = i
            agent.active = True
            agent.captured = False
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 20.0 if agent.adversary else 20.0
            agent.max_speed = pred_vel if agent.adversary else prey_vel # better visibility

        # discrete actions
        world.discrete_actions = discrete

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # generate predators in random circle of random radius with random angles
        redraw = True
        if world.use_sensor_range:
            sample_outside_range = np.random.uniform(0, 1) > (1.0 - world.init_thresh)
        # i = 0
        while redraw:
            # i+=1
            # print(i)
            correct_range = False

            # draw location for prey
            prey_pt = world.origin + np.random.normal(0.0, 0.05, size=2)

            # draw predator locations
            # init_pts = [np.random.uniform(0.0, world.size, size=2) for _ in range(self.n_preds)]
            angles = (np.linspace(0, 2*math.pi, self.n_preds, endpoint=False) + np.random.uniform(0, 2*math.pi)) % 2*math.pi
            radius = np.random.uniform(0.0, 20.0)
            init_pts = [world.origin + (np.array([math.cos(ang), math.sin(ang)])*radius) for ang in angles]

            # predator distance to prey pt
            dists = [toroidal_distance(prey_pt, pt % world.size, world.size) for pt in init_pts]

            # ensure predators not initialized on top of prey
            overlap = overlaps(prey_pt, init_pts, world.size, threshold=0.5)

            if world.mode == 'test' and world.use_sensor_range and world.sensor_range <= 3.5:
                if sample_outside_range:
                    # ensure ALL predators inited outside sensing range
                    correct_range = any(d < world.sensor_range for d in dists)
                else:
                    # ensure ALL predators inited inside sensing range
                    correct_range = any(d > world.sensor_range for d in dists)

                redraw = overlap or correct_range
            else:
                redraw = overlap

        # set initial states
        init_pts.append(prey_pt)
        for i, agent in enumerate(world.agents):
            agent.active = True
            agent.captured = False

            # agents can move beyond confines of camera image --> need to adjust coords accordingly
            agent.state.coords = init_pts[i]
            agent.state.p_pos = agent.state.coords % world.size
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.theta = 0.0
            agent.state.c = np.zeros(world.dim_c)


    def benchmark_data(self, agent, world):
        return { 'active' : agent.active }

    def is_collision(self, agent1, agent2):
        if agent1 == agent2:
            return False
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all active agents that are not adversaries
    def active_good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary and agent.active]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    # return all active adversarial agents
    def active_adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary and agent.active]

    def reward(self, agent, world):
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        if agent.active:
            # Agents are negatively rewarded if caught by adversaries
            rew = 0.1
            shape = False
            adversaries = self.active_adversaries(world)
            if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
                for adv in adversaries:
                    # TODO: IF USING REWARD SHAPING, NEED TO CHANGE TO TOROIDAL DISTANCE
                    rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
            if agent.collide:
                for a in adversaries:
                    if self.is_collision(a, agent):
                        agent.captured = True 
                        rew -= 50
                        break
            return rew
        else:
            return 0.0

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = -0.1
        shape = False
        agents = self.active_good_agents(world)
        adversaries = self.active_adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                # TODO: IF USING REWARD SHAPING, NEED TO CHANGE TO TOROIDAL DISTANCE
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            capture_idxs = []
            for i, ag in enumerate(agents):
                for j, adv in enumerate(adversaries):
                    if self.is_collision(ag, adv):
                        capture_idxs.append(i)
                        ag.captured = True 

            rew += 50 * len(set(capture_idxs))
        return rew

    def terminal(self, agent, world):
        if agent.adversary:
            # predator done if all prey caught
            return all([agent.captured for agent in self.good_agents(world)])
        else:
            # prey done if caught
            return agent.captured

    def observation(self, agent, world):
        # pred/prey observations
        other_pos, other_coords, viz_bits = [], [], []
        extra_ids = []
        for other in world.agents:
            if other is agent: continue

            # sensor range on prey position
            if world.use_sensor_range and not other.adversary:
                pos, bit = self.alter_prey_loc(agent.state.p_pos, other.state.p_pos, world.size, world.sensor_range)
                other_pos.append(pos)
                other_coords.append(pos) # both are [0, 0] when outside sensing range
                viz_bits.append(bit)
            else:
                other_pos.append(other.state.p_pos)
                other_coords.append(other.state.coords)

        # if agent.adversary:
            # other_pos = self.symmetrize(agent.id, other_pos)
            # other_coords = self.symmetrize(agent.id, other_coords)
        
        if world.use_sensor_range:
            obs = np.concatenate([agent.state.p_pos] + other_pos + viz_bits)
        else:
            obs = np.concatenate([agent.state.p_pos] + other_pos)

        return obs

    def alter_prey_loc(self, pred_pos, prey_pos, size, thresh):
        dist = toroidal_distance(pred_pos, prey_pos, size)
        if dist < thresh:
            return prey_pos, np.array([1])
        else:
            return np.zeros_like(prey_pos), np.array([0])

    def symmetrize(self, agent_id, arr):
        # ensure symmetry in obervation space
        # P1 --> IN: P2, P3, P4, P5, P6, P7, P8, P9, P10
        #    --> OUT: P2, P3, P4, P5, P6, P7, P8, P9, P10

        # P2 --> IN: P1, P3, P4, P5, P6, P7, P8, P9, P10
        #    --> OUT: P3, P4, P5, P6, P7, P8, P9, P10, P1

        # P3 --> IN: P1, P2, P4, P5, P6, P7, P8, P9, P10
        #    --> OUT: P4, P5, P6, P7, P8, P9, P10, P1, P2

        # P4 --> IN: P1, P2, P3, P5, P6, P7, P8, P9, P10
        #    --> OUT: P5, P6, P7, P8, P9, P10, P1, P2, P3

        # P5 --> IN: P1, P2, P3, P4, P6, P7, P8, P9, P10
        #    --> OUT: P6, P7, P8, P9, P10, P1, P2, P3, P4

        # P6 --> IN: P1, P2, P3, P4, P5, P7, P8, P9, P10
        #    --> OUT: P7, P8, P9, P10, P1, P2, P3, P4, P5

        # P7 --> IN: P1, P2, P3, P4, P5, P6, P8, P9, P10
        #    --> OUT: P8, P9, P10, P1, P2, P3, P4, P5, P6

        # P8 --> IN: P1, P2, P3, P4, P5, P6, P7, P9, P10
        #    --> OUT: P9, P10, P1, P2, P3, P4, P5, P6, P7

        # P9 --> IN: P1, P2, P3, P4, P5, P6, P7, P8, P10
        #    --> OUT: P10, P1, P2, P3, P4, P5, P6, P7, P8

        # P10 --> IN: P1, P2, P3, P4, P5, P6, P7, P8, P9
        #    --> OUT: P1, P2, P3, P4, P5, P6, P7, P8, P9
        if agent_id == 1:
            # P2
            return arr[1:] + [arr[0]]
        elif agent_id == 2:
            # P3
            return arr[2:] + arr[:2]
        elif agent_id == 3:
            # P4
            return arr[3:] + arr[:3]
        elif agent_id == 4:
            # P5
            return arr[4:] + arr[:4]
        elif agent_id == 5:
            # P6
            return arr[5:] + arr[:5]
        elif agent_id == 6:
            # P7
            return arr[6:] + arr[:6]
        elif agent_id == 7:
            # P8
            return arr[7:] + arr[:7]
        elif agent_id == 8:
            # P9
            return arr[8:] + arr[:8]
        else:
            # P1, P10
            return arr

        

