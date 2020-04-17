import numpy as np
import math
import copy
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, size, bounded, n_preds, pred_vel, prey_vel, baseline, pred_init, noise=False, discrete=True):
        world = World()
        # set any world properties
        world.dim_c = 2
        world.size = size
        world.origin = np.array([world.size/2, world.size/2])

        num_good_agents = 1
        self.n_preds = n_preds
        num_adversaries = n_preds
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0
        # num_landmarks = 2


        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            # agent.silent = False
            agent.adversary = True if i < num_adversaries else False
            agent.silent = False if agent.adversary else True
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 3.0
            agent.max_speed = pred_vel if agent.adversary else prey_vel # better visibility

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False

        # handle boundaries
        if bounded:
            # add border walls
            world.bounded = True
            left_wall = Wall(orient='V', axis_pos=2.25, endpoints=(-0.75, 2.75), width=0.5)
            right_wall = Wall(orient='V', axis_pos=-0.25, endpoints=(-0.75, 2.75), width=0.5)
            top_wall = Wall(axis_pos=2.25, endpoints=(-0.75, 2.75), width=0.5)
            bot_wall = Wall(axis_pos=-0.25, endpoints=(-0.75, 2.75), width=0.5)
            world.walls = [left_wall, top_wall, bot_wall, right_wall]
        else:
            world.bounded = False
            
        # baseline: always generate landmarks in same location
        self.baseline = baseline
        if self.baseline:
            self.landmark_pos = [np.array([0.25, 1.2]), np.array([0.7, 0.45])]

        # initial predator positions
        self.angs = np.linspace(0, 2*math.pi, self.n_preds, endpoint=False)

        self.pred_init = pred_init
        if self.pred_init == 'circle':
            # init around circle

            # example 1
            self.circle_pts = [world.origin + (np.array([math.cos(ang), math.sin(ang)])*3.0) for ang in self.angs]
            self.circle_pts.append(world.origin)

            # example 2
            # self.circle_pts = [world.origin + (np.array([math.cos(ang), math.sin(ang)])*6.0) for ang in self.angs]
            # self.circle_pts.append(world.origin)

            # example 3
            # new_origin = world.origin + np.array([4.0, 4.0])
            # self.circle_pts = [new_origin + (np.array([math.cos(ang), math.sin(ang)])*3.5) for ang in self.angs]
            # self.circle_pts.append(new_origin)

            # example 4 -- radius 5.0 in top left
            # new_origin = world.origin + np.array([-2.0, 3.0])
            # self.circle_pts = [new_origin + (np.array([math.cos(ang), math.sin(ang)])*5.0) for ang in self.angs]
            # self.circle_pts.append(new_origin)

            
        # gaussian noise
        self.noise = noise

        # discrete actions
        world.discrete_actions = discrete

        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        if self.pred_init == 'circle':
            temp_pts = copy.deepcopy(self.circle_pts)
            # add some noise to prey init (not exactly in middle)
            # temp_pts.append(np.random.normal(world.origin[0], 0.15, size=2))
        elif self.pred_init == 'rand-circ':
            # self.angs += np.random.uniform(0, math.pi)
            new_origin = world.origin + np.random.uniform(-4.0, 4.0, size=2)
            new_radius = np.random.uniform(3.0, 6.0)
            # print('origin = {}, radius = {}'.format(new_origin, new_radius))
            temp_pts = [new_origin + (np.array([math.cos(ang), math.sin(ang)])*new_radius) for ang in self.angs]
            temp_pts.append(new_origin)
            # temp_pts.append(np.random.normal(new_origin[0], 0.15, size=2))

        
        # properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
        # roperties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])


        # set initial states
        for i, agent in enumerate(world.agents):
            if self.baseline:
                if self.pred_init == 'circle' or self.pred_init == 'rand-circ':
                    # init around circle
                    agent.state.real_pos = temp_pts[i]
                else:
                    # sample position from Gaussian centered at origin
                    agent.state.real_pos = np.random.normal(world.size/2, 0.75, world.dim_p)
            else:
                if self.pred_init == 'circle' or self.pred_init == 'rand-circ':
                    # init around circle
                    agent.state.real_pos = temp_pts[i]
                else:
                    # random over whole world
                    agent.state.real_pos = np.random.uniform(0, world.size, world.dim_p)

            # agents can move beyond confines of camera image --> need to adjust coords accordingly
            agent.state.p_pos = agent.state.real_pos % world.size
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            if self.baseline:
                if not landmark.boundary:
                    landmark.state.p_pos = self.landmark_pos[i]
                    landmark.state.p_vel = np.zeros(world.dim_p)
            else:
                if not landmark.boundary:
                    landmark.state.p_pos = np.random.uniform(0.1, world.size-0.1, world.dim_p)
                    landmark.state.p_vel = np.zeros(world.dim_p)



    def hit_boundary(self, x, size):
        if x > size - 0.1:
            return True
        else:
            return False

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions, 0
        else:
            # check if prey hit boundary
            for p in range(world.dim_p):
                x = abs(agent.state.p_pos[p])
                if self.hit_boundary(x, world.size):
                    return 0, 1
                
            return 0, 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        # for p in range(world.dim_p):
        #     x = abs(agent.state.p_pos[p])
        #     rew -= self.bound(x, self.size)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos)

        # pred/prey observations
        pred_p_pos, pred_r_pos, pred_vel, pred_comm = [], [], [], []
        prey_p_pos, prey_r_pos, prey_vel, prey_comm = [], [], [], []
        for other in world.agents:
            if other is agent: continue
            if other.adversary:
                pred_p_pos.append(other.state.p_pos)
                pred_r_pos.append(other.state.real_pos)
                pred_vel.append(other.state.p_vel)
                pred_comm.append(other.state.c)
            else:
                prey_p_pos.append(other.state.p_pos)
                prey_r_pos.append(other.state.real_pos)
                prey_vel.append(other.state.p_vel) 
                prey_comm.append(other.state.c)

        obs = {
            'pos' : agent.state.p_pos,
            'r_pos' : agent.state.real_pos,
            'vel' : agent.state.p_vel,
            'entity_pos': entity_pos,
            'pred_pos' : pred_p_pos,
            'pred_coords' : pred_r_pos,
            'pred_vel' :pred_vel,
            'pred_comm': pred_comm,
            'prey_pos' : prey_p_pos,
            'prey_coords' : prey_r_pos,
            'prey_vel' : prey_vel,
            'prey_comm': prey_comm
        }

        if self.noise:
            obs = self.add_noise(obs)

        return obs


    def add_noise(self, obs):
        # add noise to positions
        obs['pos'] += np.random.normal(0, 0.05, 2)
        obs['pred_pos'] = [p + np.random.normal(0, 0.1, 2) for p in obs['pred_pos']]
        obs['prey_pos'] = [p + np.random.normal(0, 0.1, 2) for p in obs['prey_pos']]

        return obs

