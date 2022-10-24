import numpy as np
import math
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario
from multiagent.utils import overlaps, toroidal_distance
from scipy.spatial import distance
from scipy.special import softmax

class Scenario(BaseScenario):
    def make_world(self, config):
        world = World()
        # set any world properties
        world.n_steps = 500
        world.torus = True
        world.dim_c = 2
        world.size = config.world_size
        world.origin = np.array([world.size/2, world.size/2])
        world.use_sensor_range = False
        world.symmetric = config.symmetric
        world.partial = False
        world.shape = config.rew_shape

        print('world size = {}'.format(world.size))
        print('num keepers = {}'.format(config.n_keepers))
        print('num seekers = {}'.format(config.n_seekers))
        print('keeper vel = {}'.format(config.prey_vel))
        print('seeker vel = {}'.format(config.pred_vel))

        # self.n_seekers = num_seekers = config.n_seekers
        self.n_seekers = num_seekers = 1
        self.n_keepers = num_keepers = config.n_keepers
        num_agents = num_seekers + num_keepers
        

        # add agents
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent {}'.format(i)
            agent.id = i
            agent.active = True
            agent.captured = False
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_seekers else False
            agent.size = 0.1 if agent.adversary else 0.1
            agent.accel = 20.0
            if agent.adversary:
                if isinstance(config.pred_vel, list):
                    agent.max_speed = config.pred_vel[i]
                else:
                    agent.max_speed = config.pred_vel 
            else:
                agent.max_speed = config.prey_vel

        # add balls
        num_balls = 1
        world.landmarks = [Landmark() for _ in range(num_balls)]
        for i, ball in enumerate(world.landmarks):
            ball.movable = True
            ball.size = 0.075
            ball.initial_mass = 10.0
            ball.accel = 1.0
            ball.max_speed = 0.5
            # ball.max_speed = 0.25

        # discrete actions
        world.discrete_actions = config.discrete

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.origin = np.array([world.size/2, world.size/2])

        # agent colors
        for i, agent in enumerate(world.agents):
            if agent.adversary:
                # agent.color = world.predator_colors[i]
                agent.color = np.array([0.85, 0.35, 0.35])
            else:
                agent.color = np.array([0.35, 0.85, 0.35]) 

        # properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # init seekers in center, keepers randomly, and ball near keeper
        redraw = True
        while redraw:
            # draw location for seekers
            # seeker_pts = [world.origin + np.random.normal(0.0, 0.0001, size=2) for _ in range(self.n_seekers)]
            seeker_pt = world.origin + np.random.normal(0.0, 0.0001, size=2)

            # draw predator locations
            keeper_pts = [np.random.uniform(0.0, world.size, size=2) for _ in range(self.n_keepers)]

            # ensure keepers are initialized a reasonable distance away from seekers
            redraw = overlaps(seeker_pt, keeper_pts, world.size, threshold=0.5)

        # init ball near one of the keepers
        starting_dists = [toroidal_distance(k, seeker_pt, world.size) for k in keeper_pts]
        starting_keeper_idx = np.argmax(starting_dists)
        starting_keeper_pt = keeper_pts[starting_keeper_idx]
        ball_pt = starting_keeper_pt + np.random.normal(0.15, 0.2, size=2)

        # set initial states
        init_pts = [seeker_pt] + keeper_pts
        for i, agent in enumerate(world.agents):
            agent.active = True
            agent.captured = False

            # agents can move beyond confines of camera image --> need to adjust coords accordingly
            agent.state.coords = init_pts[i]
            agent.state.p_pos = agent.state.coords % world.size
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.theta = 0.0
            agent.state.c = np.zeros(world.dim_c)

        for i, ball in enumerate(world.landmarks):
            ball.captured = False
            ball.state.coords = ball_pt
            ball.state.p_pos = ball.state.coords % world.size
            ball.state.p_vel = np.zeros(world.dim_p)
            ball.state.theta = 0.0

    def benchmark_data(self, agent, world):
        return { 'active' : agent.active }

    def is_collision(self, obj1, obj2):
        if obj1 == obj2:
            return False
        delta_pos = obj1.state.p_pos - obj2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = obj1.size + obj2.size
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

    def reward(self, agent, world, action):
        main_reward = self.seeker_reward(agent, world) if agent.adversary else self.keeper_reward(agent, world)
        return main_reward

    def keeper_reward(self, agent, world):
        if agent.active:
            # Keepers are negatively rewarded if ball is caught by seekers
            rew = 0.01
            # rew = 0.0
            shape = False
            adversaries = self.active_adversaries(world)
            ball = world.landmarks[0]
            for adv in adversaries:
                if self.is_collision(ball, adv):
                    ball.captured = True 
                    rew -= 5
                    break
            # return rew / 50.
            return rew
        else:
            return 0.0

    def seeker_reward(self, agent, world):
        # Seekers are rewarded for collisions with balls
        # rew = -0.1
        rew = 0.0
        adversaries = self.active_adversaries(world)
        ball = world.landmarks[0]
        if world.shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            if world.torus:
                # shaped by toroidal distance of closest predator
                rew -= 0.1 * toroidal_distance(agent.state.p_pos, ball.state.p_pos, world.size)
            else:
                # shaped by euclidean distance of closest predator
                rew -= 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - ball.state.p_pos)))

        for adv in adversaries:
            if self.is_collision(ball, adv):
                ball.captured = True 

        rew += 5 * int(ball.captured)
        # return rew / 50.
        return rew

    def terminal(self, agent, world):
        ball = world.landmarks[0]
        return ball.captured

    def observation(self, agent, world):
        if agent.adversary:
            return self.seeker_observation(agent, world)
        else:
            return self.keeper_observation(agent, world)

    def seeker_observation(self, agent, world):
        # print('\nagent {} = {}'.format(agent.id, agent.adversary))
        # print('agent pos = {}'.format(agent.state.p_pos))
        other_pos = []

        for ball in world.landmarks:
            # print('ball pos {}'.format(ball.state.p_pos))
            other_pos.append(ball.state.p_pos)

        for other in world.agents:
            if other is agent: continue

            # full observations
            # print('other pos = {}'.format(other.state.p_pos))
            other_pos.append(other.state.p_pos)

        # if world.symmetric and agent.adversary:
        #     other_pos = self.symmetrize(agent.id, other_pos)

        obs = np.concatenate([agent.state.p_pos] + other_pos)
        return obs

    def keeper_observation(self, agent, world):
        # print('\n agent {} = {}'.format(agent.id, agent.adversary))
        # print('agent pos = {}'.format(agent.state.p_pos))
        other_pos = []

        for ball in world.landmarks:
            # print('ball pos {}'.format(ball.state.p_pos))
            other_pos.append(ball.state.p_pos)

        for other in world.agents:
            if other is agent: continue

            # full observations
            # if other.adversary:
            #     print('seeker pos = {}'.format(other.state.p_pos))
            # else:
            #     print('keeper pos = {}'.format(other.state.p_pos))
            other_pos.append(other.state.p_pos)

        # if world.symmetric and agent.adversary:
        #     other_pos = self.symmetrize(agent.id, other_pos)

        obs = np.concatenate([agent.state.p_pos] + other_pos)
        # print('obs = {}\n'.format(obs))
        return obs

    def symmetrize(self, agent_id, arr):
        # ensure symmetry in obervation space
        # P1 --> P2, P3
        # P2 --> P3, P1
        # P3 --> P1, P2
        if agent_id == 0 or agent_id == 2:
            return arr
        else:
            return [arr[1], arr[0], arr[2]]
        

