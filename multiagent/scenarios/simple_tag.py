import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, bounded, n_preds, pred_vel, prey_vel, baseline, noise=False, discrete=True, tiny=False):
        world = World()
        # set any world properties
        world.dim_c = 2
        num_good_agents = 1
        num_adversaries = n_preds
        # num_adversaries = 1
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 2

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

            if tiny:
                agent.size /= 5
                agent.max_speed /= 2


        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False

            if tiny:
                landmark.size /= 2

        # handle boundaries
        if bounded:
            world.bounded = True

            # add border walls
            left_wall = Wall(orient='V', axis_pos=2.25, endpoints=(-0.75, 2.75), width=0.5)
            right_wall = Wall(orient='V', axis_pos=-0.25, endpoints=(-0.75, 2.75), width=0.5)
            top_wall = Wall(axis_pos=2.25, endpoints=(-0.75, 2.75), width=0.5)
            bot_wall = Wall(axis_pos=-0.25, endpoints=(-0.75, 2.75), width=0.5)
            world.walls = [left_wall, top_wall, bot_wall, right_wall]
        else:
            world.bounded = False
            
        # baseline setup: always generate landmarks in same location
        self.baseline = baseline
        if self.baseline:
            # self.landmark_pos = [np.random.uniform(0.1, 1.9, world.dim_p) for i in range(num_landmarks)]
            self.landmark_pos = [np.array([0.4, 1.2]), np.array([0.9, 0.45])]

        # gaussian noise
        self.noise = noise

        # discrete actions
        world.discrete_actions = discrete

        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # agent.color = np.array([0, 0, 0]) if not agent.adversary else np.array([0.5, 0.1, 0.1])

            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(0, 2, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if self.baseline:
                if not landmark.boundary:
                    landmark.state.p_pos = self.landmark_pos[i]
                    landmark.state.p_vel = np.zeros(world.dim_p)
            else:
                if not landmark.boundary:
                    # landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                    landmark.state.p_pos = np.random.uniform(0.1, 1.9, world.dim_p)
                    landmark.state.p_vel = np.zeros(world.dim_p)

            # if not landmark.boundary:
            #     # landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            #     landmark.state.p_pos = np.random.uniform(0.1, 1.9, world.dim_p)
            #     landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


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
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

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
        pred_pos, pred_vel, pred_comm = [], [], []
        prey_pos, prey_vel, prey_comm = [], [], []
        for other in world.agents:
            if other is agent: continue
            if other.adversary:
                pred_pos.append(other.state.p_pos)
                pred_vel.append(other.state.p_vel)
                pred_comm.append(other.state.c)
            else:
                prey_pos.append(other.state.p_pos)
                prey_vel.append(other.state.p_vel) 
                prey_comm.append(other.state.c)

        # TODO: try adding predator's velocity to communicated position
        obs = {
            'pos' : agent.state.p_pos,
            'vel' : agent.state.p_vel,
            'entity_pos': entity_pos,
            'pred_pos' : pred_pos,
            'pred_vel' :pred_vel,
            'pred_comm': pred_comm,
            'prey_pos' : prey_pos,
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

