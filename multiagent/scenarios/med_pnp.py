import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, config, discrete=True):
        world = World()

        # set world properties first
        world.torus = False
        world.n_steps = 500
        world.dim_c = 2
        world.size = 10.0
        world.level = 0 if config.use_curriculum else 4
        # world.level = 4

        # agent properties
        num_good_agents = 1
        num_adversaries = 3
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 15

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.active = True
            agent.captured = False
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.active = True
            landmark.collide = True
            landmark.movable = False
            landmark.size = np.random.uniform(0.35, 0.6)
            landmark.boundary = False

        # choose landmarks for first epoch    
        # n_marks = np.random.randint(2, len(world.landmarks))
        # world.curr_landmarks = np.random.choice(world.landmarks, n_marks).tolist()

        # add border walls
        left_wall = Wall(orient='V', axis_pos=world.size/2 + 0.5, endpoints=(-world.size/2 - 0.75, world.size/2 + 0.75), width=0.75)
        right_wall = Wall(orient='V', axis_pos=-world.size/2 - 0.5, endpoints=(-world.size/2 - 0.75, world.size/2 + 0.75), width=0.75)
        top_wall = Wall(axis_pos=world.size/2 + 0.5, endpoints=(-world.size/2 - 0.75, world.size/2 + 0.75), width=0.75)
        bot_wall = Wall(axis_pos=-world.size/2 - 0.5, endpoints=(-world.size/2 - 0.75, world.size/2 + 0.75), width=0.75)
        world.walls = [left_wall, top_wall, bot_wall, right_wall]

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

        if world.level == 0:   
            pred_bound = 3.5
            prey_bound = 4.5
        elif world.level == 1:        
            pred_bound = 3.0
            prey_bound = 4.0
        elif world.level == 2:        
            pred_bound = 2.5
            prey_bound = 3.0
        elif world.level == 3:        
            pred_bound = 1.25
            prey_bound = 2.0
        else:
            pred_bound = 0.05
            prey_bound = 1.0

        pred_init_pts = [np.array([-world.size/2 + pred_bound, -world.size/2 + pred_bound]),
                        np.array([-world.size/2 + pred_bound, world.size/2 - pred_bound]),
                        np.array([world.size/2 - pred_bound, -world.size/2 + pred_bound]),
                        np.array([world.size/2 - pred_bound, world.size/2 - pred_bound]),
                        np.array([-world.size/2 + pred_bound, 0])]

        # set random initial states
        for i, agent in enumerate(world.agents):
            agent.active = True
            agent.captured = False
            if agent.adversary:
                agent.state.p_pos = pred_init_pts[i]
            else:
                agent.state.p_pos = np.random.uniform(-world.size/2 + prey_bound, world.size/2 - prey_bound, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        landmark_init_pts = [
            np.array([4.054015, -4.51139905]),
            np.array([-4.10370804, 0.36195873]),
            np.array([1.95984979, 1.39413427]),
            np.array([-3.3288319, 3.62890966]),
            np.array([-1.54795894, -4.4763568]),
            np.array([-1.55855363, -3.14317905]),
            np.array([-1.75482672, -2.82588394]),
            np.array([-4.34057309, -3.13723603]),
            np.array([ 4.64193824, -2.87771924]),
            np.array([ 0.61719114, -1.10018694]),
            np.array([0.47546935, 4.3833841]),
            np.array([ 0.29096532, -2.66313494]),
            np.array([0.54558641, 0.27552597]),
            np.array([-3.23978418, -3.67931561]),
            np.array([3.7488089, 1.83948606])
        ]

        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = landmark_init_pts[i]
                # landmark.state.p_pos = np.random.uniform(-world.size/2 + 0.35, world.size/2 - 0.35, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)



    def benchmark_data(self, agent, world):
        return agent.active


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

    def active_good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary and agent.active]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

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
                    rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
            if agent.collide:
                for a in adversaries:
                    if self.is_collision(a, agent):
                        agent.captured = True 
                        rew -= 50
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
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        entity_sizes = []
        # for entity in world.curr_landmarks: # TODO: FOR DYNAMIC LANDMARK GENERATION
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos)
                # entity_sizes.append(np.array([entity.size]))
                entity_sizes.append(entity.size)

        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            if other.captured:
                other_pos.append(np.array([-10.0, -10.0]))
            else:
                other_pos.append(other.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)

        obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + [entity_sizes])
        return obs

    '''
    Predator Observations:
        agent velocity =        [0, 1]
        agent position =        [2, 3]
        landmark positions =    [4, 4+2L]
        predator positions =    [4+2L+1, (4+2L+1) + 2(P-1)] 
        prey positions =        [(4+2L+1) + 2(P-1) + 1, (4+2L+1) + 2(P-1) + 1 + 2E]
        prey velocities =       [(4+2L+1) + 2(P-1) + 1 + 2E + 1, (4+2L+1) + 2(P-1) + 1 + 2E + 1 + 2E]

    Prey Observations:
        agent velocity =        [0, 1]
        agent position =        [2, 3]
        landmark positions =    [4, 4+2L]
        predator positions =    [4+2L+1, (4+2L+1) + 2P] 
        prey positions =        [(4+2L+1) + 2P + 1, (4+2L+1) + 2P + 1 + 2(E-1)]
        prey velocities =       [(4+2L+1) + 2P + 1 + 2(E-1) + 1, (4+2L+1) + 2P + 1 + 2(E-1) + 1 + 2(E-1)]

    '''