import json
import os
import random

from .state import State


class Q_State(State):
    '''Augments the game state with Q-learning information'''

    def __init__(self, string):
        super().__init__(string)

        # key stores the state's key string (see notes in _compute_key())
        self.key = self._compute_key()

    def _compute_key(self):
        '''
        Returns a key used to index this state.

        The key should reduce the entire game state to something much smaller
        that can be used for learning. When implementing a Q table as a
        dictionary, this key is used for accessing the Q values for this
        state within the dictionary.
        '''

        # this simple key uses the 3 object characters above the frog
        # and combines them into a key string
        return ''.join([
            self.get(self.frog_x - 1, self.frog_y - 1) or '_',
            self.get(self.frog_x, self.frog_y - 1) or '_',
            self.get(self.frog_x + 1, self.frog_y - 1) or '_',
        ])

    def reward(self):
        '''Returns a reward value for the state.'''

        if self.at_goal:
            return self.score
        elif self.is_done:
            return -10
        else:
            return 0


class Agent:

    def __init__(self, train=None):

        # train is either a string denoting the name of the saved
        # Q-table file, or None if running without training
        self.train = train

        # q is the dictionary representing the Q-table
        self.q = {}

        # name is the Q-table filename
        # (you likely don't need to use or change this)
        self.name = train or 'q'

        # path is the path to the Q-table file
        # (you likely don't need to use or change this)
        self.path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'train', self.name + '.json')

        self.load()
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.prev_state = None
        self.prev_action = None

    def load(self):
        '''Loads the Q-table from the JSON file'''
        try:
            with open(self.path, 'r') as f:
                self.q = json.load(f)
            if self.train:
                print('Training {}'.format(self.path))
            else:
                print('Loaded {}'.format(self.path))
        except IOError:
            if self.train:
                print('Training {}'.format(self.path))
            else:
                raise Exception('File does not exist: {}'.format(self.path))
        return self

    def save(self):
        '''Saves the Q-table to the JSON file'''
        with open(self.path, 'w') as f:
            json.dump(self.q, f)
        return self


    def choose_action(self, state_string):
        state = Q_State(state_string)

        if self.train and random.random() < self.epsilon:
            action = random.choice(State.ACTIONS)
        else:
            q_values = {}
            for action in State.ACTIONS:
                q_values[action] = self.get_q_value(state, action)
            max_q = max(q_values.values())
            actions_with_max_q = []
            for action, q in q_values.items():
                if q == max_q:
                    actions_with_max_q.append(action)
            action = random.choice(actions_with_max_q)

        if self.train and self.prev_state is not None and self.prev_action is not None:
            reward = state.reward()
            prev_q_value = self.get_q_value(self.prev_state, self.prev_action)
            future_q_values = []
            for a in State.ACTIONS:
                future_q_values.append(self.get_q_value(state, a))
            max_future_q_value = max(future_q_values)
            new_q_value = (1 - self.alpha) * prev_q_value + self.alpha * (reward + self.gamma * max_future_q_value)
            self.set_q_value(self.prev_state, self.prev_action, new_q_value)
            self.save()

        self.prev_state = state
        self.prev_action = action

        return action

    def get_q_value(self, state, action):
        if state.key in self.q:
            if action in self.q[state.key]:
                return self.q[state.key][action]
        return 0.0

    def set_q_value(self, state, action, value):
        if state.key not in self.q:
            self.q[state.key] = {}
        self.q[state.key][action] = value
