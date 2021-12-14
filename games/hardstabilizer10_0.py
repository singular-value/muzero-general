import datetime
import os

import stim
import numpy
np = numpy
import torch

from .abstract_game import AbstractGame

n = 10
stacked_observations = 0

class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available


        ### Game
        self.observation_shape = (1, 1, 2 * n)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(2))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = stacked_observations  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class


        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 500  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Number of future moves self-simulated
        self.discount = 0.978  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 2  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 2  # Number of channels in policy head
        self.resnet_fc_reward_layers = []  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = []  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = []  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 5
        self.fc_representation_layers = [16]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 3000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 32  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.0064  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000



        ### Replay Buffer
        self.replay_buffer_size = 5000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 7  # Number of game moves to keep for every batch element
        self.td_steps = 7  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0.2  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = StabilizerEnv()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return [[observation]], reward, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.        

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(2))

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return [[self.env.reset()]]

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "Observable measurement assigned to +1 eigenvalue",
            1: "Observable measurement assigned to -1 eigenvalue",
        }
        return f"{action_number}. {actions[action_number]}"


class StabilizerEnv:
    def __init__(self, n=n):
        self.round = 0
        self.n = n
        self.state = stim.TableauSimulator()
        self.current_observable = None

    def legal_actions(self):
        return [0, 1]  # corresponds to +1 and -1 measurement outcomes respectively

    def step(self, action):
        """Returns tuple: observation, reward, whether game has ended."""
        assert action in self.legal_actions()
        self.round += 1

        if not self.is_consistent_with_qm(action):
            return self.get_observation(), 0, True

        self.update_state(action)
        return self.get_observation(), 1, False

    def is_consistent_with_qm(self, action):
        """Returns True if the action is consistent with quantum mechanics.
        I.e. checks that the player's guessed measurement outcome (action) is plausible given the
        current quantum state and the current observable being measured.
        """

        expectation = self.state.peek_observable_expectation(self.current_observable)

        if np.isclose(expectation, 0):
            return True
        elif np.isclose(expectation, +1):
            return action == 0
        elif np.isclose(expectation, -1):
            return action == 1
        assert False

    def update_state(self, action):

        self.state.h(self.n)
        for i, p in enumerate(self.current_observable):
            if p == 1:
                self.state.cnot(self.n, i)
            elif p == 2:
                self.state.cy(self.n, i)
            elif p == 3:
                self.state.cz(self.n, i)
        self.state.h(self.n)

        while True:  # janky post-select based off of github.com/quantumlib/Stim/blob/main/doc/
                     # python_api_reference_vDev.md#stimtableausimulatorcopyself---stimtableausimulator
            copy = self.state.copy()
            outcome = copy.measure(self.n)
            if outcome == action:
                self.state = copy
                if outcome == 1:
                    self.state.x(self.n)
                break

    def reset(self):
        self.state = stim.TableauSimulator()
        return self.get_observation()

    def render(self):
        print(self.current_observable)

    def get_observation(self):
        """Play 2n normal rounds (with randomly selected stabilizer), then pick from canonical stabilizers."""
        if self.round >= 2 * self.n:
            state = self.state.copy()
            state.set_num_qubits(self.n)  # truncate the ancilla
            self.current_observable = state.canonical_stabilizers()[np.random.randint(self.n)]
            self.current_observable /= self.current_observable.sign  # discard the sign, we don't care
            #print(state.canonical_stabilizers(), "\nchose: ", self.current_observable)

            observation = []
            char_to_obs = {0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1]}
            for char in self.current_observable:
                observation.extend(char_to_obs[char])
            return np.array(observation)

        observation = np.random.choice([0, 1], size=(2 * self.n))
        paulis = []

        for i, (x, z) in enumerate(zip(observation[::2], observation[1::2])):
            if x and z:
                paulis.append("Y")
            elif x:
                paulis.append("X")
            elif z:
                paulis.append("Z")
            else:
                paulis.append("I")

        self.current_observable = stim.PauliString("".join(paulis))
        return observation
