"""
Reinforcement Learning Models for Nasdaq-100 E-mini futures trading bot.
This module implements various reinforcement learning models for trading.
"""

import pandas as pd
import numpy as np
import logging
import os
import joblib
import matplotlib.pyplot as plt
from collections import deque
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Try to import TensorFlow and Keras with error handling for environments with limited resources
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model, save_model
    from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow/Keras not available. Reinforcement learning models will not function.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingEnvironment:
    """
    Trading environment for reinforcement learning.
    
    This class simulates a trading environment for the agent to interact with.
    """
    
    def __init__(self, data, initial_balance=100000.0, transaction_cost=0.0001, 
                 max_position=1, reward_function='pnl', state_features=None):
        """
        Initialize the trading environment.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data with OHLCV columns.
        initial_balance : float, optional
            Initial account balance.
        transaction_cost : float, optional
            Transaction cost as a fraction of trade value.
        max_position : int, optional
            Maximum position size (number of contracts).
        reward_function : str, optional
            Function to calculate rewards ('pnl', 'sharpe', 'sortino').
        state_features : list, optional
            List of feature names to include in the state.
        """
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.reward_function = reward_function
        
        # Set state features if not provided
        if state_features is None:
            self.state_features = [col for col in data.columns if col not in ['timestamp', 'date', 'time']]
        else:
            self.state_features = state_features
        
        # Initialize state
        self.reset()
        
        logger.info(f"Initialized TradingEnvironment with {len(data)} data points")
    
    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
        --------
        numpy.ndarray
            Initial state.
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        self.portfolio_values = [self.initial_balance]
        self.returns = []
        
        # Get initial state
        state = self._get_state()
        
        logger.info("Reset trading environment")
        return state
    
    def _get_state(self):
        """
        Get the current state of the environment.
        
        Returns:
        --------
        numpy.ndarray
            Current state.
        """
        # Get current data point
        current_data = self.data.iloc[self.current_step]
        
        # Extract features
        features = current_data[self.state_features].values
        
        # Add position information
        state = np.append(features, [self.position / self.max_position])
        
        return state
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Parameters:
        -----------
        action : int
            Action to take (0: hold, 1: buy, 2: sell).
        
        Returns:
        --------
        tuple
            (next_state, reward, done, info)
        """
        # Get current data
        current_data = self.data.iloc[self.current_step]
        current_price = current_data['close']
        
        # Execute action
        reward, info = self._execute_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Get next state
        if not done:
            next_state = self._get_state()
        else:
            next_state = None
        
        return next_state, reward, done, info
    
    def _execute_action(self, action, current_price):
        """
        Execute the given action.
        
        Parameters:
        -----------
        action : int
            Action to take (0: hold, 1: buy, 2: sell).
        current_price : float
            Current price.
        
        Returns:
        --------
        tuple
            (reward, info)
        """
        # Initialize info
        info = {
            'step': self.current_step,
            'price': current_price,
            'action': action,
            'position_before': self.position,
            'balance_before': self.balance,
            'trade_amount': 0,
            'transaction_cost': 0
        }
        
        # Execute action
        if action == 1:  # Buy
            # Calculate how many contracts to buy
            contracts_to_buy = self.max_position - self.position
            
            if contracts_to_buy > 0:
                # Calculate trade value
                trade_value = contracts_to_buy * current_price
                
                # Calculate transaction cost
                transaction_cost = trade_value * self.transaction_cost
                
                # Update balance and position
                self.balance -= trade_value + transaction_cost
                self.position += contracts_to_buy
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'type': 'buy',
                    'price': current_price,
                    'contracts': contracts_to_buy,
                    'value': trade_value,
                    'cost': transaction_cost
                })
                
                # Update info
                info['trade_amount'] = trade_value
                info['transaction_cost'] = transaction_cost
        
        elif action == 2:  # Sell
            # Calculate how many contracts to sell
            contracts_to_sell = self.position + self.max_position
            
            if contracts_to_sell > 0:
                # Calculate trade value
                trade_value = contracts_to_sell * current_price
                
                # Calculate transaction cost
                transaction_cost = trade_value * self.transaction_cost
                
                # Update balance and position
                self.balance += trade_value - transaction_cost
                self.position -= contracts_to_sell
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'type': 'sell',
                    'price': current_price,
                    'contracts': contracts_to_sell,
                    'value': trade_value,
                    'cost': transaction_cost
                })
                
                # Update info
                info['trade_amount'] = trade_value
                info['transaction_cost'] = transaction_cost
        
        # Calculate portfolio value
        portfolio_value = self.balance + (self.position * current_price)
        self.portfolio_values.append(portfolio_value)
        
        # Calculate return
        if len(self.portfolio_values) > 1:
            ret = (portfolio_value / self.portfolio_values[-2]) - 1
            self.returns.append(ret)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Update info
        info['position_after'] = self.position
        info['balance_after'] = self.balance
        info['portfolio_value'] = portfolio_value
        info['reward'] = reward
        
        return reward, info
    
    def _calculate_reward(self):
        """
        Calculate the reward based on the specified reward function.
        
        Returns:
        --------
        float
            Reward value.
        """
        if len(self.portfolio_values) < 2:
            return 0
        
        if self.reward_function == 'pnl':
            # Use change in portfolio value as reward
            return self.portfolio_values[-1] - self.portfolio_values[-2]
        
        elif self.reward_function == 'sharpe':
            # Use Sharpe ratio as reward
            if len(self.returns) < 2:
                return 0
            
            mean_return = np.mean(self.returns)
            std_return = np.std(self.returns) + 1e-6  # Add small value to avoid division by zero
            
            return mean_return / std_return
        
        elif self.reward_function == 'sortino':
            # Use Sortino ratio as reward
            if len(self.returns) < 2:
                return 0
            
            mean_return = np.mean(self.returns)
            
            # Calculate downside deviation (standard deviation of negative returns)
            negative_returns = [r for r in self.returns if r < 0]
            
            if not negative_returns:
                downside_deviation = 1e-6  # Small value to avoid division by zero
            else:
                downside_deviation = np.std(negative_returns) + 1e-6
            
            return mean_return / downside_deviation
        
        else:
            raise ValueError(f"Unknown reward function: {self.reward_function}")
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Parameters:
        -----------
        mode : str, optional
            Rendering mode ('human' or 'rgb_array').
        
        Returns:
        --------
        None or numpy.ndarray
            If mode is 'rgb_array', returns the rendered image.
        """
        if mode == 'human':
            # Print current state
            current_data = self.data.iloc[self.current_step]
            print(f"Step: {self.current_step}")
            print(f"Date: {current_data.get('timestamp', current_data.name)}")
            print(f"Price: {current_data['close']}")
            print(f"Position: {self.position}")
            print(f"Balance: {self.balance}")
            print(f"Portfolio Value: {self.portfolio_values[-1]}")
            print("---")
        
        elif mode == 'rgb_array':
            # Create a plot of portfolio value
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.portfolio_values)
            ax.set_title('Portfolio Value')
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')
            
            # Convert plot to numpy array
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            
            return img
    
    def get_performance_metrics(self):
        """
        Calculate performance metrics for the trading session.
        
        Returns:
        --------
        dict
            Dictionary of performance metrics.
        """
        # Calculate returns
        returns = np.array(self.returns)
        
        # Calculate metrics
        total_return = (self.portfolio_values[-1] / self.initial_balance) - 1
        
        if len(returns) > 0:
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6)
            
            # Calculate downside deviation (standard deviation of negative returns)
            negative_returns = returns[returns < 0]
            
            if len(negative_returns) > 0:
                sortino_ratio = np.mean(returns) / (np.std(negative_returns) + 1e-6)
            else:
                sortino_ratio = np.inf
            
            max_drawdown = 0
            peak = self.portfolio_values[0]
            
            for value in self.portfolio_values:
                if value > peak:
                    peak = value
                
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            max_drawdown = 0
        
        # Count trades
        num_trades = len(self.trades)
        
        # Calculate win rate
        if num_trades > 0:
            profitable_trades = 0
            
            for i in range(1, len(self.trades)):
                if self.trades[i]['type'] != self.trades[i-1]['type']:
                    # This is a closing trade
                    if self.trades[i-1]['type'] == 'buy' and self.trades[i]['price'] > self.trades[i-1]['price']:
                        profitable_trades += 1
                    elif self.trades[i-1]['type'] == 'sell' and self.trades[i]['price'] < self.trades[i-1]['price']:
                        profitable_trades += 1
            
            win_rate = profitable_trades / max(1, num_trades - 1)  # Avoid division by zero
        else:
            win_rate = 0
        
        # Calculate total transaction costs
        total_cost = sum(trade['cost'] for trade in self.trades)
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'total_cost': total_cost,
            'final_balance': self.balance,
            'final_position': self.position,
            'final_portfolio_value': self.portfolio_values[-1]
        }
        
        return metrics


class ReplayBuffer:
    """
    Replay buffer for reinforcement learning.
    
    This class stores and samples experiences for training.
    """
    
    def __init__(self, capacity=10000):
        """
        Initialize the replay buffer.
        
        Parameters:
        -----------
        capacity : int, optional
            Maximum capacity of the buffer.
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer.
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : numpy.ndarray
            Next state.
        done : bool
            Whether the episode is done.
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.
        
        Parameters:
        -----------
        batch_size : int
            Number of experiences to sample.
        
        Returns:
        --------
        tuple
            (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network (DQN) agent for reinforcement learning.
    """
    
    def __init__(self, state_size, action_size, hidden_layers=[64, 32], learning_rate=0.001, 
                 gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, 
                 batch_size=32, update_target_freq=100):
        """
        Initialize the DQN agent.
        
        Parameters:
        -----------
        state_size : int
            Size of the state space.
        action_size : int
            Size of the action space.
        hidden_layers : list, optional
            Number of units in each hidden layer.
        learning_rate : float, optional
            Learning rate for the optimizer.
        gamma : float, optional
            Discount factor for future rewards.
        epsilon : float, optional
            Exploration rate.
        epsilon_decay : float, optional
            Decay rate for exploration.
        epsilon_min : float, optional
            Minimum exploration rate.
        batch_size : int, optional
            Batch size for training.
        update_target_freq : int, optional
            Frequency of target network updates.
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras is not available. Cannot create DQN agent.")
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        
        # Create replay buffer
        self.memory = ReplayBuffer()
        
        # Create models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Initialize step counter
        self.step_counter = 0
        
        logger.info(f"Initialized DQN agent with state_size={state_size}, action_size={action_size}")
    
    def _build_model(self):
        """
        Build the neural network model.
        
        Returns:
        --------
        tensorflow.keras.models.Model
            Neural network model.
        """
        model = Sequential()
        
        # Add input layer
        model.add(Dense(self.hidden_layers[0], activation='relu', input_dim=self.state_size))
        model.add(BatchNormalization())
        
        # Add hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
        
        # Add output layer
        model.add(Dense(self.action_size, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_model(self):
        """Update the target model with weights from the main model."""
        self.target_model.set_weights(self.model.get_weights())
        logger.info("Updated target model weights")
    
    def act(self, state, training=True):
        """
        Choose an action based on the current state.
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state.
        training : bool, optional
            Whether the agent is in training mode.
        
        Returns:
        --------
        int
            Chosen action.
        """
        if training and np.random.rand() < self.epsilon:
            # Exploration: choose a random action
            return np.random.randint(self.action_size)
        else:
            # Exploitation: choose the best action
            state = np.reshape(state, [1, self.state_size])
            q_values = self.model.predict(state, verbose=0)[0]
            return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience in memory.
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : numpy.ndarray
            Next state.
        done : bool
            Whether the episode is done.
        """
        self.memory.add(state, action, reward, next_state, done)
    
    def replay(self):
        """
        Train the model using experiences from memory.
        
        Returns:
        --------
        float
            Loss value.
        """
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Calculate target Q values
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Update target model if needed
        self.step_counter += 1
        if self.step_counter % self.update_target_freq == 0:
            self.update_target_model()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def save(self, filepath):
        """
        Save the agent to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the agent.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save models
        self.model.save(f"{filepath}_model.h5")
        self.target_model.save(f"{filepath}_target_model.h5")
        
        # Save other attributes
        attrs = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'batch_size': self.batch_size,
            'update_target_freq': self.update_target_freq,
            'step_counter': self.step_counter
        }
        
        joblib.dump(attrs, f"{filepath}_attributes.joblib")
        logger.info(f"Saved DQN agent to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load an agent from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved agent.
        
        Returns:
        --------
        DQNAgent
            Loaded agent.
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras is not available. Cannot load DQN agent.")
        
        # Load attributes
        attrs = joblib.load(f"{filepath}_attributes.joblib")
        
        # Create instance
        agent = cls(
            state_size=attrs['state_size'],
            action_size=attrs['action_size'],
            hidden_layers=attrs['hidden_layers'],
            learning_rate=attrs['learning_rate'],
            gamma=attrs['gamma'],
            epsilon=attrs['epsilon'],
            epsilon_decay=attrs['epsilon_decay'],
            epsilon_min=attrs['epsilon_min'],
            batch_size=attrs['batch_size'],
            update_target_freq=attrs['update_target_freq']
        )
        
        # Load models
        agent.model = load_model(f"{filepath}_model.h5")
        agent.target_model = load_model(f"{filepath}_target_model.h5")
        
        # Set other attributes
        agent.step_counter = attrs['step_counter']
        
        logger.info(f"Loaded DQN agent from {filepath}")
        return agent


class A2CAgent:
    """
    Advantage Actor-Critic (A2C) agent for reinforcement learning.
    """
    
    def __init__(self, state_size, action_size, actor_hidden_layers=[64, 32], critic_hidden_layers=[64, 32], 
                 actor_learning_rate=0.001, critic_learning_rate=0.002, gamma=0.95, 
                 entropy_coef=0.01, batch_size=32):
        """
        Initialize the A2C agent.
        
        Parameters:
        -----------
        state_size : int
            Size of the state space.
        action_size : int
            Size of the action space.
        actor_hidden_layers : list, optional
            Number of units in each hidden layer of the actor network.
        critic_hidden_layers : list, optional
            Number of units in each hidden layer of the critic network.
        actor_learning_rate : float, optional
            Learning rate for the actor optimizer.
        critic_learning_rate : float, optional
            Learning rate for the critic optimizer.
        gamma : float, optional
            Discount factor for future rewards.
        entropy_coef : float, optional
            Coefficient for entropy regularization.
        batch_size : int, optional
            Batch size for training.
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras is not available. Cannot create A2C agent.")
        
        self.state_size = state_size
        self.action_size = action_size
        self.actor_hidden_layers = actor_hidden_layers
        self.critic_hidden_layers = critic_hidden_layers
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        
        # Create models
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # Initialize memory
        self.states = []
        self.actions = []
        self.rewards = []
        
        logger.info(f"Initialized A2C agent with state_size={state_size}, action_size={action_size}")
    
    def _build_actor(self):
        """
        Build the actor network.
        
        Returns:
        --------
        tensorflow.keras.models.Model
            Actor network model.
        """
        model = Sequential()
        
        # Add input layer
        model.add(Dense(self.actor_hidden_layers[0], activation='relu', input_dim=self.state_size))
        model.add(BatchNormalization())
        
        # Add hidden layers
        for units in self.actor_hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
        
        # Add output layer
        model.add(Dense(self.action_size, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.actor_learning_rate),
            loss='categorical_crossentropy'
        )
        
        return model
    
    def _build_critic(self):
        """
        Build the critic network.
        
        Returns:
        --------
        tensorflow.keras.models.Model
            Critic network model.
        """
        model = Sequential()
        
        # Add input layer
        model.add(Dense(self.critic_hidden_layers[0], activation='relu', input_dim=self.state_size))
        model.add(BatchNormalization())
        
        # Add hidden layers
        for units in self.critic_hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
        
        # Add output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.critic_learning_rate),
            loss='mse'
        )
        
        return model
    
    def act(self, state):
        """
        Choose an action based on the current state.
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state.
        
        Returns:
        --------
        int
            Chosen action.
        """
        state = np.reshape(state, [1, self.state_size])
        action_probs = self.actor.predict(state, verbose=0)[0]
        
        # Choose action based on probabilities
        action = np.random.choice(self.action_size, p=action_probs)
        
        return action
    
    def remember(self, state, action, reward):
        """
        Store an experience in memory.
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state.
        action : int
            Action taken.
        reward : float
            Reward received.
        """
        self.states.append(state)
        
        # Convert action to one-hot encoding
        action_onehot = np.zeros(self.action_size)
        action_onehot[action] = 1
        self.actions.append(action_onehot)
        
        self.rewards.append(reward)
    
    def train(self):
        """
        Train the actor and critic networks.
        
        Returns:
        --------
        tuple
            (actor_loss, critic_loss)
        """
        if len(self.states) == 0:
            return 0, 0
        
        # Convert to numpy arrays
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        
        # Calculate discounted rewards
        discounted_rewards = self._discount_rewards(rewards)
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        
        # Get value predictions
        values = self.critic.predict(states, verbose=0).flatten()
        
        # Calculate advantages
        advantages = discounted_rewards - values
        
        # Train critic
        critic_loss = self.critic.fit(states, discounted_rewards, epochs=1, verbose=0).history['loss'][0]
        
        # Train actor
        actor_loss = self._train_actor(states, actions, advantages)
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        
        return actor_loss, critic_loss
    
    def _train_actor(self, states, actions, advantages):
        """
        Train the actor network.
        
        Parameters:
        -----------
        states : numpy.ndarray
            States.
        actions : numpy.ndarray
            Actions (one-hot encoded).
        advantages : numpy.ndarray
            Advantages.
        
        Returns:
        --------
        float
            Loss value.
        """
        with tf.GradientTape() as tape:
            # Get action probabilities
            action_probs = self.actor(states, training=True)
            
            # Calculate policy loss
            policy_loss = -tf.reduce_sum(tf.math.log(tf.reduce_sum(action_probs * actions, axis=1) + 1e-8) * advantages)
            
            # Calculate entropy loss
            entropy_loss = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8))
            
            # Calculate total loss
            loss = policy_loss - self.entropy_coef * entropy_loss
        
        # Calculate gradients
        grads = tape.gradient(loss, self.actor.trainable_variables)
        
        # Apply gradients
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        
        return loss.numpy()
    
    def _discount_rewards(self, rewards):
        """
        Calculate discounted rewards.
        
        Parameters:
        -----------
        rewards : numpy.ndarray
            Rewards.
        
        Returns:
        --------
        numpy.ndarray
            Discounted rewards.
        """
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        
        return discounted_rewards
    
    def save(self, filepath):
        """
        Save the agent to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the agent.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save models
        self.actor.save(f"{filepath}_actor.h5")
        self.critic.save(f"{filepath}_critic.h5")
        
        # Save other attributes
        attrs = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'actor_hidden_layers': self.actor_hidden_layers,
            'critic_hidden_layers': self.critic_hidden_layers,
            'actor_learning_rate': self.actor_learning_rate,
            'critic_learning_rate': self.critic_learning_rate,
            'gamma': self.gamma,
            'entropy_coef': self.entropy_coef,
            'batch_size': self.batch_size
        }
        
        joblib.dump(attrs, f"{filepath}_attributes.joblib")
        logger.info(f"Saved A2C agent to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load an agent from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved agent.
        
        Returns:
        --------
        A2CAgent
            Loaded agent.
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras is not available. Cannot load A2C agent.")
        
        # Load attributes
        attrs = joblib.load(f"{filepath}_attributes.joblib")
        
        # Create instance
        agent = cls(
            state_size=attrs['state_size'],
            action_size=attrs['action_size'],
            actor_hidden_layers=attrs['actor_hidden_layers'],
            critic_hidden_layers=attrs['critic_hidden_layers'],
            actor_learning_rate=attrs['actor_learning_rate'],
            critic_learning_rate=attrs['critic_learning_rate'],
            gamma=attrs['gamma'],
            entropy_coef=attrs['entropy_coef'],
            batch_size=attrs['batch_size']
        )
        
        # Load models
        agent.actor = load_model(f"{filepath}_actor.h5")
        agent.critic = load_model(f"{filepath}_critic.h5")
        
        logger.info(f"Loaded A2C agent from {filepath}")
        return agent


class RLTrainer:
    """
    Class for training and evaluating reinforcement learning agents.
    """
    
    def __init__(self, data, agent_type='dqn', state_features=None, initial_balance=100000.0, 
                 transaction_cost=0.0001, max_position=1, reward_function='pnl', 
                 train_test_split=0.8, random_state=42):
        """
        Initialize the RL trainer.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data with OHLCV columns.
        agent_type : str, optional
            Type of agent to use ('dqn' or 'a2c').
        state_features : list, optional
            List of feature names to include in the state.
        initial_balance : float, optional
            Initial account balance.
        transaction_cost : float, optional
            Transaction cost as a fraction of trade value.
        max_position : int, optional
            Maximum position size (number of contracts).
        reward_function : str, optional
            Function to calculate rewards ('pnl', 'sharpe', 'sortino').
        train_test_split : float, optional
            Proportion of the data to use for training.
        random_state : int, optional
            Random seed for reproducibility.
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras is not available. Cannot create RL trainer.")
        
        self.data = data
        self.agent_type = agent_type
        self.state_features = state_features
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.reward_function = reward_function
        self.train_test_split = train_test_split
        self.random_state = random_state
        
        # Split data into train and test sets
        train_size = int(len(data) * train_test_split)
        self.train_data = data.iloc[:train_size]
        self.test_data = data.iloc[train_size:]
        
        logger.info(f"Initialized RLTrainer with {len(data)} data points")
        logger.info(f"Train set: {len(self.train_data)} samples, Test set: {len(self.test_data)} samples")
        
        # Create environments
        self.train_env = TradingEnvironment(
            data=self.train_data,
            initial_balance=initial_balance,
            transaction_cost=transaction_cost,
            max_position=max_position,
            reward_function=reward_function,
            state_features=state_features
        )
        
        self.test_env = TradingEnvironment(
            data=self.test_data,
            initial_balance=initial_balance,
            transaction_cost=transaction_cost,
            max_position=max_position,
            reward_function=reward_function,
            state_features=state_features
        )
        
        # Determine state and action sizes
        state = self.train_env.reset()
        self.state_size = len(state)
        self.action_size = 3  # 0: hold, 1: buy, 2: sell
        
        # Create agent
        if agent_type == 'dqn':
            self.agent = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size
            )
        elif agent_type == 'a2c':
            self.agent = A2CAgent(
                state_size=self.state_size,
                action_size=self.action_size
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def train(self, episodes=100, batch_size=32, render_freq=10):
        """
        Train the agent.
        
        Parameters:
        -----------
        episodes : int, optional
            Number of episodes to train for.
        batch_size : int, optional
            Batch size for training.
        render_freq : int, optional
            Frequency of rendering during training.
        
        Returns:
        --------
        dict
            Training metrics.
        """
        logger.info(f"Training {self.agent_type} agent for {episodes} episodes")
        
        # Initialize metrics
        metrics = {
            'episode_rewards': [],
            'episode_losses': [],
            'portfolio_values': []
        }
        
        for episode in range(episodes):
            # Reset environment
            state = self.train_env.reset()
            done = False
            episode_reward = 0
            episode_losses = []
            
            while not done:
                # Choose action
                if self.agent_type == 'dqn':
                    action = self.agent.act(state)
                else:  # a2c
                    action = self.agent.act(state)
                
                # Take action
                next_state, reward, done, info = self.train_env.step(action)
                
                # Remember experience
                if self.agent_type == 'dqn':
                    self.agent.remember(state, action, reward, next_state, done)
                    
                    # Train agent
                    loss = self.agent.replay()
                    if loss > 0:
                        episode_losses.append(loss)
                else:  # a2c
                    self.agent.remember(state, action, reward)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
            
            # Train A2C agent at the end of the episode
            if self.agent_type == 'a2c':
                actor_loss, critic_loss = self.agent.train()
                episode_losses.append(critic_loss)
            
            # Get performance metrics
            performance = self.train_env.get_performance_metrics()
            
            # Update metrics
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_losses'].append(np.mean(episode_losses) if episode_losses else 0)
            metrics['portfolio_values'].append(performance['final_portfolio_value'])
            
            # Log progress
            if episode % render_freq == 0 or episode == episodes - 1:
                logger.info(f"Episode {episode}/{episodes}, Reward: {episode_reward:.2f}, "
                           f"Loss: {metrics['episode_losses'][-1]:.4f}, "
                           f"Portfolio Value: {performance['final_portfolio_value']:.2f}, "
                           f"Total Return: {performance['total_return']:.2%}")
        
        logger.info("Training completed")
        return metrics
    
    def test(self):
        """
        Test the agent on the test set.
        
        Returns:
        --------
        dict
            Test metrics.
        """
        logger.info("Testing agent on test set")
        
        # Reset environment
        state = self.test_env.reset()
        done = False
        test_reward = 0
        
        while not done:
            # Choose action
            if self.agent_type == 'dqn':
                action = self.agent.act(state, training=False)
            else:  # a2c
                action = self.agent.act(state)
            
            # Take action
            next_state, reward, done, info = self.test_env.step(action)
            
            # Update state and reward
            state = next_state
            test_reward += reward
        
        # Get performance metrics
        performance = self.test_env.get_performance_metrics()
        
        # Log results
        logger.info(f"Test Reward: {test_reward:.2f}, "
                   f"Portfolio Value: {performance['final_portfolio_value']:.2f}, "
                   f"Total Return: {performance['total_return']:.2%}, "
                   f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}, "
                   f"Max Drawdown: {performance['max_drawdown']:.2%}")
        
        return performance
    
    def plot_results(self, metrics, save_path=None):
        """
        Plot training results.
        
        Parameters:
        -----------
        metrics : dict
            Training metrics.
        save_path : str, optional
            Path to save the plot. If None, the plot is displayed.
        
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure.
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot rewards
        ax1.plot(metrics['episode_rewards'])
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        
        # Plot losses
        ax2.plot(metrics['episode_losses'])
        ax2.set_title('Episode Losses')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        
        # Plot portfolio values
        ax3.plot(metrics['portfolio_values'])
        ax3.set_title('Portfolio Values')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved results plot to {save_path}")
        
        return fig
    
    def save_agent(self, filepath):
        """
        Save the agent to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the agent.
        """
        self.agent.save(filepath)
    
    def load_agent(self, filepath):
        """
        Load an agent from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved agent.
        """
        if self.agent_type == 'dqn':
            self.agent = DQNAgent.load(filepath)
        else:  # a2c
            self.agent = A2CAgent.load(filepath)
        
        logger.info(f"Loaded {self.agent_type} agent from {filepath}")


if __name__ == "__main__":
    # Example usage
    if TENSORFLOW_AVAILABLE:
        print("Reinforcement Learning Models module ready for use")
    else:
        print("TensorFlow/Keras not available. Reinforcement learning models will not function.")
