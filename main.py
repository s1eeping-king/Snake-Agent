# Import required libraries
import numpy as np
import pygame
import random
import time
import os
import cv2
import datetime
import json
from collections import deque

class SnakeGame:
    def __init__(self, width=640, height=480):
        # Basic game parameter settings
        self.width = width
        self.height = height
        self.block_size = 20
        self.grid_width = width // self.block_size
        self.grid_height = height // self.block_size
        
        # Training related parameters
        self.current_episode = 0
        self.total_episodes = 0
        
        self.reset()
        
        # Pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Snake Q-Learning')
        self.clock = pygame.time.Clock()
        self.game_speed = 60
        
        # Game color settings
        self.snake_color = (0, 255, 0)
        self.head_color = (0, 191, 255)
        self.food_color = (255, 0, 0)
        self.background_color = (0, 0, 0)

    def reset(self):
        # Reset game state
        self.snake_pos = [(self.grid_width//2, self.grid_height//2)]
        self.snake_direction = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
        self.place_food()
        self.score = 0
        self.game_over = False
        return self.get_state()

    def place_food(self):
        # Randomly place food, ensuring no overlap with snake body
        while True:
            self.food_pos = (random.randint(0, self.grid_width-1),
                           random.randint(0, self.grid_height-1))
            if self.food_pos not in self.snake_pos:
                break

    def get_state(self):
        head_x, head_y = self.snake_pos[0]
        food_x, food_y = self.food_pos
        
        # Get state information: danger, movement direction, food position
        danger_straight = self.is_collision(self.get_next_head_position())
        danger_right = self.is_collision(self.get_next_head_position('right'))
        danger_left = self.is_collision(self.get_next_head_position('left'))
        
        dir_left = self.snake_direction == (-1, 0)
        dir_right = self.snake_direction == (1, 0)
        dir_up = self.snake_direction == (0, -1)
        dir_down = self.snake_direction == (0, 1)
        
        food_left = food_x < head_x
        food_right = food_x > head_x
        food_up = food_y < head_y
        food_down = food_y > head_y
        
        return np.array([
            danger_straight, danger_right, danger_left, # Danger in front, right, and left directions
            dir_left, dir_right, dir_up, dir_down, # Current movement direction
            food_left, food_right, food_up, food_down # Relative food direction
        ], dtype=int)

    def get_next_head_position(self, turn='none'):
        current_direction = self.snake_direction
        # Calculate new direction after turning
        if turn == 'right':
            if current_direction == (1, 0): new_direction = (0, 1)
            elif current_direction == (-1, 0): new_direction = (0, -1)
            elif current_direction == (0, 1): new_direction = (-1, 0)
            else: new_direction = (1, 0)
        elif turn == 'left':
            if current_direction == (1, 0): new_direction = (0, -1)
            elif current_direction == (-1, 0): new_direction = (0, 1)
            elif current_direction == (0, 1): new_direction = (1, 0)
            else: new_direction = (-1, 0)
        else:
            new_direction = current_direction
            
        head_x, head_y = self.snake_pos[0]
        return (head_x + new_direction[0], head_y + new_direction[1])

    def is_collision(self, position):
        x, y = position
        return (x < 0 or x >= self.grid_width or
                y < 0 or y >= self.grid_height or
                position in self.snake_pos[:-1])

    def step(self, action):
        # Update snake direction based on action
        if action == 1:  # Turn right
            if self.snake_direction == (1, 0): self.snake_direction = (0, 1)
            elif self.snake_direction == (-1, 0): self.snake_direction = (0, -1)
            elif self.snake_direction == (0, 1): self.snake_direction = (-1, 0)
            else: self.snake_direction = (1, 0)
        elif action == 2:  # Turn left
            if self.snake_direction == (1, 0): self.snake_direction = (0, -1)
            elif self.snake_direction == (-1, 0): self.snake_direction = (0, 1)
            elif self.snake_direction == (0, 1): self.snake_direction = (1, 0)
            else: self.snake_direction = (-1, 0)

        # Calculate new head position and update state
        new_head = (self.snake_pos[0][0] + self.snake_direction[0],
                   self.snake_pos[0][1] + self.snake_direction[1])
        
        reward = 0
        if self.is_collision(new_head):
            self.game_over = True
            reward = -10
        else:
            self.snake_pos.insert(0, new_head)
            if new_head == self.food_pos:
                self.score += 1
                reward = 10
                self.place_food()
            else:
                self.snake_pos.pop()
                food_distance = abs(new_head[0] - self.food_pos[0]) + abs(new_head[1] - self.food_pos[1])
                reward = -food_distance / (self.width + self.height)

        return self.get_state(), reward, self.game_over

    def render(self):
        self.screen.fill(self.background_color)
        
        # Draw food and snake
        pygame.draw.rect(self.screen, self.food_color, 
                        pygame.Rect(self.food_pos[0] * self.block_size,
                                  self.food_pos[1] * self.block_size,
                                  self.block_size, self.block_size))
        
        for i, pos in enumerate(self.snake_pos):
            color = self.head_color if i == 0 else self.snake_color
            pygame.draw.rect(self.screen, color,
                           pygame.Rect(pos[0] * self.block_size,
                                     pos[1] * self.block_size,
                                     self.block_size, self.block_size))
            
            if i == 0:  # Add border to snake head
                pygame.draw.rect(self.screen, (255, 255, 255),
                               pygame.Rect(pos[0] * self.block_size,
                                         pos[1] * self.block_size,
                                         self.block_size, self.block_size), 1)
        
        # Display score and training episodes
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        if hasattr(self, 'current_episode') and hasattr(self, 'total_episodes'):
            episode_text = font.render(f'Episode: {self.current_episode}/{self.total_episodes}', True, (255, 255, 255))
            self.screen.blit(episode_text, (10, self.height - 40))
        
        pygame.display.update()
        self.clock.tick(self.game_speed)

class QLearningAgent:
    def __init__(self, state_size, action_size):
        # Initialize Q-learning parameters
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.alpha = 0.1
        self.gamma = 0.9

    def get_action(self, state):
        state_key = tuple(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        # Choose action using ε-greedy policy
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state_key])

    def update(self, state, action, reward, next_state):
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        
        # Initialize Q-values for new state
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        # Q-learning update formula
        old_value = self.q_table[state_key][action]
        next_max = np.max(self.q_table[next_state_key])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state_key][action] = new_value

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train():
    env = SnakeGame()
    agent = QLearningAgent(state_size=11, action_size=3)
    episodes = 1000
    env.total_episodes = episodes
    scores = []
    best_score = 0

    if not os.path.exists('plot'):
        os.makedirs('plot')
    
    training_data = {
        'episodes': [],
        'scores': [],
        'epsilon': []
    }

    # Video recording settings
    record_video = input("Do you want to record training video? (y/n): ").lower().strip() == 'y'
    video_writer = None
    last_frame_time = None

    if record_video:
        if not os.path.exists('videos_train'):
            os.makedirs('videos_train')
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f'videos_train/snake_game_{timestamp}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 60,
                                     (env.width, env.height))
        
        last_frame_time = time.time()
        frame_interval = 1.0 / 60
        print(f"Video will be saved to: {video_path}")

    try:
        # Main training loop
        for episode in range(episodes):
            env.current_episode = episode + 1
            
            # Dynamically adjust frame rate
            if episode < 1500:
                if episode % 100 < 2:
                    env.game_speed = 3000
                else:
                    env.game_speed = 3000
            
            state = env.reset()
            total_reward = 0
            
            while not env.game_over:
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                
                env.render()
                
                if record_video:
                    current_time = time.time()
                    if current_time - last_frame_time >= frame_interval:
                        pygame_surface = pygame.display.get_surface()
                        frame = pygame.surfarray.array3d(pygame_surface)
                        frame = frame.swapaxes(0, 1)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        video_writer.write(frame)
                        last_frame_time = current_time
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt

                env.clock.tick(env.game_speed)

            if env.score > best_score:
                best_score = env.score
            
            scores.append(env.score)
            
            # Record training data
            training_data['episodes'].append(episode)
            training_data['scores'].append(env.score)
            training_data['epsilon'].append(agent.epsilon)
            
            if episode % 1 == 0:
                avg_score = np.mean(scores[-1:])
                print(f"Episode: {episode}, Score: {env.score}, Best Score: {best_score}")
                print(f"Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    except (KeyboardInterrupt, SystemExit):
        print("\nTraining interrupted, saving data...")
    finally:
        # Save training data and Q-table
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_data_path = f'plot/training_data_{timestamp}.json'
        with open(plot_data_path, 'w') as f:
            json.dump(training_data, f)
        print(f"Training data saved to: {plot_data_path}")
        
        save_q_table(agent.q_table)
        if record_video and video_writer is not None:
            video_writer.release()
            print(f"\nVideo saved to: {video_path}")
        pygame.quit()

def save_q_table(q_table):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"q_table/q_table_{timestamp}.json"
    
    serializable_q_table = {
        str(state): values.tolist() for state, values in q_table.items()
    }
    
    with open(filename, 'w') as f:
        json.dump(serializable_q_table, f, indent=2)
    
    print(f"Q-table saved to file: {filename}")

def load_q_table(filename):
    with open(filename, 'r') as f:
        serialized_q_table = json.load(f)
    
    q_table = {
        tuple(map(int, state.strip('()').split(','))): np.array(values)
        for state, values in serialized_q_table.items()
    }
    
    return q_table

if __name__ == "__main__":
    train()
