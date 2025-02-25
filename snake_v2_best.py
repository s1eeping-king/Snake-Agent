import numpy as np
import pygame
import random
from collections import deque
import time

class SnakeGame:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.block_size = 20
        self.reset()
        
        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Snake Q-Learning')
        self.clock = pygame.time.Clock()
        self.game_speed = 30  # 添加游戏速度参数

    def reset(self):
        # 初始化蛇的位置和方向
        self.snake_pos = [(self.width//2, self.height//2)]
        self.snake_direction = random.choice([(0, -self.block_size), (0, self.block_size),
                                            (-self.block_size, 0), (self.block_size, 0)])
        # 随机放置食物
        self.place_food()
        self.score = 0
        self.game_over = False
        return self.get_state()

    def place_food(self):
        while True:
            self.food_pos = (random.randint(0, (self.width-self.block_size)//self.block_size) * self.block_size,
                           random.randint(0, (self.height-self.block_size)//self.block_size) * self.block_size)
            if self.food_pos not in self.snake_pos:
                break

    def get_state(self):
        head_x, head_y = self.snake_pos[0]
        food_x, food_y = self.food_pos
        
        # 获取蛇头周围的危险位置
        danger_straight = self.is_collision(self.get_next_head_position())
        danger_right = self.is_collision(self.get_next_head_position('right'))
        danger_left = self.is_collision(self.get_next_head_position('left'))
        
        # 当前方向
        dir_left = self.snake_direction == (-self.block_size, 0)
        dir_right = self.snake_direction == (self.block_size, 0)
        dir_up = self.snake_direction == (0, -self.block_size)
        dir_down = self.snake_direction == (0, self.block_size)
        
        # 食物相对位置
        food_left = food_x < head_x
        food_right = food_x > head_x
        food_up = food_y < head_y
        food_down = food_y > head_y
        
        return np.array([
            danger_straight, danger_right, danger_left,
            dir_left, dir_right, dir_up, dir_down,
            food_left, food_right, food_up, food_down
        ], dtype=int)

    def get_next_head_position(self, turn='none'):
        current_direction = self.snake_direction
        if turn == 'right':
            if current_direction == (self.block_size, 0): new_direction = (0, self.block_size)
            elif current_direction == (-self.block_size, 0): new_direction = (0, -self.block_size)
            elif current_direction == (0, self.block_size): new_direction = (-self.block_size, 0)
            else: new_direction = (self.block_size, 0)
        elif turn == 'left':
            if current_direction == (self.block_size, 0): new_direction = (0, -self.block_size)
            elif current_direction == (-self.block_size, 0): new_direction = (0, self.block_size)
            elif current_direction == (0, self.block_size): new_direction = (self.block_size, 0)
            else: new_direction = (-self.block_size, 0)
        else:
            new_direction = current_direction
            
        head_x, head_y = self.snake_pos[0]
        return (head_x + new_direction[0], head_y + new_direction[1])

    def is_collision(self, position):
        x, y = position
        return (x < 0 or x >= self.width or
                y < 0 or y >= self.height or
                position in self.snake_pos[:-1])

    def step(self, action):
        # 0: 直行, 1: 右转, 2: 左转
        if action == 1:  # 右转
            if self.snake_direction == (self.block_size, 0): self.snake_direction = (0, self.block_size)
            elif self.snake_direction == (-self.block_size, 0): self.snake_direction = (0, -self.block_size)
            elif self.snake_direction == (0, self.block_size): self.snake_direction = (-self.block_size, 0)
            else: self.snake_direction = (self.block_size, 0)
        elif action == 2:  # 左转
            if self.snake_direction == (self.block_size, 0): self.snake_direction = (0, -self.block_size)
            elif self.snake_direction == (-self.block_size, 0): self.snake_direction = (0, self.block_size)
            elif self.snake_direction == (0, self.block_size): self.snake_direction = (self.block_size, 0)
            else: self.snake_direction = (-self.block_size, 0)

        # 移动蛇头
        new_head = (self.snake_pos[0][0] + self.snake_direction[0],
                   self.snake_pos[0][1] + self.snake_direction[1])
        
        # 检查碰撞
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
                # 根据距离食物的远近给予小奖励
                food_distance = abs(new_head[0] - self.food_pos[0]) + abs(new_head[1] - self.food_pos[1])
                reward = -food_distance / (self.width + self.height)  # 归一化距离奖励

        return self.get_state(), reward, self.game_over

    def render(self):
        self.screen.fill((0, 0, 0))
        
        # 画蛇
        for pos in self.snake_pos:
            pygame.draw.rect(self.screen, (0, 255, 0),
                           pygame.Rect(pos[0], pos[1], self.block_size, self.block_size))
        
        # 画食物
        pygame.draw.rect(self.screen, (255, 0, 0),
                        pygame.Rect(self.food_pos[0], self.food_pos[1], self.block_size, self.block_size))
        
        pygame.display.flip()
        self.clock.tick(self.game_speed)  # 使用游戏速度参数

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.epsilon = 1.0  # 初始探索率设为1
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.alpha = 0.1
        self.gamma = 0.9

    def get_action(self, state):
        state_key = tuple(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state_key])

    def update(self, state, action, reward, next_state):
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        old_value = self.q_table[state_key][action]
        next_max = np.max(self.q_table[next_state_key])
        
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state_key][action] = new_value

        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train():
    env = SnakeGame()
    env.game_speed = 1000
    agent = QLearningAgent(state_size=11, action_size=3)
    episodes = 1000
    scores = []
    best_score = 0

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while not env.game_over:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            
            env.render()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

        # 保存最佳得分
        if env.score > best_score:
            best_score = env.score
        
        scores.append(env.score)
        if episode % 100 == 0:
            print(f"Episode: {episode}, Score: {env.score}, Best Score: {best_score}")
            print(f"Average Score: {np.mean(scores[-100:]):.2f}, Epsilon: {agent.epsilon:.3f}")

if __name__ == "__main__":
    train()
