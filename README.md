# Snake-Agent

A Snake AI implementation based on Q-Learning reinforcement learning.
![snake_game_20250225_011428](https://github.com/user-attachments/assets/0f1348a5-5050-4d9e-8066-ab69be7cb59f)

## Project Structure

```
.
├── snake_v2_best.py      # Original framework implementation
├── main.py    # Optimized version
├── snake_v2_test.py      # AI testing script
├── check_missing_state.py # State space integrity checker
├── show.py               # Training data visualization tool
├── plot/                 # Training process data
├── q_table/             # Trained Q-tables
├── videos_test/         # Test process videos
└── videos_train/        # Training process videos
```

## Main Files

### snake_v2_best.py
Original snake game framework implementation, including:
- Basic game environment
- Q-Learning agent framework
- Training loop implementation

### main.py
Optimized version with improvements:
- More precise state space design
- Optimized reward function
- Improved exploration strategy
- More efficient training process

### snake_v2_test.py
For testing trained AI models:
- Load saved Q-table
- Visualize AI behavior
- Record test data
- Save test videos

### Utility Tools

- **check_missing_state.py**: State space integrity checker
  - Generate all valid state combinations (256 states)
  - Check for missing states in Q-table
  - Validate state vector legality (first 3 bits arbitrary, middle 4 bits must have exactly one 1, last 4 bits must have exactly two 1s)
- **show.py**: Training process data visualization, including scores, exploration rates, etc.

### Data Directories

- **plot/**: Stores training process data, including episode scores, average scores, etc.
- **q_table/**: Stores trained Q-tables for testing or continued training
- **videos_test/**: Stores test process recordings
- **videos_train/**: Stores training process recordings

## Usage

1. Train new model:

```bash
python main.py
```

2. Test trained model:
```bash
python snake_v2_test.py
```

3. Visualize training data:
```bash
python show.py
```

4. Check state space integrity:
```bash
python check_missing_state.py
```
This checks if the Q-table contains all 256 possible valid state combinations. If missing states are found, the program will output the specific missing state vectors.

## State Space Design

State vector contains 11 binary bits:
- First 3 bits: Danger states (obstacles in front, right, left, can be any combination)
- Middle 4 bits: Movement direction (up, down, left, right, must have exactly one 1)
- Last 4 bits: Food relative position (food relative to snake head, must have exactly two 1s)

Total states: 8 * 4 * 8 = 256 possible states
- First 3 bits: 2^3 = 8 combinations
- Middle 4 bits: 4 combinations (must have one 1)
- Last 4 bits: C(4,2) = 6 combinations (must have two 1s)

## Action Space

3 possible actions:
- 0: Turn left
- 1: Go straight
- 2: Turn right

## Reward Design

- Eating food: +10
- Hitting wall or self: -10
- Distance penalty: -food_distance / (self.width + self.height)

## Training Parameters

- Learning rate (alpha): 0.1
- Discount factor (gamma): 0.9
- Initial exploration rate (epsilon): 1.0
- Minimum exploration rate: 0.01
- Exploration decay rate: 0.9995

## Training Process

1. Initialize Q-table
2. For each training episode:
   - Reset game environment
   - Get initial state
   - Loop until game over:
     - Choose action using ε-greedy policy
     - Execute action and get reward
     - Update Q-value
     - Update state
   - Decay exploration rate
3. Save trained Q-table
![image](https://github.com/user-attachments/assets/90c4703c-33cb-452c-b6fe-dcf6fe61d146)

## Notes

1. FPS can be adjusted during testing for better observation
2. All training data and models are automatically saved, can be interrupted and continued anytime

## Future Improvements

1. Implement Deep Q-Learning version
2. Add more state features, such as dead-end detection, loop detection, etc.
3. Optimize reward function design

## Link
Github Link: https://github.com/s1eeping-king/Snake-Agent
Youtube Link: https://www.youtube.com/watch?v=hTUsdgDKyW8
