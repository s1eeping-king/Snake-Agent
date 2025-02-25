import numpy as np
import pygame
import json
import cv2
import os
from datetime import datetime
from main import SnakeGame  # Import game environment

def load_q_table(filename):
    """Load Q-table from file"""
    with open(filename, 'r') as f:
        serialized_q_table = json.load(f)
    
    # Convert string keys back to tuples and lists back to numpy arrays
    q_table = {
        tuple(map(int, state.strip('()').split(','))): np.array(values)
        for state, values in serialized_q_table.items()
    }
    return q_table

def test_agent(q_table, episodes=10):
    """Test agent using loaded Q-table"""
    env = SnakeGame()
    env.game_speed = 60  # Set default frame rate to 30
    env.head_color = (0, 191, 255)  # Set snake head color to blue
    
    # Add episode count attributes
    env.total_episodes = episodes
    env.current_episode = 0
    
    # Create video save directory
    if not os.path.exists('videos_test'):
        os.makedirs('videos_test')
    
    # Set up video recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f'videos_test/snake_game_{timestamp}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, env.game_speed, 
                                 (env.width, env.height))
    
    scores = []
    max_score = 0
    clock = pygame.time.Clock()  # Create clock object
    
    try:
        for episode in range(episodes):
            env.current_episode = episode + 1  # Update current episode
            state = env.reset()
            episode_score = 0
            steps = 0
            max_steps = 3000
            
            print(f"\nStarting episode {episode + 1}")
            
            while not env.game_over and steps < max_steps:
                # Handle quit events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        video_writer.release()
                        pygame.quit()
                        return
                
                # Get Q-values for current state
                state_key = tuple(state)
                if state_key in q_table:
                    action = np.argmax(q_table[state_key])
                else:
                    action = np.random.randint(0, 3)
                
                # Execute action
                next_state, reward, done = env.step(action)
                state = next_state
                episode_score = env.score
                steps += 1
                
                # Render game screen
                env.render()
                
                # Convert pygame surface to numpy array and write to video
                pygame_surface = pygame.display.get_surface()
                frame = pygame.surfarray.array3d(pygame_surface)
                frame = frame.swapaxes(0, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)
                
                pygame.display.flip()
                clock.tick(env.game_speed)
            
            scores.append(episode_score)
            max_score = max(max_score, episode_score)
            print(f"Episode {episode + 1} ended, Score: {episode_score}, Steps: {steps}")
            
        print("\nTest Results:")
        print(f"Average Score: {np.mean(scores):.2f}")
        print(f"Highest Score: {max_score}")
        print(f"All Scores: {scores}")
        print(f"\nVideo saved to: {video_path}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        video_writer.release()
        pygame.quit()

def main():
    # Load Q-table
    q_table_file = "q_table/q_table_20250224_201821.json"
    try:
        q_table = load_q_table(q_table_file)
        print(f"Successfully loaded Q-table with {len(q_table)} states")
        
        # Run test
        episodes = int(input("Enter number of episodes to test: "))
        test_agent(q_table, episodes)
        
    except FileNotFoundError:
        print(f"Error: Q-table file not found {q_table_file}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
