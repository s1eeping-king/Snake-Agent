import json
import matplotlib.pyplot as plt
import os
import glob

def plot_learning_curve(data_file):
    # Load training data
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Calculate moving average
    window_size = 100
    scores = data['scores']
    moving_avg = []
    for i in range(len(scores)):
        if i < window_size:
            moving_avg.append(sum(scores[:i+1]) / (i+1))
        else:
            moving_avg.append(sum(scores[i-window_size+1:i+1]) / window_size)
    
    # Plot score curve
    plt.subplot(2, 1, 1)
    plt.plot(data['episodes'], data['scores'], 'b-', alpha=0.3, label='Score')
    plt.plot(data['episodes'], moving_avg, 'r-', label='Moving Average (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    # Plot exploration rate curve
    plt.subplot(2, 1, 2)
    plt.plot(data['episodes'], data['epsilon'], 'g-', label='Epsilon')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate')
    plt.legend()
    plt.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plot_file = data_file.replace('.json', '.png')
    plt.savefig(plot_file)
    print(f"Learning curve saved to: {plot_file}")
    
    # Show figure
    plt.show()

def main():
    # Get the latest training data file
    data_files = glob.glob('plot/training_data_*.json')
    if not data_files:
        print("No training data file found!")
        return
    
    latest_file = max(data_files, key=os.path.getctime)
    print(f"Processing latest training data file: {latest_file}")
    plot_learning_curve(latest_file)

if __name__ == "__main__":
    main()
