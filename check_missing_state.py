import json
import itertools

def load_q_table(filename):
    with open(filename, 'r') as f:
        serialized_q_table = json.load(f)
    
    q_table = {
        tuple(map(int, state.strip('()').split(','))): values
        for state, values in serialized_q_table.items()
    }
    return q_table

def generate_valid_states():
    # First 3 danger states can be 0 or 1 (8 possibilities)
    danger_states = list(itertools.product([0, 1], repeat=3))
    
    # Middle 4 direction states can only have one 1 (4 possibilities)
    direction_states = [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)]
    
    # Last 4 states must have exactly two 1s (8 possibilities)
    food_states = []
    for i in range(4):
        for j in range(i+1, 4):
            state = [0, 0, 0, 0]
            state[i] = 1
            state[j] = 1
            food_states.append(tuple(state))
    
    # Combine all possible states
    valid_states = set()
    for d in danger_states:
        for m in direction_states:
            for f in food_states:
                valid_states.add(d + m + f)
    
    return valid_states

def find_missing_states(q_table_file):
    q_table = load_q_table(q_table_file)
    existing_states = set(q_table.keys())
    valid_states = generate_valid_states()
    
    print(f"Total valid states: {len(valid_states)} (should be 256)")
    print(f"States in Q-table: {len(existing_states)}")
    
    missing_states = valid_states - existing_states
    if missing_states:
        print("\nMissing state:")
        print(list(missing_states)[0])

if __name__ == "__main__":
    q_table_file = "q_table/q_table_20250224_201821.json"
    find_missing_states(q_table_file)
