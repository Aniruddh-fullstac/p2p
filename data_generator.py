import numpy as np
import pandas as pd
import random

def generate_peer_ratings(hierarchy_config=None, hierarchy_levels=None):
    """
    Generate peer ratings with custom names, roles, and hierarchy levels
    hierarchy_config: dict of format {'role_name': [list_of_names]}
    hierarchy_levels: dict of format {'role_name': level_number} or list in descending order
    """
    # Default hierarchy if none provided
    if hierarchy_config is None:
        hierarchy_config = {
            'Regional Manager': ['Sarah Chen'],
            'Store Manager': ['Michael Rodriguez', 'David Kim'],
            'Shift Lead': ['Emma Watson', 'James Smith', 'Priya Patel'],
            'Senior Staff': ['John Doe', 'Maria Garcia'],
            'Junior Staff': ['Alex Johnson', 'Lisa Zhang']
        }
    
    # Default hierarchy levels if none provided
    if hierarchy_levels is None:
        # Can be provided as dict with explicit levels
        hierarchy_levels = {
            'Regional Manager': 5,
            'Store Manager': 4,
            'Shift Lead': 3,
            'Senior Staff': 2,
            'Junior Staff': 1
        }
        # Or as ordered list from highest to lowest
        # hierarchy_levels = ['Regional Manager', 'Store Manager', 'Shift Lead', 'Senior Staff', 'Junior Staff']
    
    # Convert list format to dict if needed
    if isinstance(hierarchy_levels, list):
        hierarchy_levels = {role: len(hierarchy_levels) - i 
                          for i, role in enumerate(hierarchy_levels)}
    
    # Create list to store names, roles, and levels
    names_and_roles = []
    for role, names in hierarchy_config.items():
        for name in names:
            names_and_roles.append({
                'Name': name,
                'Role': role,
                'Level': hierarchy_levels[role]
            })
    
    # Create the names DataFrame
    names_df = pd.DataFrame(names_and_roles)
    names_df.set_index('Name', inplace=True)
    
    # Calculate total participants
    NUM_PARTICIPANTS = len(names_df)
    
    # Generate ratings matrix
    ratings = np.zeros((NUM_PARTICIPANTS, NUM_PARTICIPANTS))
    
    # Define personality types
    PERSONALITIES = {
        'overconfident': {'self': (9, 10), 'others': (5, 7)},
        'underconfident': {'self': (4, 5), 'others': (7, 9)},
        'self_biased': {'self': (9, 10), 'others': (4, 8)},
        'category_biased': {'same_tier': (8, 10), 'other_tier': (5, 7)},
        'cross_category_biased': {'preferred_tier': (8, 10), 'other_tier': (5, 7)},
        'neutral': {'all': (6, 8)}
    }
    
    # Assign personalities randomly
    personality_assignments = {}
    available_personalities = list(PERSONALITIES.keys())
    
    for idx, name in enumerate(names_df.index, 1):
        personality_assignments[name] = random.choice(available_personalities)
    
    def get_tier(name):
        """Get the role/tier of a person based on their name"""
        return names_df.loc[name, 'Role'].lower()
    
    # Generate ratings using names instead of numbers
    for i, rater_name in enumerate(names_df.index):
        rater_personality = personality_assignments[rater_name]
        rater_tier = get_tier(rater_name)
        
        for j, rated_name in enumerate(names_df.index):
            rated_tier = get_tier(rated_name)
            
            # Determine rating based on personality
            if rater_name == rated_name:  # Self-rating
                if rater_personality == 'overconfident':
                    rating = random.uniform(*PERSONALITIES['overconfident']['self'])
                elif rater_personality == 'underconfident':
                    rating = random.uniform(*PERSONALITIES['underconfident']['self'])
                else:
                    rating = random.uniform(7, 10)
            else:
                if rater_personality == 'neutral':
                    rating = random.uniform(*PERSONALITIES['neutral']['all'])
                elif rater_personality == 'category_biased':
                    if rater_tier == rated_tier:
                        rating = random.uniform(*PERSONALITIES['category_biased']['same_tier'])
                    else:
                        rating = random.uniform(*PERSONALITIES['category_biased']['other_tier'])
                elif rater_personality == 'cross_category_biased':
                    if rated_tier == 'managers':
                        rating = random.uniform(*PERSONALITIES['cross_category_biased']['preferred_tier'])
                    else:
                        rating = random.uniform(*PERSONALITIES['cross_category_biased']['other_tier'])
                else:
                    rating = random.uniform(4, 8)
            
            ratings[i][j] = round(rating, 1)
    
    # Create the ratings DataFrame with the proper index
    df = pd.DataFrame(ratings, 
                     columns=names_df.index, 
                     index=names_df.index)
    
    # Combine ratings with role and level information
    final_df = pd.concat([df, names_df], axis=1)
    
    # Save to CSV
    final_df.to_csv('peer_ratings.csv')
    
    return personality_assignments

if __name__ == "__main__":
    personality_assignments = generate_peer_ratings()
    print("Data generated and saved to peer_ratings.csv")
    print("\nPersonality assignments:")
    for person, personality in personality_assignments.items():
        print(f"{person}: {personality}") 