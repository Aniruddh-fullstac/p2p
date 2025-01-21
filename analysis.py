import pandas as pd
import numpy as np
from scipy import stats
import networkx as nx

class PeerRatingAnalyzer:
    def __init__(self, ratings_df, hierarchy_config=None):
        """
        Initialize analyzer with flexible hierarchy configuration
        ratings_df: DataFrame with 'Role' and 'Level' columns
        hierarchy_config: dict of format {'role_name': level_number}
        """
        # Separate ratings from role information
        self.roles = ratings_df['Role']
        self.levels = ratings_df['Level']
        self.ratings = ratings_df.drop(columns=['Role', 'Level'])
        self.num_participants = len(self.ratings)
        
        # Create role and level mappings
        self.role_mapping = {name: role for name, role in self.roles.items()}
        self.level_mapping = {name: level for name, level in self.levels.items()}
        
        # Store hierarchy configuration
        if hierarchy_config is None:
            # Create hierarchy config from role mapping
            self.hierarchy_config = {}
            for role in set(self.roles):
                members = [name for name, r in self.role_mapping.items() if r == role]
                self.hierarchy_config[role] = members
        else:
            self.hierarchy_config = hierarchy_config
        
        # Create hierarchy structure
        self.hierarchy_structure = self._create_hierarchy_structure()
    
    def _create_hierarchy_structure(self):
        """Create hierarchy structure with levels and reporting relationships"""
        # Get unique roles and their levels
        role_levels = {}
        for name, role in self.role_mapping.items():
            level = self.level_mapping[name]
            if role not in role_levels:
                role_levels[role] = level
        
        # Sort roles by level
        sorted_roles = sorted(role_levels.items(), key=lambda x: x[1], reverse=True)
        
        # Create hierarchy structure
        hierarchy = {}
        for role, level in sorted_roles:
            hierarchy[role] = {
                'level': level,
                'reports_to': [r for r, l in sorted_roles if l == level + 1],
                'manages': [r for r, l in sorted_roles if l == level - 1]
            }
        
        return hierarchy
    
    def get_reporting_chain(self, person_name):
        """Get the reporting chain for a person"""
        role = self.role_mapping[person_name]
        level = self.level_mapping[person_name]
        
        superiors = []
        current_level = level
        while current_level < max(self.levels):
            superior_roles = [r for r, l in self.role_mapping.items() 
                            if self.level_mapping[r] == current_level + 1]
            if superior_roles:
                superiors.extend(superior_roles)
            current_level += 1
        
        subordinates = []
        current_level = level
        while current_level > min(self.levels):
            subordinate_roles = [r for r, l in self.role_mapping.items() 
                               if self.level_mapping[r] == current_level - 1]
            if subordinate_roles:
                subordinates.extend(subordinate_roles)
            current_level -= 1
        
        return {
            'superiors': superiors,
            'subordinates': subordinates,
            'peers': [n for n, r in self.role_mapping.items() 
                     if self.level_mapping[n] == level and n != person_name]
        }
    
    def analyze_hierarchical_ratings(self):
        """Analyze ratings patterns between different hierarchical levels"""
        level_ratings = []
        
        for rater_level in sorted(set(self.levels)):
            raters = [name for name, level in self.level_mapping.items() 
                     if level == rater_level]
            
            for rated_level in sorted(set(self.levels)):
                rated = [name for name, level in self.level_mapping.items() 
                        if level == rated_level]
                
                if raters and rated:
                    mean_rating = self.ratings.loc[raters, rated].mean().mean()
                    level_ratings.append({
                        'Rater_Level': rater_level,
                        'Rated_Level': rated_level,
                        'Mean_Rating': mean_rating,
                        'Rating_Type': 'Upward' if rater_level < rated_level else
                                     'Downward' if rater_level > rated_level else 'Peer'
                    })
        
        return pd.DataFrame(level_ratings)
    
    def get_person_role(self, person_name):
        """Get role for a given person name"""
        return self.roles.get(person_name, 'Unknown')
    
    def calculate_self_bias(self):
        """Calculate bias between self-rating and average rating from others"""
        self_ratings = np.diag(self.ratings)
        peer_ratings = []
        
        for i in range(self.num_participants):
            others_ratings = np.concatenate([self.ratings.iloc[i, :i], self.ratings.iloc[i, i+1:]])
            peer_ratings.append(np.mean(others_ratings))
            
        return pd.DataFrame({
            'Person': self.ratings.index,
            'Self_Rating': self_ratings,
            'Peer_Rating': peer_ratings,
            'Bias': self_ratings - peer_ratings
        })
    
    def calculate_rating_consistency(self):
        """Calculate variance in ratings given by each person"""
        return pd.DataFrame({
            'Person': self.ratings.index,
            'Rating_Variance': [np.var(self.ratings.iloc[i]) for i in range(self.num_participants)],
            'Rating_Mean': [np.mean(self.ratings.iloc[i]) for i in range(self.num_participants)]
        })
    
    def analyze_reciprocity(self):
        """Analyze reciprocity in ratings between pairs"""
        reciprocity_scores = []
        for i in range(self.num_participants):
            for j in range(i+1, self.num_participants):
                rating_ij = self.ratings.iloc[i, j]
                rating_ji = self.ratings.iloc[j, i]
                difference = abs(rating_ij - rating_ji)
                reciprocity_scores.append({
                    'Person1': self.ratings.index[i],
                    'Person2': self.ratings.index[j],
                    'Rating1': rating_ij,
                    'Rating2': rating_ji,
                    'Difference': difference
                })
        return pd.DataFrame(reciprocity_scores)
    
    def analyze_category_bias(self):
        """Analyze rating patterns between different hierarchy levels"""
        bias_data = []
        
        # For each role (raters)
        for rater_role in set(self.roles):  # Use unique roles instead of hierarchy_config
            # Get all raters of this role
            raters = [name for name, role in self.role_mapping.items() 
                     if role == rater_role]
            
            # For each role (rated)
            for rated_role in set(self.roles):  # Changed from self.hierarchy_config.keys()
                # Get all rated people of this role
                rated = [name for name, role in self.role_mapping.items() 
                        if role == rated_role]
                
                if raters and rated:  # Only if we have people in both roles
                    # Calculate mean rating
                    mean_rating = self.ratings.loc[raters, rated].mean().mean()
                    
                    bias_data.append({
                        'Rater_Category': rater_role,
                        'Rated_Category': rated_role,
                        'Mean_Rating': mean_rating
                    })
        
        return pd.DataFrame(bias_data)
    
    def calculate_influence_scores(self):
        """Calculate influence scores using eigenvector centrality"""
        G = nx.from_pandas_adjacency(self.ratings)
        centrality = nx.eigenvector_centrality(G, weight='weight')
        
        return pd.DataFrame({
            'Person': self.ratings.index,
            'Influence_Score': list(centrality.values())
        })
    
    def get_individual_reciprocity(self, person_id):
        """Analyze reciprocal relationships for a specific person"""
        try:
            # Convert person_id to actual name if numeric
            if isinstance(person_id, int):
                person_name = self.ratings.index[person_id-1]
            else:
                person_name = person_id
            
            reciprocal_relations = []
            
            for other_name in self.ratings.index:
                if other_name != person_name:
                    rating_given = self.ratings.loc[person_name, other_name]
                    rating_received = self.ratings.loc[other_name, person_name]
                    
                    # Calculate reciprocity score
                    reciprocity = min(rating_given, rating_received) / max(rating_given, rating_received)
                    
                    # Analyze relationship type
                    relationship = self.analyze_relationship(person_name, other_name)
                    
                    reciprocal_relations.append({
                        'Other_Person': other_name,
                        'Rating_Given': rating_given,
                        'Rating_Received': rating_received,
                        'Reciprocity_Score': reciprocity,
                        'Type': relationship['type'],
                        'Sentiment': relationship['sentiment'],
                        'Direction': relationship['direction']
                    })
            
            return pd.DataFrame(reciprocal_relations)
        except Exception as e:
            print(f"Error in get_individual_reciprocity: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def predict_personality_type(self, person_id):
        """Predict personality type based on rating patterns"""
        try:
            # Get actual name from index
            person = self.ratings.index[person_id-1]
            
            # Get key metrics
            self_rating = self.ratings.loc[person, person]  # How they rate themselves
            peer_ratings = self.ratings[person].mean()  # How others rate them
            ratings_given = self.ratings.loc[person]  # All ratings they gave
            
            # Get hierarchical ratings
            third = len(self.ratings) // 3
            managers_ratings = ratings_given.iloc[:third].mean()
            team_leads_ratings = ratings_given.iloc[third:2*third].mean()
            employees_ratings = ratings_given.iloc[2*third:].mean()
            
            # Calculate personality scores
            personality_scores = {
                'overconfident': (
                    1 if self_rating >= 8 and self_rating > peer_ratings + 2 else 0
                ),
                'underconfident': (
                    1 if self_rating <= 6 and self_rating < peer_ratings - 2 else 0
                ),
                'category_biased': (
                    1 if max(managers_ratings, team_leads_ratings, employees_ratings) -
                    min(managers_ratings, team_leads_ratings, employees_ratings) >= 2 else 0
                ),
                'neutral': (
                    1 if abs(self_rating - peer_ratings) <= 1 and 
                    (ratings_given.max() - ratings_given.min()) <= 2 else 0
                )
            }
            
            # Get the most likely personality type
            personality_match = max(personality_scores.items(), key=lambda x: x[1])[0]
            confidence = personality_scores[personality_match]
            
            return personality_match, confidence
            
        except Exception as e:
            print(f"Error in personality prediction: {e}")
            return 'neutral', 0
    
    def get_category_bias_detail(self, person_id):
        """Analyze which category the person is biased towards"""
        try:
            # Get actual name from index
            person = self.ratings.index[person_id-1]
            
            # Calculate average ratings for each category
            category_ratings = {}
            for role, members in self.hierarchy_config.items():
                # Get names of people in this role
                role_members = [name for name, role_name in self.role_mapping.items() 
                              if role_name == role]
                if role_members:  # Only calculate if there are members in this role
                    category_ratings[role] = self.ratings.loc[person, role_members].mean()
            
            # Find highest rated category
            if category_ratings:
                preferred_category = max(category_ratings.items(), key=lambda x: x[1])[0]
                own_category = self.get_person_role(person)
                
                if preferred_category == own_category:
                    return f"Shows bias towards own category ({own_category})"
                else:
                    return f"Shows bias towards {preferred_category}"
            else:
                return "No category bias detected"
            
        except Exception as e:
            print(f"Error in category bias detail: {e}")
            return "Unable to determine category bias"
    
    def calculate_leadership_index(self, person_id):
        """Calculate leadership index with improved error handling"""
        try:
            # Get actual name from index
            person = self.ratings.index[person_id-1]
            
            # Get ratings data
            ratings_given = self.ratings.loc[person].drop(person)
            peer_ratings = self.ratings[person].drop(person)
            
            # Calculate recognition (0-1 scale)
            rating_range = ratings_given.max() - ratings_given.min()
            rating_std = ratings_given.std()
            recognition = min(1.0, (0.5 * (rating_range / 9) + 
                              0.5 * (1 - rating_std / 3)))
            
            # Calculate alignment (0-1 scale)
            all_ratings = self.ratings.drop(person)
            high_performers = all_ratings[all_ratings.mean() > 7].mean()
            if not high_performers.empty:
                alignment = min(1.0, 1 - (abs(ratings_given - high_performers).mean() / 9))
            else:
                alignment = min(1.0, 1 - (abs(ratings_given - all_ratings.mean()).mean() / 9))
            
            # Calculate influence (0-1 scale)
            G = nx.from_pandas_adjacency(self.ratings)
            centrality = nx.eigenvector_centrality(G, weight='weight')[person]
            avg_rating_received = peer_ratings.mean()
            influence = min(1.0, 0.7 * centrality + 0.3 * (avg_rating_received / 10))
            
            # Calculate fairness (0-1 scale)
            role_averages = {}
            for role in set(self.roles):
                role_members = [name for name, role_name in self.role_mapping.items() 
                              if role_name == role]
                if role_members:
                    role_averages[role] = self.ratings.loc[person, role_members].mean()
            
            if len(role_averages) >= 2:
                max_diff = max(abs(a - b) 
                             for i, a in role_averages.items() 
                             for j, b in role_averages.items() 
                             if i < j)
                fairness = min(1.0, np.exp(-max_diff/3))
            else:
                fairness = 1.0
            
            # Calculate final leadership score
            leadership_score = min(1.0, (
                0.25 * recognition +
                0.25 * alignment +
                0.25 * influence +
                0.25 * fairness
            ))
            
            return {
                'leadership_score': leadership_score,
                'recognition': recognition,
                'alignment': alignment,
                'influence': influence,
                'fairness': fairness
            }
            
        except Exception as e:
            print(f"Error calculating leadership index: {e}")
            return {
                'leadership_score': 0.5,
                'recognition': 0.5,
                'alignment': 0.5,
                'influence': 0.5,
                'fairness': 0.5
            }
    
    def calculate_base_metrics(self, person):
        """Calculate base metrics for a person"""
        person_ratings = self.ratings.loc[person]
        received_ratings = self.ratings[person]
        
        # Remove self-rating for given/received calculations
        given_ratings = person_ratings[person_ratings.index != person]
        received_ratings = received_ratings[received_ratings.index != person]
        
        metrics = {
            'Self Score': person_ratings[person],
            'Average Given': given_ratings.mean(),
            'Average Received': received_ratings.mean(),
            'SD Given': given_ratings.std(),
            'SD Received': received_ratings.std(),
            'Variance Given': given_ratings.var(),
            'Variance Received': received_ratings.var()
        }
        
        return metrics
    
    def calculate_adjusted_metrics(self, person):
        """Calculate adjusted metrics removing bias effects"""
        person_ratings = self.ratings.loc[person]
        received_ratings = self.ratings[person]
        
        # Remove self-rating
        given_ratings = person_ratings[person_ratings.index != person]
        received_ratings = received_ratings[received_ratings.index != person]
        
        # Calculate global mean and SD for normalization
        global_mean = self.ratings.mean().mean()
        global_std = self.ratings.std().std()
        
        # Adjust ratings by normalizing
        adjusted_given = (given_ratings - global_mean) / global_std
        adjusted_received = (received_ratings - global_mean) / global_std
        
        metrics = {
            'Adjusted Average Given': adjusted_given.mean(),
            'Adjusted Average Received': adjusted_received.mean(),
            'Adjusted SD Given': adjusted_given.std(),
            'Adjusted SD Received': adjusted_received.std(),
            'Adjusted Variance Given': adjusted_given.var(ddof=1),  # Using ddof=1 for sample variance
            'Adjusted Variance Received': adjusted_received.var(ddof=1)  # Using ddof=1 for sample variance
        }
        
        return metrics
    
    def analyze_personality_traits(self, person):
        """Analyze personality traits based on rating patterns"""
        base_metrics = self.calculate_base_metrics(person)
        adjusted_metrics = self.calculate_adjusted_metrics(person)
        
        traits = {}
        
        # Grading Style
        if base_metrics['SD Given'] < 0.5:
            traits['Rating Style'] = "Casual Rater (Low variation in ratings)"
        elif base_metrics['SD Given'] > 1.5:
            traits['Rating Style'] = "Serious Rater (High variation in ratings)"
        
        if base_metrics['Average Given'] > 8:
            traits['Rating Tendency'] = "Lenient Rater"
        elif base_metrics['Average Given'] < 6:
            traits['Rating Tendency'] = "Strict Rater"
        
        # Personality Insights
        self_bias = base_metrics['Self Score'] - base_metrics['Average Received']
        if abs(self_bias) < 0.5:
            traits['Self-Awareness'] = "High (Accurate self-perception)"
        elif self_bias > 1:
            traits['Self-Awareness'] = "Self-Inflating"
        else:
            traits['Self-Awareness'] = "Self-Critical"
        
        # Leadership and Influence
        influence_score = self.calculate_leadership_index(
            list(self.ratings.index).index(person) + 1
        )
        traits['Leadership Influence'] = f"{influence_score:.2f}/1.00"
        
        return traits 
    
    def analyze_rating_tendency(self, person):
        """Analyze if a person is a serious or non-serious rater based on multiple factors"""
        try:
            ratings_given = self.ratings.loc[person].drop(person)
            
            # Calculate metrics that indicate rating seriousness
            metrics = {
                # 1. Rating Spread/Variance (serious raters show thoughtful differentiation)
                'rating_spread': ratings_given.std(),
                
                # 2. Correlation with consensus
                'consensus_correlation': self._calculate_consensus_correlation(person),
                
                # 3. Rating Pattern Analysis
                'pattern_score': self._analyze_rating_pattern(ratings_given),
                
                # 4. Hierarchical Alignment
                'hierarchy_alignment': self._calculate_hierarchy_alignment(person),
                
                # 5. Self vs Others Rating Gap
                'self_rating_gap': abs(self.ratings.loc[person, person] - ratings_given.mean())
            }
            
            # Calculate seriousness score (0-1 scale)
            seriousness_score = (
                0.25 * self._normalize_spread(metrics['rating_spread']) +
                0.25 * metrics['consensus_correlation'] +
                0.20 * metrics['pattern_score'] +
                0.20 * metrics['hierarchy_alignment'] +
                0.10 * (1 - min(1, metrics['self_rating_gap']/5))
            )
            
            # Classify rater (threshold at 0.6 for 67/33 split)
            return "Serious" if seriousness_score >= 0.6 else "Non-serious"
            
        except Exception as e:
            print(f"Error analyzing rating tendency: {e}")
            return "Non-serious"  # Default to non-serious on error
    
    def _calculate_consensus_correlation(self, person):
        """Calculate how well a rater's ratings correlate with consensus"""
        try:
            ratings_given = self.ratings.loc[person].drop(person)
            consensus_ratings = self.ratings.drop(person).mean()
            common_rated = ratings_given.index.intersection(consensus_ratings.index)
            
            if len(common_rated) < 2:
                return 0.5
            
            correlation = ratings_given[common_rated].corr(consensus_ratings[common_rated])
            return max(0, (correlation + 1) / 2)  # Convert from [-1,1] to [0,1]
            
        except Exception as e:
            return 0.5
    
    def _analyze_rating_pattern(self, ratings):
        """Analyze rating pattern for signs of thoughtfulness"""
        try:
            # Check for signs of non-serious rating
            red_flags = 0
            
            # 1. Too many identical ratings
            value_counts = ratings.value_counts()
            if (value_counts.max() / len(ratings)) > 0.7:  # If >70% ratings are identical
                red_flags += 1
            
            # 2. Too many extreme ratings
            extreme_ratings = ratings[(ratings <= 2) | (ratings >= 9)].count()
            if (extreme_ratings / len(ratings)) > 0.5:  # If >50% ratings are extreme
                red_flags += 1
            
            # 3. Alternating pattern check
            diffs = ratings.diff().dropna()
            alternating = (diffs * diffs.shift(1) < 0).mean()  # Proportion of alternating ratings
            if alternating > 0.8:  # If >80% ratings alternate up/down
                red_flags += 1
            
            return max(0, 1 - (red_flags / 3))
            
        except Exception as e:
            return 0.5
    
    def _calculate_hierarchy_alignment(self, person):
        """Calculate how well ratings align with organizational hierarchy"""
        try:
            ratings_given = self.ratings.loc[person].drop(person)
            rater_level = self.level_mapping[person]
            
            # Calculate average ratings by level
            level_ratings = {}
            for rated_person in ratings_given.index:
                rated_level = self.level_mapping[rated_person]
                if rated_level not in level_ratings:
                    level_ratings[rated_level] = []
                level_ratings[rated_level].append(ratings_given[rated_person])
            
            level_averages = {k: np.mean(v) for k, v in level_ratings.items()}
            
            # Check if ratings generally respect hierarchy
            aligned_pairs = 0
            total_pairs = 0
            
            for level1 in level_averages:
                for level2 in level_averages:
                    if level1 < level2:
                        total_pairs += 1
                        if level_averages[level1] < level_averages[level2]:
                            aligned_pairs += 1
            
            return aligned_pairs / total_pairs if total_pairs > 0 else 0.5
            
        except Exception as e:
            return 0.5
    
    def _normalize_spread(self, std_dev):
        """Normalize standard deviation to 0-1 scale"""
        # A std_dev of 0 means all same ratings (bad), while 2-3 is good spread
        return min(1, std_dev / 2)
    
    def analyze_relationship(self, person1, person2):
        """Analyze the relationship between two people"""
        rating_given = self.ratings.loc[person1, person2]
        rating_received = self.ratings.loc[person2, person1]
        
        # Determine relationship type and sentiment
        if abs(rating_given - rating_received) <= 1:  # Within 1 point difference
            relationship_type = 'mutual'
            sentiment = 'friendly' if (rating_given + rating_received)/2 >= 6.5 else 'hostile'
            direction = None
        else:
            relationship_type = 'one-sided'
            if rating_given > rating_received:
                direction = 'outgoing'
                sentiment = 'hostile' if rating_received < 6.5 else 'friendly'
            else:
                direction = 'incoming'
                sentiment = 'hostile' if rating_given < 6.5 else 'friendly'
        
        return {
            'type': relationship_type,
            'sentiment': sentiment,
            'direction': direction,
            'rating_given': rating_given,
            'rating_received': rating_received
        }
    
    def calculate_performance_score(self, person):
        """Calculate overall performance score for a person"""
        try:
            # Get all ratings received by this person (excluding self-rating)
            ratings_received = self.ratings[person].drop(person)
            
            # Calculate weighted average based on rater levels
            weighted_sum = 0
            total_weights = 0
            
            for rater, rating in ratings_received.items():
                rater_level = self.level_mapping[rater]
                weight = rater_level / sum(self.level_mapping.values())  # Normalize weights
                weighted_sum += rating * weight
                total_weights += weight
            
            # Return weighted average on 1-10 scale
            return weighted_sum / total_weights if total_weights > 0 else 5.0
            
        except Exception as e:
            print(f"Error calculating performance score: {e}")
            return 5.0  # Return neutral score on error
    
    def _get_hierarchy_weights(self, num_levels):
        """Get hierarchy weights based on number of levels"""
        if num_levels == 5:
            return {5: 0.35, 4: 0.25, 3: 0.20, 2: 0.12, 1: 0.08}
        elif num_levels == 4:
            return {4: 0.50, 3: 0.30, 2: 0.15, 1: 0.05}
        elif num_levels == 3:
            return {3: 0.52, 2: 0.32, 1: 0.16}
        elif num_levels == 2:
            return {2: 0.60, 1: 0.40}
        else:
            return {1: 1.0}
    
    def detect_groupism(self, threshold=0.8):
        """Detect rating patterns that suggest groupism"""
        groupism_patterns = []
        
        for rater in self.ratings.index:
            ratings_given = self.ratings.loc[rater].drop(rater)
            
            # Find similar rating patterns
            for other_rater in self.ratings.index:
                if other_rater != rater:
                    try:
                        other_ratings = self.ratings.loc[other_rater].drop(other_rater)
                        
                        # Ensure we're comparing the same set of rated people
                        common_rated = ratings_given.index.intersection(other_ratings.index)
                        if len(common_rated) < 2:  # Need at least 2 common ratings
                            continue
                            
                        # Calculate correlation between rating patterns
                        correlation = ratings_given[common_rated].corr(other_ratings[common_rated])
                        
                        if correlation > threshold:
                            # Find who they rated differently from others
                            avg_ratings = self.ratings.mean()
                            common_low_rated = []
                            
                            for rated_person in common_rated:
                                if (ratings_given[rated_person] < avg_ratings[rated_person] - 1 and 
                                    other_ratings[rated_person] < avg_ratings[rated_person] - 1):
                                    common_low_rated.append(rated_person)
                            
                            if common_low_rated:
                                groupism_patterns.append({
                                    'Group_Members': [rater, other_rater],
                                    'Correlation': correlation,
                                    'Targeted_Members': common_low_rated
                                })
                    except Exception as e:
                        print(f"Error processing ratings for {rater} and {other_rater}: {str(e)}")
                        continue
        
        return groupism_patterns 
    
    def calculate_confidence_score(self, person):
        """Calculate confidence score based on self vs peer ratings"""
        try:
            # Get self rating and peer ratings
            self_rating = self.ratings.loc[person, person]
            peer_ratings = self.ratings[person].drop(person)
            peer_avg = peer_ratings.mean()
            
            # Calculate confidence score (0-1 scale)
            # Score closer to 0 means underconfident, closer to 1 means overconfident
            confidence_score = (self_rating - peer_avg + 10) / 20  # Normalize to 0-1 scale
            
            return min(1.0, max(0.0, confidence_score))  # Ensure score is between 0 and 1
        except Exception as e:
            print(f"Error calculating confidence score: {e}")
            return 0.5  # Return neutral score on error 
    
    def analyze_individual_performance(self, person):
        """Analyze individual performance metrics"""
        try:
            # Calculate confidence score
            confidence_score = self.calculate_confidence_score(person)
            
            # Calculate recognition (from ratings given)
            ratings_given = self.ratings.loc[person].drop(person)
            rating_range = ratings_given.max() - ratings_given.min()
            rating_std = ratings_given.std()
            recognition = min(1.0, (0.5 * (rating_range / 9) + 
                              0.5 * (1 - rating_std / 3)))
            
            # Calculate influence
            G = nx.from_pandas_adjacency(self.ratings)
            centrality = nx.eigenvector_centrality(G, weight='weight')[person]
            peer_ratings = self.ratings[person].drop(person)
            avg_rating_received = peer_ratings.mean()
            influence = min(1.0, 0.7 * centrality + 0.3 * (avg_rating_received / 10))
            
            # Calculate fairness in ratings
            role_averages = {}
            for role in set(self.roles):
                role_members = [name for name, role_name in self.role_mapping.items() 
                              if role_name == role]
                if role_members:
                    role_averages[role] = self.ratings.loc[person, role_members].mean()
            
            if len(role_averages) >= 2:
                max_diff = max(abs(a - b) 
                             for i, a in role_averages.items() 
                             for j, b in role_averages.items() 
                             if i < j)
                fairness = min(1.0, np.exp(-max_diff/3))
            else:
                fairness = 1.0
            
            # Calculate alignment with high performers
            all_ratings = self.ratings.drop(person)
            high_performers = all_ratings[all_ratings.mean() > 7].mean()
            if not high_performers.empty:
                alignment = min(1.0, 1 - (abs(ratings_given - high_performers).mean() / 9))
            else:
                alignment = min(1.0, 1 - (abs(ratings_given - all_ratings.mean()).mean() / 9))
            
            return {
                'confidence': confidence_score,
                'recognition': recognition,
                'influence': influence,
                'fairness': fairness,
                'alignment': alignment
            }
            
        except Exception as e:
            print(f"Error analyzing individual performance: {e}")
            return {
                'confidence': 0.5,
                'recognition': 0.5,
                'influence': 0.5,
                'fairness': 0.5,
                'alignment': 0.5
            } 
    
    def calculate_quadrant_metrics(self, person):
        """Calculate metrics for quadrant analysis"""
        try:
            # Get ratings
            ratings_given = self.ratings.loc[person].drop(person)  # Exclude self-rating
            ratings_received = self.ratings[person].drop(person)   # Exclude self-rating
            self_rating = self.ratings.loc[person, person]
            
            # Calculate metrics
            avg_received = ratings_received.mean()  # Average rating received from others
            
            # X-axis: Average Rating Received (high means others rate them well)
            performance_score = avg_received
            
            # Y-axis: Confidence Score (positive means overconfident)
            confidence_score = self_rating - avg_received
            
            return {
                'person': person,
                'performance_score': performance_score,
                'confidence_score': confidence_score,
                'role': self.get_person_role(person),
                'level': self.level_mapping[person]
            }
            
        except Exception as e:
            print(f"Error calculating quadrant metrics: {e}")
            return None 