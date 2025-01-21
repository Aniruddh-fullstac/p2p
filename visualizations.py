import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import streamlit as st
import random
import pandas as pd
import numpy as np

class PeerRatingVisualizer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        
    def create_heatmap(self):
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.analyzer.ratings, 
                   annot=True, 
                   fmt='.1f', 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Rating'})
        plt.title('Peer-to-Peer Ratings Heatmap')
        plt.xlabel('Rater')
        plt.ylabel('Rated Person')
        return plt
    
    def create_bias_chart(self):
        bias_data = self.analyzer.calculate_self_bias()
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=bias_data['Person'],
            y=bias_data['Bias'],
            name='Self-Perception Bias'
        ))
        
        fig.update_layout(
            title='Self-Perception Bias by Person',
            xaxis_title='Person',
            yaxis_title='Bias (Self Rating - Peer Rating)'
        )
        return fig
    
    def create_consistency_scatter(self):
        consistency_data = self.analyzer.calculate_rating_consistency()
        fig = px.scatter(
            consistency_data,
            x='Rating_Variance',
            y='Rating_Mean',
            text='Person',
            title='Rating Consistency vs Average Rating Given'
        )
        return fig
    
    def create_category_preference_plot(self):
        category_data = self.analyzer.analyze_category_bias()
        fig = px.bar(
            category_data,
            x='Rater_Category',
            y='Mean_Rating',
            color='Rated_Category',
            barmode='group',
            title='Average Ratings by Hierarchy Tier'
        )
        return fig
    
    def create_radar_chart(self, person_id):
        """Create radar chart for individual characteristics"""
        try:
            # Get actual name from index
            person = self.analyzer.ratings.index[person_id-1]
            
            # Calculate metrics
            self_bias = self.analyzer.calculate_self_bias()
            consistency = self.analyzer.calculate_rating_consistency()
            influence = self.analyzer.calculate_influence_scores()
            leadership = self.analyzer.calculate_leadership_index(person_id)
            
            # Calculate metrics with proper indexing
            metrics = {}
            
            # Leadership
            metrics['Leadership'] = leadership['leadership_score']
            metrics['Influence'] = influence[influence['Person'] == person].iloc[0]['Influence_Score']
            metrics['Self-Awareness'] = 1 - abs(self_bias[self_bias['Person'] == person].iloc[0]['Bias']) / self_bias['Bias'].abs().max()
            metrics['Consistency'] = 1 - (consistency[consistency['Person'] == person].iloc[0]['Rating_Variance'] / consistency['Rating_Variance'].max())
            metrics['Generosity'] = (consistency[consistency['Person'] == person].iloc[0]['Rating_Mean'] - 1) / 9
            metrics['Confidence'] = self.analyzer.calculate_confidence_score(person)
            
            # Create radar chart
            fig = go.Figure()
            
            # Add person's metrics
            fig.add_trace(go.Scatterpolar(
                r=list(metrics.values()),
                theta=list(metrics.keys()),
                fill='toself',
                name=person,
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=False,
                title=f'Characteristics Profile for {person}'
            )
            
            return fig, metrics
        except Exception as e:
            print(f"Error creating radar chart: {e}")
            return None, {}
    
    def create_reciprocity_chart(self, person_id):
        """Create visualization of reciprocal relationships"""
        try:
            # Convert person_id to name if needed
            if isinstance(person_id, int):
                person = self.analyzer.ratings.index[person_id-1]
            else:
                person = person_id
            
            reciprocity_data = self.analyzer.get_individual_reciprocity(person)
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Add ratings given
            fig.add_trace(go.Bar(
                name='Rating Given',
                x=reciprocity_data['Other_Person'],
                y=reciprocity_data['Rating_Given'],
                marker_color='blue'
            ))
            
            # Add ratings received
            fig.add_trace(go.Bar(
                name='Rating Received',
                x=reciprocity_data['Other_Person'],
                y=reciprocity_data['Rating_Received'],
                marker_color='red'
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Reciprocal Relationships for {person}',
                barmode='group',
                yaxis_title='Rating',
                xaxis_title='Other Participants',
                height=500
            )
            
            return fig
        except Exception as e:
            print(f"Error in create_reciprocity_chart: {str(e)}")
            return go.Figure()  # Return empty figure on error
    
    def create_relationship_network(self, person):
        """Create network visualization with improved quadrant layout"""
        try:
            G = nx.from_pandas_adjacency(self.analyzer.ratings)
            
            # Create figure with quadrant design
            fig = go.Figure()
            
            # Add quadrant background colors for better visibility
            fig.add_shape(type="rect", x0=0, y0=0.5, x1=0.5, y1=1,
                         fillcolor="rgba(211,211,211,0.2)", line=dict(width=0))  # Complex
            fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=1, y1=1,
                         fillcolor="rgba(144,238,144,0.2)", line=dict(width=0))  # Mutual Trust
            fig.add_shape(type="rect", x0=0, y0=0, x1=0.5, y1=0.5,
                         fillcolor="rgba(255,99,71,0.2)", line=dict(width=0))  # Hostile
            fig.add_shape(type="rect", x0=0.5, y0=0, x1=1, y1=0.5,
                         fillcolor="rgba(255,165,0,0.2)", line=dict(width=0))  # One-sided
            
            # Add quadrant labels with improved styling
            quadrant_labels = [
                dict(x=0.25, y=0.75, text="Complex<br>Relationships",
                     font=dict(size=14, color='rgba(0,0,0,0.7)', family='Arial')),
                dict(x=0.75, y=0.75, text="Mutual Trust<br>& Support",
                     font=dict(size=14, color='rgba(0,100,0,0.7)', family='Arial')),
                dict(x=0.25, y=0.25, text="Challenging<br>Relationships",
                     font=dict(size=14, color='rgba(139,0,0,0.7)', family='Arial')),
                dict(x=0.75, y=0.25, text="One-sided<br>Trust",
                     font=dict(size=14, color='rgba(255,140,0,0.7)', family='Arial'))
            ]
            
            for label in quadrant_labels:
                fig.add_annotation(
                    x=label['x'], y=label['y'],
                    text=label['text'],
                    showarrow=False,
                    font=label['font'],
                    bgcolor='rgba(255,255,255,0.8)',
                    borderpad=4,
                    borderwidth=1,
                    bordercolor='rgba(0,0,0,0.1)'
                )
            
            # Rest of the visualization code remains the same...
            
            # Update layout with improved styling
            fig.update_layout(
                showlegend=False,
                title=dict(
                    text=f'Relationship Network for {person}',
                    font=dict(size=16, family='Arial'),
                    y=0.95
                ),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
                plot_bgcolor='white',
                width=800,
                height=800,
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating relationship network: {e}")
            return go.Figure()
    
    def create_hierarchy_chart(self):
        """Create a hierarchical organization chart"""
        hierarchy = self.analyzer.hierarchy_structure
        
        # Create nodes and edges for the org chart
        nodes = []
        edges = []
        node_positions = {}  # Store positions of nodes
        
        # First create all nodes
        for role, info in hierarchy.items():
            nodes.append(
                dict(
                    id=role,
                    label=f"{role}<br>Level {info['level']}",
                    level=info['level']
                )
            )
            
            for managed_role in info['manages']:
                edges.append(
                    dict(
                        source=role,
                        target=managed_role
                    )
                )
        
        # Create the figure
        fig = go.Figure()
        
        # Layout the hierarchy
        levels = sorted(set(node['level'] for node in nodes))
        level_count = {level: 0 for level in levels}
        
        # Calculate and store positions
        for node in nodes:
            level = node['level']
            count = level_count[level]
            x = count * 2
            y = level * 2
            
            # Store position for this node
            node_positions[node['id']] = {'x': x, 'y': y}
            level_count[level] += 1
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        
        for node in nodes:
            pos = node_positions[node['id']]
            node_x.append(pos['x'])
            node_y.append(pos['y'])
            node_text.append(node['label'])
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            text=node_text,
            mode='markers+text',
            textposition='bottom center',
            marker=dict(
                size=30,
                color='lightblue',
                line=dict(color='black', width=1)
            ),
            name='Roles'
        )
        
        fig.add_trace(node_trace)
        
        # Add edges
        for edge in edges:
            source_pos = node_positions[edge['source']]
            target_pos = node_positions[edge['target']]
            
            fig.add_trace(go.Scatter(
                x=[source_pos['x'], target_pos['x']],
                y=[source_pos['y'], target_pos['y']],
                mode='lines',
                line=dict(color='gray', width=1),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title='Organizational Hierarchy',
            showlegend=False,
            hovermode='closest',
            plot_bgcolor='white',
            margin=dict(t=40, l=20, r=20, b=20),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            )
        )
        
        return fig 
    
    def create_personality_radar(self, person_metrics):
        """Create radar chart with confidence score and thresholds"""
        # Convert -10 to 10 scale to 0-1 scale
        confidence_score = (person_metrics['self_bias'] + 10) / 20
        
        # Add confidence thresholds
        underconfident_threshold = (8) / 20  # -2 on original scale
        overconfident_threshold = (12) / 20  # +2 on original scale
        
        # Add to radar chart
        categories = ['Confidence', 'Recognition', 'Influence', 'Fairness', 'Alignment']
        values = [confidence_score, person_metrics['recognition'], 
                 person_metrics['influence'], person_metrics['fairness'],
                 person_metrics['alignment']]
        
        fig = go.Figure()
        
        # Add threshold circles
        fig.add_trace(go.Scatterpolar(
            r=[underconfident_threshold]*len(categories),
            theta=categories,
            fill='toself',
            name='Underconfident Threshold'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[overconfident_threshold]*len(categories),
            theta=categories,
            fill='toself',
            name='Overconfident Threshold'
        ))
        
        def create_performance_radar(self, person):
            """Create radar chart for performance metrics"""
            try:
                # Get performance metrics
                metrics = self.analyzer.analyze_individual_performance(person)
                
                # Create radar chart
                categories = list(metrics.keys())
                values = list(metrics.values())
                
                fig = go.Figure()
                
                # Add confidence thresholds
                underconfident_threshold = 0.4
                overconfident_threshold = 0.6
                
                # Add underconfident threshold circle
                fig.add_trace(go.Scatterpolar(
                    r=[underconfident_threshold] * len(categories),
                    theta=categories,
                    fill='none',
                    mode='lines',
                    line=dict(color='rgba(255,0,0,0.2)', dash='dot'),
                    name='Underconfident Threshold'
                ))
                
                # Add overconfident threshold circle
                fig.add_trace(go.Scatterpolar(
                    r=[overconfident_threshold] * len(categories),
                    theta=categories,
                    fill='none',
                    mode='lines',
                    line=dict(color='rgba(255,165,0,0.2)', dash='dot'),
                    name='Overconfident Threshold'
                ))
                
                # Add performance metrics
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=person,
                    line=dict(color='blue')
                ))
                
                # Update layout
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    showlegend=True,
                    title=f'Performance Profile for {person}'
                )
                
                return fig
                
            except Exception as e:
                print(f"Error creating performance radar: {e}")
                # Return empty chart with default values
                categories = ['Confidence', 'Recognition', 'Influence', 'Fairness', 'Alignment']
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=[0.5] * len(categories),
                    theta=categories,
                    fill='toself',
                    name='Error'
                ))
                fig.update_layout(
                    showlegend=False,
                    title='Error Loading Profile'
                )
                return fig
    
    def create_individual_statistics(self, person):
        """Create detailed individual statistics"""
        try:
            # Get various metrics
            self_bias = self.analyzer.calculate_self_bias()
            person_bias = self_bias[self_bias['Person'] == person].iloc[0]
            
            consistency = self.analyzer.calculate_rating_consistency()
            person_consistency = consistency[consistency['Person'] == person].iloc[0]
            
            ratings_given = self.analyzer.ratings.loc[person].drop(person)
            ratings_received = self.analyzer.ratings[person].drop(person)
            
            # Calculate statistics
            stats = {
                'Average Rating Given': ratings_given.mean(),
                'Average Rating Received': ratings_received.mean(),
                'Rating Range Given': ratings_given.max() - ratings_given.min(),
                'Rating Range Received': ratings_received.max() - ratings_received.min(),
                'Self Rating': self.analyzer.ratings.loc[person, person],
                'Self-Perception Bias': person_bias['Bias'],
                'Rating Consistency': 1 - (person_consistency['Rating_Variance'] / 
                                        consistency['Rating_Variance'].max()),
                'Highest Rating Given To': ratings_given.idxmax(),
                'Lowest Rating Given To': ratings_given.idxmin(),
                'Highest Rating From': ratings_received.idxmax(),
                'Lowest Rating From': ratings_received.idxmin()
            }
            
            return stats
        except Exception as e:
            print(f"Error creating individual statistics: {e}")
            return {}
    
    def create_reciprocation_table(self, person):
        """Create detailed reciprocation analysis table"""
        try:
            ratings_given = self.analyzer.ratings.loc[person]
            ratings_received = self.analyzer.ratings[person]
            
            data = []
            for other_person in self.analyzer.ratings.index:
                if other_person != person:
                    given = ratings_given[other_person]
                    received = ratings_received[other_person]
                    difference = given - received
                    reciprocity = "High" if abs(difference) < 1 else "Medium" if abs(difference) < 2 else "Low"
                    
                    data.append({
                        'Person': other_person,
                        'Rating Given': given,
                        'Rating Received': received,
                        'Difference': difference,
                        'Reciprocity': reciprocity
                    })
            
            return pd.DataFrame(data).sort_values('Difference', ascending=False)
            
        except Exception as e:
            print(f"Error creating reciprocation table: {e}")
            return pd.DataFrame()
    
    def get_relationship_summary(self, person):
        """Get summary of relationships for a person"""
        try:
            relationships = {
                'mutual_trust': 0,
                'one_sided_given': 0,
                'one_sided_received': 0,
                'complex': 0
            }
            
            for other_person in self.analyzer.ratings.index:
                if other_person != person:
                    rating_given = self.analyzer.ratings.loc[person, other_person]
                    rating_received = self.analyzer.ratings.loc[other_person, person]
                    
                    # Classify relationship
                    if rating_given >= 7 and rating_received >= 7:
                        relationships['mutual_trust'] += 1
                    elif rating_given >= 7 and rating_received < 4:
                        relationships['one_sided_given'] += 1
                    elif rating_received >= 7 and rating_given < 4:
                        relationships['one_sided_received'] += 1
                    else:
                        relationships['complex'] += 1
            
            # Add percentages
            total = sum(relationships.values())
            if total > 0:
                for key in relationships:
                    relationships[f"{key}_pct"] = f"{(relationships[key] / total) * 100:.1f}%"
            
            # Add relationship quality indicators
            relationships['primary_style'] = max(relationships, key=relationships.get)
            relationships['balance_score'] = relationships['mutual_trust'] / total if total > 0 else 0
            
            return relationships
            
        except Exception as e:
            print(f"Error getting relationship summary: {e}")
            return {
                'mutual_trust': 0,
                'one_sided_given': 0,
                'one_sided_received': 0,
                'complex': 0,
                'mutual_trust_pct': '0%',
                'one_sided_given_pct': '0%',
                'one_sided_received_pct': '0%',
                'complex_pct': '0%',
                'primary_style': 'unknown',
                'balance_score': 0
            }
    
    def get_detailed_metrics(self, person):
        """Calculate detailed metrics for a person"""
        try:
            # Calculate leadership score
            ratings_received = self.analyzer.ratings[person]
            leadership_score = ratings_received.mean()
            
            # Calculate rating style
            person_ratings = self.analyzer.ratings.loc[person]
            avg_rating_given = person_ratings.mean()
            overall_mean = self.analyzer.ratings.values.mean()
            
            if avg_rating_given > overall_mean + 0.5:
                rating_style = "Lenient"
            elif avg_rating_given < overall_mean - 0.5:
                rating_style = "Strict"
            else:
                rating_style = "Balanced"
            
            # Calculate self-perception
            self_rating = self.analyzer.ratings.loc[person, person]
            others_rating = ratings_received[ratings_received.index != person].mean()
            
            if self_rating > others_rating + 1:
                self_perception = "Self-Confident"
            elif self_rating < others_rating - 1:
                self_perception = "Self-Critical"
            else:
                self_perception = "Realistic"
            
            # Generate insights
            insights = []
            if leadership_score > 7:
                insights.append("Strong leadership presence with high team trust")
            elif leadership_score < 5:
                insights.append("Opportunity to improve leadership effectiveness")
            
            if rating_style == "Balanced":
                insights.append("Shows balanced judgment in evaluating others")
            else:
                insights.append(f"Tends to be {rating_style.lower()} in rating others")
            
            if self_perception == "Realistic":
                insights.append("Has realistic self-awareness")
            else:
                insights.append(f"Shows {self_perception.lower()} tendencies")
            
            return {
                'leadership_score': leadership_score,
                'rating_style': rating_style,
                'self_perception': self_perception,
                'insights': "\n".join(f"• {insight}" for insight in insights)
            }
            
        except Exception as e:
            print(f"Error calculating detailed metrics: {e}")
            return {
                'leadership_score': 0,
                'rating_style': 'Unknown',
                'self_perception': 'Unknown',
                'insights': 'Unable to calculate metrics'
            }
    
    def get_hierarchical_bias(self, person):
        """Calculate hierarchical bias for a person"""
        try:
            person_role = self.analyzer.get_person_role(person)
            person_level = self.analyzer.level_mapping[person]
            
            # Calculate average ratings given to different levels
            ratings_by_level = {}
            for other_person in self.analyzer.ratings.index:
                if other_person != person:
                    other_level = self.analyzer.level_mapping[other_person]
                    rating = self.analyzer.ratings.loc[person, other_person]
                    if other_level not in ratings_by_level:
                        ratings_by_level[other_level] = []
                    ratings_by_level[other_level].append(rating)
            
            # Calculate average for each level and overall average
            level_averages = {level: np.mean(ratings) for level, ratings in ratings_by_level.items()}
            overall_avg = np.mean(list(level_averages.values()))
            
            # Detect bias with more nuanced thresholds
            bias_insights = []
            for level, avg in level_averages.items():
                level_diff = avg - overall_avg
                if level > person_level and level_diff > 1:
                    bias_insights.append(f"Shows favorable bias towards superiors (Level {level}: {avg:.1f} vs Overall: {overall_avg:.1f})")
                elif level < person_level and level_diff < -1:
                    bias_insights.append(f"Shows unfavorable bias towards subordinates (Level {level}: {avg:.1f} vs Overall: {overall_avg:.1f})")
                elif level == person_level and abs(level_diff) > 1:
                    if level_diff > 0:
                        bias_insights.append(f"Shows favorable bias towards peers (Level {level}: {avg:.1f} vs Overall: {overall_avg:.1f})")
                    else:
                        bias_insights.append(f"Shows unfavorable bias towards peers (Level {level}: {avg:.1f} vs Overall: {overall_avg:.1f})")
            
            return bias_insights if bias_insights else ["No significant hierarchical bias detected"]
        
        except Exception as e:
            print(f"Error calculating hierarchical bias: {e}")
            return ["Unable to calculate hierarchical bias"]
    
    def get_reporting_structure(self, person):
        """Get reporting structure for a person"""
        try:
            chain = self.analyzer.get_reporting_chain(person)
            return {
                'Reports To': chain['superiors'],
                'Peers': chain['peers'],
                'Manages': chain['subordinates']
            }
        except Exception as e:
            print(f"Error getting reporting structure: {e}")
            return {'Reports To': [], 'Peers': [], 'Manages': []}
    
    def create_reciprocal_chart(self, person):
        """Create reciprocal relationships bar chart and table"""
        try:
            # Get all relationships except self
            others = [p for p in self.analyzer.ratings.index if p != person]
            ratings_given = [self.analyzer.ratings.loc[person, other] for other in others]
            ratings_received = [self.analyzer.ratings.loc[other, person] for other in others]
            
            # Create bar chart
            fig = go.Figure()
            
            # Add bars for ratings given
            fig.add_trace(go.Bar(
                name='Rating Given',
                x=others,
                y=ratings_given,
                marker_color='blue'
            ))
            
            # Add bars for ratings received
            fig.add_trace(go.Bar(
                name='Rating Received',
                x=others,
                y=ratings_received,
                marker_color='red'
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Reciprocal Relationships for {person}',
                xaxis_title='Other Participants',
                yaxis_title='Rating',
                barmode='group',
                yaxis_range=[0, 8],
                height=400
            )
            
            # Create relationship table data
            table_data = []
            for other, given, received in zip(others, ratings_given, ratings_received):
                difference = abs(given - received)
                reciprocity_score = 1 - (difference / 10)
                
                # Determine relationship type
                relationship = self.analyzer.analyze_relationship(person, other)
                
                # Format relationship type with tags
                if relationship['type'] == 'mutual':
                    rel_type = f"🤝 Mutual {'💚 Friendly' if relationship['sentiment'] == 'friendly' else '❌ Hostile'}"
                else:
                    direction = "→" if relationship['direction'] == 'outgoing' else "←"
                    rel_type = f"↔️ One-sided {direction} {'💚 Friendly' if relationship['sentiment'] == 'friendly' else '❌ Hostile'}"
                
                table_data.append({
                    'Other_Person': other,
                    'Rating_Given': f"{given:.1f}",
                    'Rating_Received': f"{received:.1f}",
                    'Difference': f"{difference:.1f}",
                    'Reciprocity_Score': f"{reciprocity_score:.4f}",
                    'Relationship_Type': rel_type
                })
            
            # Sort by reciprocity score
            table_df = pd.DataFrame(table_data).sort_values('Difference')
            
            return fig, table_df
        
        except Exception as e:
            print(f"Error creating reciprocal chart: {e}")
            return None, pd.DataFrame()
    
    def create_network_visualization(self, person, selected_nodes=None):
        """Create network visualization with relationship data"""
        try:
            # Create figure
            fig = go.Figure()
            
            # Add quadrant background colors
            fig.add_shape(type="rect", x0=0, y0=0.5, x1=0.5, y1=1,
                         fillcolor="rgba(255,165,0,0.2)", line=dict(width=0))  # One-sided (From Other)
            fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=1, y1=1,
                         fillcolor="rgba(144,238,144,0.2)", line=dict(width=0))  # Mutual Friendly
            fig.add_shape(type="rect", x0=0, y0=0, x1=0.5, y1=0.5,
                         fillcolor="rgba(255,99,71,0.2)", line=dict(width=0))  # Mutual Hostile
            fig.add_shape(type="rect", x0=0.5, y0=0, x1=1, y1=0.5,
                         fillcolor="rgba(255,165,0,0.2)", line=dict(width=0))  # One-sided (From Self)
            
            # Add quadrant labels
            quadrant_labels = [
                {'x': 0.25, 'y': 0.75, 'text': "One-sided<br>(From Other)"},
                {'x': 0.75, 'y': 0.75, 'text': "Mutually<br>Friendly"},
                {'x': 0.25, 'y': 0.25, 'text': "Mutually<br>Hostile"},
                {'x': 0.75, 'y': 0.25, 'text': "One-sided<br>(From Self)"}
            ]
            
            for label in quadrant_labels:
                fig.add_annotation(
                    x=label['x'], y=label['y'],
                    text=label['text'],
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor='rgba(255,255,255,0.8)'
                )
            
            # Get relationship data
            nodes_data = {}
            others = [p for p in self.analyzer.ratings.index if p != person]
            
            for other in others:
                rating_given = self.analyzer.ratings.loc[person, other]
                rating_received = self.analyzer.ratings.loc[other, person]
                
                # Calculate difference
                diff = rating_given - rating_received
                
                # Determine quadrant and color based on ratings and difference
                if abs(diff) <= 2.5:  # Within 2.5 points
                    if rating_given >= 7 and rating_received >= 7:
                        quadrant = 'Mutually Friendly'
                        color = '#2ecc71'  # Green
                    elif rating_given < 7 and rating_received < 7:
                        quadrant = 'Mutually Hostile'
                        color = '#e74c3c'  # Red
                    else:
                        # If one rating is high and one is low but difference is small,
                        # treat as one-sided based on who gave higher rating
                        if rating_given > rating_received:
                            quadrant = 'One-sided (From Self)'
                            color = '#f1c40f'  # Yellow
                        else:
                            quadrant = 'One-sided (From Other)'
                            color = '#f1c40f'  # Yellow
                else:  # Difference > 2.5 points
                    if rating_given > rating_received:
                        quadrant = 'One-sided (From Self)'
                        color = '#f1c40f'  # Yellow
                    else:
                        quadrant = 'One-sided (From Other)'
                        color = '#f1c40f'  # Yellow
                
                # Only include node if it's in selected_nodes (if specified)
                if selected_nodes is None or other in selected_nodes:
                    nodes_data[other] = {
                        'x': rating_given / 10,  # Normalize for plotting
                        'y': rating_received / 10,
                        'rating_given': rating_given,
                        'rating_received': rating_received,
                        'quadrant': quadrant,
                        'color': color,
                        'difference': diff
                    }
            
            # Create relationship table first
            relationship_table = self._create_relationship_table(nodes_data)
            
            # Add nodes and edges only for selected nodes
            self._add_nodes_and_edges(fig, person, nodes_data)
            
            return fig, relationship_table
            
        except Exception as e:
            print(f"Error creating network visualization: {e}")
            return go.Figure(), pd.DataFrame()

    def _calculate_relationship_positions(self, person):
        """Calculate positions and relationships for all nodes"""
        nodes_data = {}
        others = [p for p in self.analyzer.ratings.index if p != person]
        
        for other in others:
            rating_given = self.analyzer.ratings.loc[person, other]
            rating_received = self.analyzer.ratings.loc[other, person]
            
            # Normalize ratings to 0-1 scale for positioning
            x = rating_given / 10
            y = rating_received / 10
            
            # Determine quadrant and relationship type
            if rating_given >= 7 and rating_received >= 7:
                quadrant = 'Strong Mutual Trust'
                color = '#27ae60'
            elif rating_given < 4 and rating_received < 4:
                quadrant = 'Challenging Dynamic'
                color = '#e74c3c'
            elif abs(rating_given - rating_received) >= 3:
                quadrant = 'Complex Relationship'
                color = '#f1c40f'
            else:
                quadrant = 'One-Sided Trust'
                color = '#3498db'
            
            nodes_data[other] = {
                'x': x,
                'y': y,
                'quadrant': quadrant,
                'color': color,
                'rating_given': rating_given,
                'rating_received': rating_received
            }
        
        return nodes_data

    def _create_relationship_table(self, nodes_data):
        """Create a table of relationships with quadrant information"""
        rows = []
        for person, data in nodes_data.items():
            rows.append({
                'Person': person,
                'Rating Given': data['rating_given'],
                'Rating Received': data['rating_received'],
                'Difference': data['difference'],
                'Quadrant': data['quadrant']
            })
        return pd.DataFrame(rows)

    def _add_nodes_and_edges(self, fig, person, nodes_data):
        """Add nodes and edges to the figure"""
        # Add central node
        fig.add_trace(go.Scatter(
            x=[0.5],
            y=[0.5],
            mode='markers+text',
            marker=dict(size=40, color='#2c3e50'),
            text=[f"👤 {person}<br>{self.analyzer.get_person_role(person)}"],
            textposition="top center",
            hoverinfo='text',
            showlegend=False
        ))
        
        # Add other nodes and edges
        for other, data in nodes_data.items():
            if other != person:
                # Add node
                fig.add_trace(go.Scatter(
                    x=[data['x']],
                    y=[data['y']],
                    mode='markers+text',
                    marker=dict(size=30, color=data['color']),
                    text=[f"👤 {other}<br>{self.analyzer.get_person_role(other)}"],
                    textposition="top center",
                    hoverinfo='text',
                    showlegend=False
                ))
                
                # Add edge
                fig.add_trace(go.Scatter(
                    x=[0.5, data['x']],
                    y=[0.5, data['y']],
                    mode='lines',
                    line=dict(color=data['color'], width=2),
                    hovertext=f"Given: {data['rating_given']:.1f}<br>Received: {data['rating_received']:.1f}",
                    hoverinfo='text',
                    showlegend=False
                ))

    def create_relationship_summary(self, person):
        """Create a comprehensive relationship summary visualization"""
        try:
            # Calculate rating tendency (strict/lenient)
            ratings_given = self.analyzer.ratings.loc[person].drop(person)
            others_ratings = self.analyzer.ratings.drop(person).mean().mean()
            person_avg = ratings_given.mean()
            
            # Normalize both averages to 0-1 scale and compare
            normalized_diff = (person_avg - others_ratings)
            
            if normalized_diff < -2:
                rater_style = "Strict Rater"
                style_color = "red"
            elif normalized_diff > 2:
                rater_style = "Lenient Rater"
                style_color = "green"
            else:
                rater_style = "Neutral Rater"
                style_color = "gray"
            
            # Get relationships data
            relationships = []
            others = [p for p in self.analyzer.ratings.index if p != person]
            
            for other in others:
                rating_given = self.analyzer.ratings.loc[person, other]
                rating_received = self.analyzer.ratings.loc[other, person]
                diff = rating_given - rating_received
                
                # Determine relationship type
                if abs(diff) <= 2.5:
                    if rating_given >= 7 and rating_received >= 7:
                        category = 'Mutually Friendly'
                        color = '#2ecc71'
                        symbol = '🤝'
                    elif rating_given < 7 and rating_received < 7:
                        category = 'Mutually Hostile'
                        color = '#e74c3c'
                        symbol = '⚠️'
                    else:
                        category = 'Mixed'
                        color = '#f1c40f'
                        symbol = '❓'
                else:
                    if rating_given > rating_received:
                        category = 'One-sided (From Self)'
                        color = '#3498db'
                        symbol = '➡️'
                    else:
                        category = 'One-sided (From Other)'
                        color = '#9b59b6'
                        symbol = '⬅️'
                
                relationships.append({
                    'Person': other,
                    'Role': self.analyzer.get_person_role(other),
                    'Given': rating_given,
                    'Received': rating_received,
                    'Difference': diff,
                    'Category': category,
                    'Color': color,
                    'Symbol': symbol
                })
            
            # Create figure
            fig = go.Figure()
            
            # Add scatter plot points
            for rel in relationships:
                fig.add_trace(go.Scatter(
                    x=[rel['Given']],
                    y=[rel['Received']],
                    mode='markers+text',
                    marker=dict(size=40, color=rel['Color'], line=dict(width=2, color='white')),
                    text=[rel['Symbol']],
                    name=rel['Person'],
                    customdata=[[rel['Person'], rel['Role'], rel['Category'], rel['Difference']]],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>" +
                        "Role: %{customdata[1]}<br>" +
                        "Category: %{customdata[2]}<br>" +
                        "Rating Given: %{x:.1f}<br>" +
                        "Rating Received: %{y:.1f}<br>" +
                        "Difference: %{customdata[3]:.1f}<br>" +
                        "<extra></extra>"
                    )
                ))
            
            # Add diagonal line
            fig.add_trace(go.Scatter(
                x=[0, 10], y=[0, 10],
                mode='lines', line=dict(color='gray', dash='dash'),
                showlegend=False, hoverinfo='skip'
            ))
            
            # Add overconfident/underconfident regions
            fig.add_shape(
                type="rect",
                x0=0, y0=0, x1=10, y1=10,
                line=dict(width=0),
                fillcolor="rgba(255,255,255,0)"
            )
            
            # Add annotations for confidence regions
            fig.add_annotation(
                x=8, y=2,
                text="Overconfident Region",
                showarrow=False,
                font=dict(size=10, color="red")
            )
            fig.add_annotation(
                x=2, y=8,
                text="Underconfident Region",
                showarrow=False,
                font=dict(size=10, color="blue")
            )
            
            # Update layout
            fig.update_layout(
                title=f'Relationship Summary for {person}<br><sub>{rater_style} (Avg: {person_avg:.1f} vs Global: {others_ratings:.1f})</sub>',
                xaxis=dict(title='Rating Given', range=[0, 10], gridcolor='lightgray'),
                yaxis=dict(title='Rating Received', range=[0, 10], gridcolor='lightgray'),
                plot_bgcolor='white',
                showlegend=True,
                legend_title='Team Members',
                height=600,
                width=800
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating relationship summary: {e}")
            return go.Figure()

    def get_rater_style(self, person):
        """Get detailed rater style analysis"""
        try:
            ratings_given = self.analyzer.ratings.loc[person].drop(person)
            others_ratings = self.analyzer.ratings.drop(person).mean().mean()
            person_avg = ratings_given.mean()
            person_std = ratings_given.std()
            
            # Calculate normalized difference
            normalized_diff = (person_avg - others_ratings)
            
            # Determine rating style
            if normalized_diff < -1.5:
                style = {
                    'category': 'Strict Rater',
                    'color': '#e74c3c',
                    'description': f'Tends to give lower ratings than average (Personal avg: {person_avg:.1f} vs Global: {others_ratings:.1f})'
                }
            elif normalized_diff > 1.5:
                style = {
                    'category': 'Lenient Rater',
                    'color': '#2ecc71',
                    'description': f'Tends to give higher ratings than average (Personal avg: {person_avg:.1f} vs Global: {others_ratings:.1f})'
                }
            else:
                style = {
                    'category': 'Neutral Rater',
                    'color': '#3498db',
                    'description': f'Gives ratings close to average (Personal avg: {person_avg:.1f} vs Global: {others_ratings:.1f})'
                }
            
            # Add consistency information
            if person_std > 2:
                style['consistency'] = 'High variability in ratings'
            else:
                style['consistency'] = 'Consistent rating pattern'
            
            return style
            
        except Exception as e:
            print(f"Error analyzing rater style: {e}")
            return {
                'category': 'Unknown',
                'color': 'gray',
                'description': 'Unable to determine rating style',
                'consistency': 'Unknown'
            }

    def create_individual_radar(self, person):
        """Create radar chart for individual analysis"""
        metrics = self.analyzer.analyze_individual_performance(person)
        
        # Add confidence metrics
        self_rating = self.analyzer.ratings.loc[person, person]
        avg_received = self.analyzer.ratings[person].drop(person).mean()
        confidence_diff = self_rating - avg_received
        
        if confidence_diff > 2:
            metrics['self_perception'] = 'Overconfident'
        elif confidence_diff < -2:
            metrics['self_perception'] = 'Underconfident'
        else:
            metrics['self_perception'] = 'Balanced'
        
        categories = ['Recognition', 'Influence', 'Fairness', 'Alignment']
        values = [metrics[k.lower()] for k in categories]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=f'{person}\n({metrics["self_perception"]})'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=f"Individual Analysis for {person}"
        )
        
        return fig

    def create_quadrant_analysis(self):
        """Create quadrant analysis plot"""
        # Collect data for all persons
        quadrant_data = []
        for person in self.analyzer.ratings.index:
            metrics = self.analyzer.calculate_quadrant_metrics(person)
            if metrics:
                quadrant_data.append(metrics)
        
        df = pd.DataFrame(quadrant_data)
        
        # Calculate the median performance score as threshold
        performance_threshold = df['performance_score'].median()
        
        # Create quadrant plot
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=df['performance_score'],
            y=df['confidence_score'],
            mode='markers+text',
            text=df['person'],
            textposition="top center",
            marker=dict(
                size=12,
                color=df['level'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Hierarchy Level")
            ),
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "Avg Rating Received: %{x:.2f}<br>" +
                "Confidence Score: %{y:.2f}<br>" +
                "Role: %{customdata[0]}<br>" +
                "<extra></extra>"
            ),
            customdata=df[['role']].values
        ))
        
        # Add quadrant lines at median performance and zero confidence
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=performance_threshold, line_dash="dash", line_color="gray")
        
        # Add quadrant labels with correct descriptions
        fig.add_annotation(
            x=performance_threshold + 1.5, y=1.5,
            text="Requires Promotion<br>(High Performance, Confident)",  # Top Right
            showarrow=False, font=dict(size=12)
        )
        fig.add_annotation(
            x=performance_threshold + 1.5, y=-1.5,
            text="Can Consider<br>(High Performance, Underconfident)",  # Bottom Right
            showarrow=False, font=dict(size=12)
        )
        fig.add_annotation(
            x=performance_threshold - 1.5, y=1.5,
            text="Required Firing<br>(Low Performance, Overconfident)",  # Top Left
            showarrow=False, font=dict(size=12)
        )
        fig.add_annotation(
            x=performance_threshold - 1.5, y=-1.5,
            text="Required Training<br>(Low Performance, Underconfident)",  # Bottom Left
            showarrow=False, font=dict(size=12)
        )
        
        # Update layout
        fig.update_layout(
            title="Employee Performance-Confidence Analysis",
            xaxis_title="Average Rating Received (Performance)",
            yaxis_title="Confidence Score (Self - Received)",
            xaxis=dict(range=[4, 10]),  # Adjusted for rating scale
            yaxis=dict(range=[-3, 3]),
            showlegend=False
        )
        
        return fig
