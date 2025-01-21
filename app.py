import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_generator import generate_peer_ratings
from analysis import PeerRatingAnalyzer
from visualizations import PeerRatingVisualizer
import json
import numpy as np

def main():
    st.title('Peer Rating Analysis Dashboard')
    
    # Add download sample file button and format explanation
    with st.expander("CSV Format Explanation"):
        st.markdown("""
        ### Required CSV Format:
        - First column should be named 'Name' containing employee names
        - Middle columns should be the same names as ratings given by each person
        - Last two columns should be 'Role' and 'Level'
        - Ratings should be on a scale of 1-10
        
        Example format:
        ```
        Name,Person1,Person2,...,PersonN,Role,Level
        Person1,5.0,8.0,...,7.0,Manager,3
        Person2,7.0,6.0,...,8.0,Staff,2
        ...
        ```
        """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload your peer ratings CSV file", type=['csv'])
    
    if uploaded_file is None:
        st.warning("Please upload a CSV file to begin analysis")
        st.stop()
        
    try:
        ratings_df = pd.read_csv(uploaded_file, index_col=0)
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        st.stop()
    
    # Initialize analyzer with hierarchy config
    try:
        analyzer = PeerRatingAnalyzer(ratings_df)
        visualizer = PeerRatingVisualizer(analyzer)
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.stop()
        
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Select Analysis View",
        ["Overview", "Individual Analysis", "Hierarchical Analysis", "Performance Analysis"]
    )
    
    if page == "Overview":
        st.header("Overview Analysis")
        
        # Display heatmap
        st.subheader("Peer-to-Peer Ratings Heatmap")
        st.pyplot(visualizer.create_heatmap())
        
        # Display bias chart
        st.subheader("Self-Perception Bias")
        st.plotly_chart(visualizer.create_bias_chart())
        
        # Display consistency scatter plot
        st.subheader("Rating Consistency Analysis")
        st.plotly_chart(visualizer.create_consistency_scatter())
        
    elif page == "Individual Analysis":
        st.header("Individual Analysis")
        
        # Get list of people
        people = list(analyzer.ratings.index)
        selected_person = st.selectbox("Select Person", options=people, index=0)
        
        # Create tabs
        tabs = st.tabs(["Profile", "Relationship Analysis", "Network"])
        
        with tabs[0]:
            # 1. Spider Chart
            radar_fig, metrics = visualizer.create_radar_chart(people.index(selected_person) + 1)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)
            
            # 2. Hierarchical Bias
            st.subheader("Rating Style Analysis")
            rater_style = visualizer.get_rater_style(selected_person)
            
            # Display rating style with color
            st.markdown(f"""
            <div style='padding: 10px; border-radius: 5px; background-color: {rater_style['color']}20;'>
                <h3 style='color: {rater_style['color']};'>{rater_style['category']}</h3>
                <p>{rater_style['description']}</p>
                <p><i>{rater_style['consistency']}</i></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display hierarchical bias
            st.subheader("Hierarchical Bias Analysis")
            bias_insights = visualizer.get_hierarchical_bias(selected_person)
            for insight in bias_insights:
                if "favorable" in insight.lower():
                    st.info(insight)
                elif "unfavorable" in insight.lower():
                    st.warning(insight)
                else:
                    st.success(insight)
            
            # 3. Reporting Structure
            st.subheader("Organizational Structure")
            structure = visualizer.get_reporting_structure(selected_person)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Reports To")
                for superior in structure['Reports To']:
                    st.write(f"• {superior}")
                
                st.markdown("##### Peers")
                for peer in structure['Peers']:
                    st.write(f"• {peer}")
            
            with col2:
                st.markdown("##### Manages")
                for subordinate in structure['Manages']:
                    st.write(f"• {subordinate}")
            
            # 4. Individual Statistics
            st.subheader("Individual Statistics")
            self_rating = analyzer.ratings.loc[selected_person, selected_person]
            peer_rating = analyzer.ratings[selected_person].mean()
            bias = self_rating - peer_rating
            
            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric("Self Rating", f"{self_rating:.2f}")
            with col4:
                st.metric("Peer Rating", f"{peer_rating:.2f}")
            with col5:
                st.metric("Bias", f"{bias:.2f}")
            
            # 5. Detailed Rating Analysis
            st.subheader("Detailed Rating Analysis")
            with st.expander("Basic Metrics"):
                # Get detailed metrics
                ratings_given = analyzer.ratings.loc[selected_person].drop(selected_person)
                ratings_received = analyzer.ratings[selected_person].drop(selected_person)
                
                col6, col7, col8 = st.columns(3)
                with col6:
                    st.metric("Self Score", f"{self_rating:.2f}")
                    st.metric("Average Given", f"{ratings_given.mean():.2f}")
                    st.metric("SD Given", f"{ratings_given.std():.2f}")
                with col7:
                    st.metric("Average Received", f"{ratings_received.mean():.2f}")
                    st.metric("SD Received", f"{ratings_received.std():.2f}")
                with col8:
                    st.metric("Variance Given", f"{ratings_given.var():.2f}")
                    st.metric("Variance Received", f"{ratings_received.var():.2f}")
            
            with st.expander("Adjusted Metrics (Normalized)"):
                # Calculate normalized metrics
                overall_mean = analyzer.ratings.values.mean()
                overall_std = analyzer.ratings.values.std()
                
                adj_given = (ratings_given.mean() - overall_mean) / overall_std
                adj_received = (ratings_received.mean() - overall_mean) / overall_std
                
                col9, col10 = st.columns(2)
                with col9:
                    st.metric("Adjusted Average Given", f"{adj_given:.2f}")
                    st.metric("Adjusted SD Given", f"{ratings_given.std() / overall_std:.2f}")
                    st.metric("Adjusted Variance Given", f"{ratings_given.var() / analyzer.ratings.values.var():.2f}")
                with col10:
                    st.metric("Adjusted Average Received", f"{adj_received:.2f}")
                    st.metric("Adjusted SD Received", f"{ratings_received.std() / overall_std:.2f}")
                    st.metric("Adjusted Variance Received", f"{ratings_received.var() / analyzer.ratings.values.var():.2f}")
        
        with tabs[1]:
            # Create reciprocal relationships visualization
            st.subheader("Reciprocal Relationships")
            
            # Add chart and table
            recip_fig, recip_table = visualizer.create_reciprocal_chart(selected_person)
            
            if recip_fig:
                # Display chart
                st.plotly_chart(recip_fig, use_container_width=True)
                
                # Display table with custom formatting
                st.dataframe(
                    recip_table,
                    column_config={
                        'Other_Person': 'Person',
                        'Rating_Given': st.column_config.NumberColumn('Rating Given', format="%.1f"),
                        'Rating_Received': st.column_config.NumberColumn('Rating Received', format="%.1f"),
                        'Difference': st.column_config.NumberColumn('Difference', format="%.1f"),
                        'Reciprocity_Score': st.column_config.NumberColumn('Reciprocity Score', format="%.4f"),
                        'Relationship_Type': 'Relationship Type'
                    },
                    hide_index=True,
                    use_container_width=True
                )
        
        with tabs[2]:
            st.subheader("Relationship Network")
            
            # Show the new relationship summary visualization
            summary_fig = visualizer.create_relationship_summary(selected_person)
            st.plotly_chart(summary_fig, use_container_width=True)
            
            # Add legend explanation
            st.markdown("""
            ### Legend
            - 🤝 Mutually Friendly: Both ratings ≥7 and difference ≤2.5
            - ⚠️ Mutually Hostile: Both ratings <7 and difference ≤2.5
            - ➡️ One-sided (From Self): You rate them higher by >2.5 points
            - ⬅️ One-sided (From Other): They rate you higher by >2.5 points
            - ❓ Mixed: Ratings within 2.5 points but not clearly friendly/hostile
            """)
            
            # Show the tables below
            st.markdown("### Detailed Relationships")
            col1, col2 = st.columns(2)
            
            # Rest of your existing table code...
    
    elif page == "Hierarchical Analysis":
        st.header("Hierarchical Analysis")
        
        # Display organizational hierarchy chart
        st.subheader("Organizational Structure")
        st.plotly_chart(visualizer.create_hierarchy_chart())
        
        # Get hierarchical ratings data
        level_data = analyzer.analyze_hierarchical_ratings()
        
        # Display upward vs downward ratings analysis
        st.subheader("Rating Flow Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Upward Ratings")
            upward_data = level_data[level_data['Rating_Type'] == 'Upward']
            upward_avg = upward_data['Mean_Rating'].mean()
            st.metric(
                "Average Rating from Lower to Higher Levels",
                f"{upward_avg:.2f}",
                help="How subordinates rate their superiors"
            )
            
        with col2:
            st.markdown("##### Downward Ratings")
            downward_data = level_data[level_data['Rating_Type'] == 'Downward']
            downward_avg = downward_data['Mean_Rating'].mean()
            st.metric(
                "Average Rating from Higher to Lower Levels",
                f"{downward_avg:.2f}",
                help="How superiors rate their subordinates"
            )
        
        # Add peer ratings analysis
        st.markdown("##### Peer Level Ratings")
        peer_data = level_data[level_data['Rating_Type'] == 'Peer']
        
        # Group peer ratings by level
        peer_by_level = {}
        for level in sorted(set(analyzer.levels)):
            level_peer_data = peer_data[peer_data['Rater_Level'] == level]
            if not level_peer_data.empty:
                peer_by_level[f"Level {level}"] = level_peer_data['Mean_Rating'].mean()
        
        # Create bar chart for peer ratings
        fig = go.Figure(data=[
            go.Bar(
                x=list(peer_by_level.keys()),
                y=list(peer_by_level.values()),
                text=[f"{v:.2f}" for v in peer_by_level.values()],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Average Peer Ratings by Hierarchy Level",
            xaxis_title="Hierarchy Level",
            yaxis_title="Average Rating",
            height=400,
            showlegend=False,
            yaxis_range=[0, 10]
        )
        
        st.plotly_chart(fig)
        
        # Display inter-role rating patterns
        st.subheader("Inter-Role Rating Patterns")
        st.plotly_chart(visualizer.create_category_preference_plot())
        
        # Add insights about hierarchical dynamics
        st.subheader("Hierarchical Insights")
        
        # Calculate some key metrics
        rating_bias = upward_avg - downward_avg
        peer_rating_variance = np.var(list(peer_by_level.values()))
        
        # Display insights
        insights = []
        
        if abs(rating_bias) > 1:
            if rating_bias > 0:
                insights.append("⚠️ There appears to be significant upward rating inflation - subordinates tend to rate superiors higher than vice versa.")
            else:
                insights.append("⚠️ There appears to be significant downward rating inflation - superiors tend to rate subordinates higher than vice versa.")
        else:
            insights.append("✅ Ratings appear relatively balanced between hierarchical levels.")
            
        if peer_rating_variance > 1:
            insights.append("⚠️ There is high variance in peer ratings across different levels, suggesting potential silos or communication gaps.")
        else:
            insights.append("✅ Peer ratings are consistent across hierarchical levels, suggesting good horizontal collaboration.")
            
        for insight in insights:
            st.markdown(insight)
    
    elif page == "Performance Analysis":
        st.header("Performance Analysis")
        
        # Calculate and display performance scores
        all_scores = []
        for person in ratings_df.index:
            score = analyzer.calculate_performance_score(person)
            all_scores.append({
                'Person': person,
                'Score': score,
                'Role': analyzer.get_person_role(person),
                'Level': analyzer.level_mapping[person]
            })
        
        scores_df = pd.DataFrame(all_scores)
        
        # Display scores
        st.subheader("Performance Scores")
        fig = px.bar(scores_df.sort_values('Score', ascending=False),
                     x='Person', y='Score', 
                     color='Role',
                     title='Performance Scores by Person')
        fig.update_layout(
            yaxis=dict(
                range=[0, 10],  # Set fixed range from 0 to 10
                title="Score (out of 10)"
            ),
            xaxis_title="Person",
            height=500
        )
        st.plotly_chart(fig)
        
        # Display groupism analysis
        st.subheader("Groupism Analysis")
        groupism_patterns = analyzer.detect_groupism()
        
        if groupism_patterns:
            for pattern in groupism_patterns:
                st.warning(f"""
                    **Potential Group Behavior Detected**
                    - Group Members: {', '.join(pattern['Group_Members'])}
                    - Correlation: {pattern['Correlation']:.2f}
                    - Potentially Targeted: {', '.join(pattern['Targeted_Members'])}
                """)
        else:
            st.success("No significant groupism patterns detected")
        
        # Add this in the Performance Analysis section, before groupism analysis
        st.subheader("Performance Quadrant Analysis")
        quadrant_fig = visualizer.create_quadrant_analysis()
        st.plotly_chart(quadrant_fig)

if __name__ == "__main__":
    main() 