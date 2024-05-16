import numpy as np
import streamlit as st
import pyBKT_model as custom_bkt

st.set_page_config(
        page_title="Learning Path",
        page_icon="",
    )
def display_learning_path(predicted_levels, skill_names):
    print(predicted_levels)
    # Define CSS styles for the skill box
    box_styles = {
        "box": "border: 1px solid #ddd; background-color: #f9f9f9; padding: 20px; margin-bottom: 20px; border-radius: 5px;"
    }
    
    # Construct HTML content for learning path
    learning_path_html = ""
    
    for i, skill_name in enumerate(skill_names):
        # Start box for each skill
        learning_path_html += f"<div style='{box_styles['box']}'>"
        # Skill name
        learning_path_html += f"<h3 style='font-weight: bold;'>{skill_name}</h3>"
        
        # Skill details
        if predicted_levels[i * 3] == 1:
            learning_path_html += "<div style='background-color: #d4edda; color: #155724; padding: 10px; margin-bottom: 5px; border-radius: 3px; font-size: 22px;'>Easy</div>"
        if predicted_levels[i * 3 + 1] == 1:
            learning_path_html += "<div style='background-color: #fff3cd; color: #856404; padding: 10px; margin-bottom: 5px; border-radius: 3px; font-size: 22px;'>Medium</div>"
        if predicted_levels[i * 3 + 2] == 1:
            learning_path_html += "<div style='background-color: #f8d7da; color: #721c24; padding: 10px; margin-bottom: 5px; border-radius: 3px; font-size: 22px;'>Hard</div>"
        
        # End box for each skill
        learning_path_html += "</div>"
    
    # Display learning path
    st.markdown(learning_path_html, unsafe_allow_html=True)

predicted_levels = np.array([[0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]])
skill_names = ["Array", "LinkedList", "Stack", "Queue", "Heap", "Tree", "Graph", "Hash Map", "Sorting", "Searching", "Recursion", "Backtracking", "Dynamic Programming", "String"]

display_learning_path(predicted_levels[0], skill_names)
