import streamlit as st
import json
import random
import pyBKT_model as custom_bkt

def run():
    st.set_page_config(
        page_title="Champ Learn",
        page_icon="❓",
    )

show_learning_path = False

def display_learning_path(predicted_levels, skill_names):
    # Define CSS styles for the skill box
    box_styles = {
        "box": "border: 1px solid #ddd; background-color: #f9f9f9; padding: 20px; margin-bottom: 20px; border-radius: 5px;"
    }
    
    # Construct HTML content for learning path
    learning_path_html = "<div style=padding: 10px; margin-bottom: 5px; font-size: 40px;'>Recommended Learning Path</div>"
    
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
    with st.empty():
    # Display learning path
        st.markdown(learning_path_html, unsafe_allow_html=True)

def find_learning_path(score):
    print(score)
    predicted_levels = custom_bkt.predict_output(score)
    print(predicted_levels)
    skill_names = ["Array", "LinkedList", "Stack", "Queue", "Tree", "Graph", "Searching", "Sorting", "Recursion", "Dynamic Programming"]
    display_learning_path(predicted_levels[0], skill_names)


if __name__ == "__main__":
    run()

# Custom CSS for the buttons
st.markdown("""
<style>
div.stButton > button:first-child {
    display: block;
    margin: 0 auto;
</style>
""",unsafe_allow_html=True)

# Initialize session variables if they do not exist
default_values = {'current_index': 0,'score': 0,'selected_option': None,'answer_submitted': False,'current_question': None,'quiz_data': [], 'correct_attempt': []}
for key,value in default_values.items():
    st.session_state.setdefault(key,value)

# Load questions from each category
array_questions = []
with open('content/array-quiz_data.json','r',encoding='utf-8') as f:
    array_questions = json.load(f)

linkedlist_questions = []
with open('content/linked-list-quiz_data.json','r',encoding='utf-8') as f:
    linkedlist_questions = json.load(f)

stack_questions = []
with open('content/stack-quiz_data.json','r',encoding='utf-8') as f:
    stack_questions = json.load(f)

queue_questions = []
with open('content/queue-quiz_data.json','r',encoding='utf-8') as f:
    queue_questions = json.load(f)

tree_questions = []
with open('content/tree-quiz_data.json','r',encoding='utf-8') as f:
    tree_questions = json.load(f)

graph_questions = []
with open('content/graph-quiz_data.json','r',encoding='utf-8') as f:
    graph_questions = json.load(f)

searching_questions = []
with open('content/searching-quiz_data.json','r',encoding='utf-8') as f:
    searching_questions = json.load(f)

sorting_questions = []
with open('content/sorting-quiz_data.json','r',encoding='utf-8') as f:
    sorting_questions = json.load(f)

recursion_questions = []
with open('content/recursion-quiz_data.json','r',encoding='utf-8') as f:
    recursion_questions = json.load(f)

dp_questions = []
with open('content/dp-quiz_data.json','r',encoding='utf-8') as f:
    dp_questions = json.load(f)

if not st.session_state.quiz_data:
    random.shuffle(array_questions)
    random.shuffle(linkedlist_questions)
    random.shuffle(stack_questions)
    random.shuffle(queue_questions)
    random.shuffle(tree_questions)
    random.shuffle(graph_questions)
    random.shuffle(searching_questions)
    random.shuffle(sorting_questions)
    random.shuffle(recursion_questions)
    random.shuffle(dp_questions)
    

    st.session_state.quiz_data.extend(array_questions[:5])
    st.session_state.quiz_data.extend(linkedlist_questions[:5])
    st.session_state.quiz_data.extend(stack_questions[:5])
    st.session_state.quiz_data.extend(queue_questions[:5])
    st.session_state.quiz_data.extend(tree_questions[:5])
    st.session_state.quiz_data.extend(graph_questions[:5])
    st.session_state.quiz_data.extend(searching_questions[:5])
    st.session_state.quiz_data.extend(sorting_questions[:5])
    st.session_state.quiz_data.extend(recursion_questions[:5])
    st.session_state.quiz_data.extend(dp_questions[:5])

def restart_quiz():
    st.session_state.current_index = 0
    st.session_state.score = 0
    st.session_state.selected_option = None
    st.session_state.answer_submitted = False
    st.session_state.current_question = st.session_state.quiz_data[0]

if not st.session_state.current_question:
    restart_quiz()

def submit_answer():

    # Check if an option has been selected
    if st.session_state.selected_option is not None:
        # Mark the answer as submitted
        st.session_state.answer_submitted = True
        # Check if the selected option is correct
        if st.session_state.selected_option == st.session_state.current_question['answer']:
            st.session_state.score += 10
            st.session_state.correct_attempt.append(1)
        else:
            st.session_state.correct_attempt.append(0)
    else:
        # If no option selected,show a message and do not mark as submitted
        st.warning("Please select an option before submitting.")

def next_question():
    st.session_state.current_index += 1
    st.session_state.selected_option = None
    st.session_state.answer_submitted = False
    if st.session_state.current_index < len(st.session_state.quiz_data):
        st.session_state.current_question = st.session_state.quiz_data[st.session_state.current_index]

# Progress bar
progress_bar_value = (st.session_state.current_index + 1) / len(st.session_state.quiz_data)
st.metric(label="Score",value=f"{st.session_state.score} / {len(st.session_state.quiz_data) * 10}")
st.progress(progress_bar_value)

# Display the question and answer options
question_item = st.session_state.current_question
st.subheader(f"Question {st.session_state.current_index + 1}")
st.title(f"{question_item['question']}")
st.markdown("<br>",unsafe_allow_html=True)

# Answer selection
options = question_item['options']
correct_answer = question_item['answer']

if st.session_state.answer_submitted:
    for i,option in enumerate(options):
        label = option
        if option == correct_answer:
            st.success(f"{label} (Correct answer)")
        elif option == st.session_state.selected_option:
            st.error(f"{label} (Incorrect answer)")
        else:
            st.write(label)

    # Display the information
    st.markdown(""" ___""")
    st.write(question_item['information'])
else:
    for i,option in enumerate(options):
        if st.button(option,key=i,use_container_width=True):
            st.session_state.selected_option = option


st.markdown("<br>",unsafe_allow_html=True)

# Submission button and response logic
if st.session_state.answer_submitted:
    if st.session_state.current_index < len(st.session_state.quiz_data) - 1:
        st.button('Next',on_click=next_question)
    else:
        st.balloons()
        st.write(f"Quiz completed! Your score is: {st.session_state.score} / {len(st.session_state.quiz_data) * 10}")
        find_learning_path(st.session_state.correct_attempt)
        # if st.button('Restart',on_click=restart_quiz):
        #     pass
else:
    if st.session_state.current_index < len(st.session_state.quiz_data):
        st.button('Submit',on_click=submit_answer)

