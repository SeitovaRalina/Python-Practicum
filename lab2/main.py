import random
import streamlit as st
from rbtree import RedBlackTree

st.set_page_config(
    page_title="Red-Black Tree",
    page_icon="🌲"
)

st.title(':red[Red] - :gray[Black] Tree Visualizer')

session = st.session_state

if 'tree' not in session:
    session.tree = RedBlackTree()

if 'inserted_values' not in session:
    session.inserted_values = []

if 'session_iteration' not in session:
    session.session_iteration = 0

sidebar = st.sidebar
sidebar.title('⚙️Settings')

sidebar.subheader('🔢 Inserting a new value')
sidebar.text_input(label = 'df', key='insert_field', placeholder='Введите целое число', label_visibility='collapsed')
def clear_insert_text():
    session.new_value = session.insert_field
    session["insert_field"] = ""
sidebar.button(label='GO!', key='insert_button', on_click=clear_insert_text, use_container_width=True)

if session.insert_button:
    try:
        new_value = int(session.new_value)
    except ValueError as e:
        new_value = None
        st.error(f'⛔️ InCorrEcT InpUT: {e}')

    try:
        if new_value in session.inserted_values: raise 
        session.tree.insert(new_value)
        session.inserted_values.append(new_value)
        st.success(f'Succesfully added new node : {new_value}', icon='✅')
    except:
        st.warning(f'Node {new_value} is ALREADY contained in 🌲', icon='⚠️')
        

sidebar.subheader('🗑 Deleting a value')
sidebar.text_input(label='', placeholder='Введите целое число', key='delete_field', label_visibility='collapsed')
def clear_delete_text():
    session.deleting_value = session.delete_field
    session["delete_field"] = ""
sidebar.button(label='GO!', key='delete_button', on_click=clear_delete_text, use_container_width=True)

if session.delete_button:
    try:
        del_value = int(session.deleting_value)
    except ValueError as e:
        del_value = None
        st.error(f'⛔️ InCorrEcT InpUT: {e}')

    try:
        session.tree.delete(del_value)
        session.inserted_values.remove(del_value)
        st.success(f'Succesfully deleted node : {del_value}', icon='✅')
    except TypeError:
        st.warning(f'Node {del_value} is NOT contained in 🌲', icon='⚠️')


sidebar.header("✨ :rainbow[RANDOM] ✨")

sidebar.subheader('🎲Inserting random values up to a 1000')
random_insert_slider = sidebar.slider(label='Pick the number', key=f'insert_slider_{session.session_iteration}', min_value=1, max_value=10)

def insert_random():
    session.session_iteration += 1
    random_insert_values_count = session[f"insert_slider_{session.session_iteration - 1}"]
    sequence = set(range(-100, 100)) - set(session.inserted_values)
    values = random.sample(list(sequence), random_insert_values_count)
    for value in values : session.tree.insert(value)
    session.inserted_values.extend(values)
    if values:
        st.success(f'Successfully added nodes: {", ".join(map(str, values))}', icon='✅')
sidebar.button(label='GO!', key='random_insert_button', on_click=insert_random, use_container_width=True)

sidebar.subheader('🎲Deleting random values')
temp_max = len(session.inserted_values) if len(session.inserted_values) != 1 else 0
random_delete_slider = sidebar.slider(label='Pick the number', key=f'delete_slider_{session.session_iteration}', min_value=1, max_value=temp_max)


def delete_random():
    session.session_iteration += 1
    random_deleting_values_count = session[f"delete_slider_{session.session_iteration - 1}"]
    if random_deleting_values_count >= len(session.inserted_values):
        session.tree = RedBlackTree()
        session.inserted_values = []
        session.session_iteration = 0
        st.success(f'The tree was successfully deleted 🌲🌲🌲', icon='✅')
    else:
        values = random.sample(session.inserted_values, random_deleting_values_count)
        for value in values:
            session.tree.delete(value)
            session.inserted_values.remove(value)
        if values:
            st.success(f'Successfully added nodes: {", ".join(map(str, values))}', icon='✅')
sidebar.button(label='GO!', key='random_delete_button', on_click=delete_random, use_container_width=True)

if session.inserted_values:
    session.tree.draw_img()
    st.image('Red_Black_Tree.png', use_column_width=True)
    st.header('', divider='rainbow')
    st.subheader(f'Tree contains the following :red[{len(session.inserted_values)}] nodes:\n {", ".join(map(str, sorted(session.inserted_values)))}')
