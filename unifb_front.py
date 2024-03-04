import streamlit as st
from streamlit_option_menu import option_menu
# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 2


def streamlit_menu(example=EXAMPLE_NO, index=0):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title=None,  # required
                options=["Home", "Projects", "Contact"],  # required
                icons=["house", "book", "envelope"],  # optional
                menu_icon="cast",  # optional
                default_index=index,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Q&A", "Advice", "More"],  # required
            icons=["house", "book", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=index,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 2. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Projects", "Contact"],  # required
            icons=["house", "book", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=index,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected


selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "Q&A":
    st.title('Upload your question and solution')

    # Query text
    query_text = st.text_input(
        'Enter your question:', placeholder='Please input the assignment\'s instruction')

    # Form input and query
    result = []
    with st.form('myform', clear_on_submit=True):
        solution = st.text_area('Assignment',
                                placeholder="Paste your draft",)
        submitted = st.form_submit_button(
            'Submit')
        if submitted:
            with st.spinner('Calculating...'):

                selected = streamlit_menu(index=1)

if selected == "Advice":
    st.title(f"You have selected {selected}")
if selected == "More":
    st.title(f"You have selected {selected}")
