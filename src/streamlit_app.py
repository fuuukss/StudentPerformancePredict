import streamlit as st

from model_usage import (
    CATEGORY_OPTIONS,
    DEFAULT_INPUT,
    FINAL_MODEL_PATH,
    format_prediction,
    load_final_model,
    predict_final_grade,
)


st.set_page_config(
    page_title="Student Performance Predict",
    page_icon="",
    layout="centered",
)


@st.cache_resource
def get_model_bundle():
    return load_final_model()


def number_input(label, key, min_value, max_value, step=1):
    return st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=DEFAULT_INPUT[key],
        step=step,
    )


model_bundle = get_model_bundle()

st.title("Student Performance Predict")
st.caption("Step 11 - Deployment / koriscenje finalnog modela")

with st.sidebar:
    st.subheader("Finalni model")
    st.write(f"Model: {model_bundle['model']}")
    st.write(f"Scenario: {model_bundle['scenario']}")
    st.write(f"Putanja: `{FINAL_MODEL_PATH.relative_to(FINAL_MODEL_PATH.parents[2])}`")

with st.form("prediction_form"):
    st.subheader("Ulazni podaci")

    left, right = st.columns(2)
    with left:
        g2 = number_input("G2 - druga periodicka ocena", "G2", 0, 20)
        g1 = number_input("G1 - prva periodicka ocena", "G1", 0, 20)
        age = number_input("Uzrast", "age", 10, 25)
        absences = number_input("Broj izostanaka", "absences", 0, 100)
        failures = number_input("Broj prethodnih neuspeha", "failures", 0, 4)

    with right:
        reason = st.selectbox(
            "Razlog izbora skole",
            CATEGORY_OPTIONS["reason"],
            index=CATEGORY_OPTIONS["reason"].index(DEFAULT_INPUT["reason"]),
        )
        school = st.selectbox(
            "Skola",
            CATEGORY_OPTIONS["school"],
            index=CATEGORY_OPTIONS["school"].index(DEFAULT_INPUT["school"]),
        )
        freetime = number_input("Slobodno vreme", "freetime", 1, 5)
        health = number_input("Zdravlje", "health", 1, 5)
        fedu = number_input("Obrazovanje oca", "Fedu", 0, 4)

    submitted = st.form_submit_button("Predvidi G3", type="primary")

values = {
    "G2": g2,
    "absences": absences,
    "G1": g1,
    "age": age,
    "reason": reason,
    "freetime": freetime,
    "health": health,
    "Fedu": fedu,
    "school": school,
    "failures": failures,
}

if submitted:
    prediction, input_df = predict_final_grade(values, model_bundle)
    st.metric("Predvidjena zavrsna ocena G3", format_prediction(prediction))
    with st.expander("Ulaz poslat modelu"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)
