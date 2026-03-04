import streamlit as st
import logging
import pandas as pd
import altair as alt

from DB.db_conn import get_connection

SCORE_COLS = ['bleu', 'rouge_1', 'rouge_2', 'rouge_l']

def display_data(df, option):
    if option == 1:
        
        melted = df.melt(id_vars='prompt_id', value_vars=SCORE_COLS,
                        var_name='Metric', value_name='Score')
        chart = alt.Chart(melted).mark_bar().encode(
            x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 1])),
            y=alt.Y('Metric:N'),
            color=alt.Color('Metric:N', legend=None)
        )
        st.altair_chart(chart, width='stretch')

    if option == 2:
        #st.write(f'{df['task_type']}')
        melted = df.melt(id_vars='prompt_id', value_vars=SCORE_COLS,
                        var_name='Metric', value_name='Score')
        chart = alt.Chart(melted).mark_bar().encode(
            x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 1])),
            y=alt.Y('Metric:N'),
            color=alt.Color('Metric:N', legend=None),
            row=alt.Row('prompt_id:N', title='Prompt ID')
        )
        st.altair_chart(chart)

def pull_generation_data(conn):
    query = """
        SELECT
            m.prompt_id,
            m.task_type,
            m.bleu,
            m.rouge_1,
            m.rouge_2,
            m.rouge_l,
            g.latency,
            g.tokens_generated,
            g.tokens_prompt,
            g.total_tokens
        FROM metrics m
        JOIN generations g ON m.response_id = g.response_id
    """
    return pd.read_sql(query, conn)


def pull_data(conn, prompt_id=None, batch_id=None):
    if prompt_id is not None:
        query = "SELECT prompt_id, bleu, rouge_1, rouge_2, rouge_l, batch_id, task_type FROM metrics WHERE prompt_id = %s;"
        return pd.read_sql(query, conn, params=(prompt_id,))
    elif batch_id is not None:
        query = "SELECT prompt_id, bleu, rouge_1, rouge_2, rouge_l, batch_id FROM metrics WHERE batch_id = %s;"
        return pd.read_sql(query, conn, params=(batch_id,))
    else:
        query = "SELECT prompt_id, bleu, rouge_1, rouge_2, rouge_l, task_type FROM metrics;"
        return pd.read_sql(query, conn)


def establish_connection():
    conn = get_connection()

    if conn is None:
        st.error("Can't establish connection to DB. Check your .env file and DB status.")
        st.stop()

    return conn


logger = logging.getLogger(__name__)

with st.sidebar:
    conn = establish_connection()
    df = pull_data(conn)

    if df.empty:
        logger.error("Error pulling data from DB for sidebar options")

    st.header("Display Options")
    option = st.selectbox(
        "Choose data you want to display...",
        ["Metric Analysis", "Generation Analysis", "Batch Comparisons"]
    )



st.title("Data Visualisations")

if option == 'Metric Analysis':
    qa_df = df[df['task_type'].str.upper() == 'QA']
    summ_df = df[df['task_type'].str.upper() == 'SUMMARISATION']

    if qa_df.empty or summ_df.empty:
        st.warning("No QA/Summarisation data found for Metric Analysis.")
    else:
        st.subheader("Average Scores - QA & Summarisation")
        qa_scores = qa_df[SCORE_COLS].mean()
        cols = st.columns(4)
        labels = {
            'bleu': 'BLEU', 
            'rouge_1': 'ROUGE-1', 
            'rouge_2': 'ROUGE-2', 
            'rouge_l': 'ROUGE-L'
        }
        
        for col, key in zip(cols, SCORE_COLS):
            col.metric(labels[key], f"{qa_scores[key]:.4f}")

        summ_scores = summ_df[SCORE_COLS].mean()

        for col, key in zip(cols, SCORE_COLS):
            col.metric(labels[key], f"{summ_scores[key]:.4f}")

        avg_data = pd.DataFrame([
            {'Task Type': 'QA', 'Metric': labels[k], 'Score': qa_scores[k]} for k in SCORE_COLS
        ] + [
            {'Task Type': 'Summarisation', 'Metric': labels[k], 'Score': summ_scores[k]} for k in SCORE_COLS
        ])
        chart = alt.Chart(avg_data).mark_bar().encode(
            x=alt.X('Metric:N', title=None),
            y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('Task Type:N'),
            xOffset='Task Type:N'
        )
        st.altair_chart(chart, use_container_width=True)
    

if option == 'Generation Analysis':
    gen_df = pull_generation_data(conn)

    numeric_cols = ['latency', 'tokens_generated', 'tokens_prompt', 'total_tokens']

    # Helps avoid to numeric error - Don't remove else breaks dashboard for gen. analysis
    gen_df[numeric_cols] = gen_df[numeric_cols].apply(pd.to_numeric, errors='coerce') 

    if gen_df.empty:
        st.warning("No generation data found. Run some evaluations first.")
    else:
        task_filter = st.selectbox("Filter by task type", ["All", "QA", "SUMMARISATION"])
        if task_filter != "All":
            gen_df = gen_df[gen_df['task_type'].str.upper() == task_filter]

        st.subheader("Generation Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Latency (ms)", f"{gen_df['latency'].mean():.0f}")
        c2.metric("Avg Tokens Generated", f"{gen_df['tokens_generated'].mean():.0f}")
        c3.metric("Avg Prompt Tokens", f"{gen_df['tokens_prompt'].mean():.0f}")
        c4.metric("Total Generations", len(gen_df))

        melted = gen_df.melt(
            id_vars=['task_type', 'latency', 'tokens_generated', 'tokens_prompt'],
            value_vars=SCORE_COLS,
            var_name='Metric',
            value_name='Score'
        )

        st.subheader("Latency vs Score")
        latency_chart = alt.Chart(melted).mark_circle(size=60, opacity=0.7).encode(
            x=alt.X('latency:Q', title='Latency (ms)'),
            y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('task_type:N', title='Task Type'),
            facet=alt.Facet('Metric:N', columns=2),
            tooltip=['task_type', 'latency', 'Score', 'Metric']
        ).properties(width=300, height=220)
        st.altair_chart(latency_chart)

        st.subheader("Tokens Generated vs Score")
        tokens_chart = alt.Chart(melted).mark_circle(size=60, opacity=0.7).encode(
            x=alt.X('tokens_generated:Q', title='Tokens Generated'),
            y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('task_type:N', title='Task Type'),
            facet=alt.Facet('Metric:N', columns=2),
            tooltip=['task_type', 'tokens_generated', 'Score', 'Metric']
        ).properties(width=300, height=220)
        st.altair_chart(tokens_chart)

        st.subheader("Prompt Tokens vs Score")
        prompt_tokens_chart = alt.Chart(melted).mark_circle(size=60, opacity=0.7).encode(
            x=alt.X('tokens_prompt:Q', title='Prompt Tokens'),
            y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('task_type:N', title='Task Type'),
            facet=alt.Facet('Metric:N', columns=2),
            tooltip=['task_type', 'tokens_prompt', 'Score', 'Metric']
        ).properties(width=300, height=220)
        st.altair_chart(prompt_tokens_chart)

display_choice = st.radio(
    label="Select data to display",
    options=["By Prompt ID", "By Batch ID"],
    captions=[
        "Returns evaluations for that Prompt ID", 
        "Returns evaluations for that Batch ID"
    ]    
)

if display_choice == "By Prompt ID":
    prompt_id = st.number_input("Enter Prompt ID", step=1)
    conn = establish_connection()
    data = pull_data(conn, prompt_id=prompt_id)
    if data.empty:
        st.warning(f"No data found for {prompt_id}")
    display_data(data, 1)

if display_choice == "By Batch ID":
    batch_id = st.text_input("Enter Batch ID")
    conn = establish_connection()
    data = pull_data(conn, batch_id=batch_id)
    if data.empty:
        st.warning(f"No data found for {batch_id}")
    display_data(data, 2)