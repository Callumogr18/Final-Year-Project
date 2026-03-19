import logging
import altair as alt
import pandas as pd
import streamlit as st

from DB.db_conn import get_connection

logger = logging.getLogger(__name__)

SCORE_COLS = ['bleu', 'rouge_1', 'rouge_2', 'rouge_l']
SCORE_LABELS = {'bleu': 'BLEU', 'rouge_1': 'ROUGE-1', 'rouge_2': 'ROUGE-2', 'rouge_l': 'ROUGE-L'}
JUDGE_COLS = ['hallucination', 'fluency', 'consistency', 'reasoning', 'coherence', 'accuracy']
JUDGE_LABELS = {
    'hallucination': 'Hallucination',
    'fluency': 'Fluency',
    'consistency': 'Consistency',
    'reasoning': 'Reasoning',
    'coherence': 'Coherence',
    'accuracy': 'Factual Accuracy',
}
PERF_COLS = ['latency', 'tokens_generated', 'tokens_prompt', 'total_tokens']


@st.cache_resource
def get_db_connection():
    conn = get_connection()
    if conn is None:
        st.error("Can't establish connection to DB. Check your .env file and DB status.")
        st.stop()
    return conn


@st.cache_data
def load_metrics(_conn):
    df = pd.read_sql(
        "SELECT prompt_id, bleu, rouge_1, rouge_2, rouge_l, batch_id, task_type FROM metrics;",
        _conn
    )
    df[SCORE_COLS] = df[SCORE_COLS].apply(pd.to_numeric, errors='coerce')
    return df


@st.cache_data
def load_generation_data(_conn):
    query = """
        SELECT m.prompt_id, m.task_type, m.bleu, m.rouge_1, m.rouge_2, m.rouge_l,
               g.latency, g.tokens_generated, g.tokens_prompt, g.total_tokens
        FROM metrics m
        JOIN generations g ON m.response_id = g.response_id
    """
    df = pd.read_sql(query, _conn)
    df[SCORE_COLS + PERF_COLS] = df[SCORE_COLS + PERF_COLS].apply(pd.to_numeric, errors='coerce')
    return df


@st.cache_data
def load_judge_data(_conn):
    query = """
        SELECT jm.prompt_id, jm.task_type,
               jm.hallucination, jm.fluency, jm.consistency,
               jm.reasoning, jm.coherence, jm.accuracy,
               m.bleu, m.rouge_1, m.rouge_2, m.rouge_l,
               g.llm_response, g.latency, g.tokens_generated, g.tokens_prompt, g.total_tokens,
               p.question, p.article
        FROM judge_metrics jm
        JOIN metrics m ON jm.response_id = m.response_id
        JOIN generations g ON jm.response_id = g.response_id
        JOIN prompts p ON jm.prompt_id = p.id
    """
    df = pd.read_sql(query, _conn)
    numeric = JUDGE_COLS + SCORE_COLS + PERF_COLS
    df[numeric] = df[numeric].apply(pd.to_numeric, errors='coerce')
    return df


st.set_page_config(page_title="LLM Evaluation Dashboard", layout="wide")
st.title("LLM Evaluation Dashboard")

conn = get_db_connection()
metrics_df = load_metrics(conn)
gen_df = load_generation_data(conn)
judge_df = load_judge_data(conn)

tab_overview, tab_metrics, tab_judge, tab_gen, tab_explorer = st.tabs([
    "Overview", "Metric Analysis", "Judge Analysis", "Generation Analysis", "Response Explorer"
])

with tab_overview:
    total = len(metrics_df)
    qa_count = len(metrics_df[metrics_df['task_type'].str.upper() == 'QA'])
    summ_count = len(metrics_df[metrics_df['task_type'].str.upper() == 'SUMMARISATION'])

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Evaluations", total)
    c2.metric("QA Evaluations", qa_count)
    c3.metric("Summarisation Evaluations", summ_count)

    st.divider()

    if not metrics_df.empty:
        st.subheader("Average Scores by Task Type")
        avg_by_task = (
            metrics_df.groupby('task_type')[SCORE_COLS]
            .mean()
            .reset_index()
            .melt(id_vars='task_type', value_vars=SCORE_COLS, var_name='Metric', value_name='Score')
        )
        avg_by_task['Metric'] = avg_by_task['Metric'].map(SCORE_LABELS)

        chart = alt.Chart(avg_by_task).mark_bar().encode(
            x=alt.X('Metric:N', title=None),
            y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('task_type:N', title='Task Type'),
            xOffset='task_type:N',
            tooltip=['task_type', 'Metric', alt.Tooltip('Score:Q', format='.4f')]
        )
        st.altair_chart(chart, use_container_width=True)

    if not judge_df.empty:
        st.subheader("Average Judge Scores by Task Type")
        avg_judge = (
            judge_df.groupby('task_type')[JUDGE_COLS]
            .mean()
            .reset_index()
            .melt(id_vars='task_type', value_vars=JUDGE_COLS, var_name='Criterion', value_name='Score')
        )
        avg_judge['Criterion'] = avg_judge['Criterion'].map(JUDGE_LABELS)

        chart = alt.Chart(avg_judge).mark_bar().encode(
            x=alt.X('Criterion:N', title=None),
            y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('task_type:N', title='Task Type'),
            xOffset='task_type:N',
            tooltip=['task_type', 'Criterion', alt.Tooltip('Score:Q', format='.4f')]
        )
        st.altair_chart(chart, use_container_width=True)


with tab_metrics:
    if metrics_df.empty:
        st.warning("No metric data found.")
    else:
        qa_df = metrics_df[metrics_df['task_type'].str.upper() == 'QA']
        summ_df = metrics_df[metrics_df['task_type'].str.upper() == 'SUMMARISATION']

        for label, subset in [("QA", qa_df), ("Summarisation", summ_df)]:
            if subset.empty:
                continue
            st.subheader(f"{label} — Average Scores")
            avg = subset[SCORE_COLS].mean()
            cols = st.columns(4)
            for col, key in zip(cols, SCORE_COLS):
                col.metric(SCORE_LABELS[key], f"{avg[key]:.4f}")

        st.divider()
        st.subheader("Per-Prompt Score Heatmap")

        task_choice = st.radio("Task type", ["QA", "Summarisation"], horizontal=True, key="heatmap_task")
        heatmap_df = qa_df if task_choice == "QA" else summ_df

        if not heatmap_df.empty:
            heatmap_data = heatmap_df[['prompt_id'] + SCORE_COLS].melt(
                id_vars='prompt_id', value_vars=SCORE_COLS, var_name='Metric', value_name='Score'
            )
            heatmap_data['Metric'] = heatmap_data['Metric'].map(SCORE_LABELS)
            heatmap_data['prompt_id'] = heatmap_data['prompt_id'].astype(str)

            heatmap = alt.Chart(heatmap_data).mark_rect().encode(
                x=alt.X('Metric:N', title=None),
                y=alt.Y('prompt_id:N', title='Prompt ID', sort=None),
                color=alt.Color('Score:Q', scale=alt.Scale(domain=[0, 1], scheme='redyellowgreen'), title='Score'),
                tooltip=['prompt_id', 'Metric', alt.Tooltip('Score:Q', format='.4f')]
            ).properties(height=max(200, len(heatmap_df) * 14))
            st.altair_chart(heatmap, use_container_width=True)


with tab_judge:
    if judge_df.empty:
        st.warning("No judge data found. Run some evaluations first.")
    else:
        qa_j = judge_df[judge_df['task_type'].str.upper() == 'QA']
        summ_j = judge_df[judge_df['task_type'].str.upper() == 'SUMMARISATION']

        for label, subset in [("QA", qa_j), ("Summarisation", summ_j)]:
            if subset.empty:
                continue
            st.subheader(f"{label} — Average Judge Scores")
            avg = subset[JUDGE_COLS].mean()
            cols = st.columns(6)
            for col, key in zip(cols, JUDGE_COLS):
                col.metric(JUDGE_LABELS[key], f"{avg[key]:.4f}")

        st.divider()
        st.subheader("Judge vs Traditional Metric Correlation")

        col1, col2 = st.columns(2)
        with col1:
            judge_metric = st.selectbox(
                "Judge criterion",
                options=JUDGE_COLS,
                format_func=lambda k: JUDGE_LABELS[k]
            )
        with col2:
            trad_metric = st.selectbox(
                "Traditional metric",
                options=SCORE_COLS,
                format_func=lambda k: SCORE_LABELS[k]
            )

        scatter = alt.Chart(judge_df).mark_circle(size=60, opacity=0.6).encode(
            x=alt.X(f'{trad_metric}:Q', title=SCORE_LABELS[trad_metric], scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(f'{judge_metric}:Q', title=JUDGE_LABELS[judge_metric], scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('task_type:N', title='Task Type'),
            tooltip=[
                'prompt_id', 'task_type',
                alt.Tooltip(f'{trad_metric}:Q', format='.4f'),
                alt.Tooltip(f'{judge_metric}:Q', format='.4f')
            ]
        )
        st.altair_chart(scatter, use_container_width=True)


with tab_gen:
    if gen_df.empty:
        st.warning("No generation data found.")
    else:
        task_filter = st.selectbox("Filter by task type", ["All", "QA", "SUMMARISATION"])
        filtered = gen_df if task_filter == "All" else gen_df[gen_df['task_type'].str.upper() == task_filter]

        st.subheader("Generation Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Latency (ms)", f"{filtered['latency'].mean():.0f}")
        c2.metric("Avg Tokens Generated", f"{filtered['tokens_generated'].mean():.0f}")
        c3.metric("Avg Prompt Tokens", f"{filtered['tokens_prompt'].mean():.0f}")
        c4.metric("Total Generations", len(filtered))

        st.divider()

        melted = filtered.melt(
            id_vars=['task_type', 'latency', 'tokens_generated', 'tokens_prompt'],
            value_vars=SCORE_COLS, var_name='Metric', value_name='Score'
        )
        melted['Metric'] = melted['Metric'].map(SCORE_LABELS)

        st.subheader("Latency vs Score")
        latency_chart = alt.Chart(melted).mark_circle(size=60, opacity=0.7).encode(
            x=alt.X('latency:Q', title='Latency (ms)'),
            y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('task_type:N', title='Task Type'),
            facet=alt.Facet('Metric:N', columns=2),
            tooltip=['task_type', 'latency', 'Score', 'Metric']
        ).properties(width=300, height=200)
        st.altair_chart(latency_chart)

        st.subheader("Tokens Generated vs Score")
        tokens_chart = alt.Chart(melted).mark_circle(size=60, opacity=0.7).encode(
            x=alt.X('tokens_generated:Q', title='Tokens Generated'),
            y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('task_type:N', title='Task Type'),
            facet=alt.Facet('Metric:N', columns=2),
            tooltip=['task_type', 'tokens_generated', 'Score', 'Metric']
        ).properties(width=300, height=200)
        st.altair_chart(tokens_chart)


with tab_explorer:
    st.subheader("Response Explorer")

    search_by = st.radio("Search by", ["Prompt ID", "Batch ID"], horizontal=True)

    if search_by == "Prompt ID":
        prompt_id = st.number_input("Enter Prompt ID", min_value=1, step=1)
        if st.button("Load", key="load_prompt"):
            row = judge_df[judge_df['prompt_id'] == prompt_id]

            if row.empty:
                fallback = metrics_df[metrics_df['prompt_id'] == prompt_id]
                if fallback.empty:
                    st.warning(f"No data found for Prompt ID {prompt_id}")
                else:
                    st.dataframe(fallback[['prompt_id', 'task_type'] + SCORE_COLS])
            else:
                r = row.iloc[0]
                st.markdown(f"**Task Type:** {r['task_type']}")
                if pd.notna(r.get('question')):
                    st.markdown(f"**Question:** {r['question']}")
                if pd.notna(r.get('article')):
                    with st.expander("Article"):
                        st.write(r['article'])

                st.markdown("**LLM Response:**")
                st.info(r['llm_response'])

                st.subheader("Scores")
                left, right = st.columns(2)
                with left:
                    st.caption("Traditional Metrics")
                    cols = st.columns(4)
                    for col, key in zip(cols, SCORE_COLS):
                        col.metric(SCORE_LABELS[key], f"{pd.to_numeric(r[key], errors='coerce'):.4f}")
                with right:
                    st.caption("Judge Scores")
                    cols = st.columns(6)
                    for col, key in zip(cols, JUDGE_COLS):
                        col.metric(JUDGE_LABELS[key], f"{pd.to_numeric(r[key], errors='coerce'):.4f}")

    else:
        batch_id = st.text_input("Enter Batch ID")
        if st.button("Load", key="load_batch") and batch_id:
            batch_data = metrics_df[metrics_df['batch_id'] == batch_id]
            if batch_data.empty:
                st.warning(f"No data found for Batch ID {batch_id}")
            else:
                st.subheader(f"Batch {batch_id} — {len(batch_data)} evaluations")
                avg = batch_data[SCORE_COLS].mean()
                cols = st.columns(4)
                for col, key in zip(cols, SCORE_COLS):
                    col.metric(SCORE_LABELS[key], f"{avg[key]:.4f}")

                melted = batch_data.melt(
                    id_vars='prompt_id', value_vars=SCORE_COLS, var_name='Metric', value_name='Score'
                )
                melted['Metric'] = melted['Metric'].map(SCORE_LABELS)

                chart = alt.Chart(melted).mark_bar().encode(
                    x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y('Metric:N'),
                    color=alt.Color('Metric:N', legend=None),
                    row=alt.Row('prompt_id:N', title='Prompt ID')
                )
                st.altair_chart(chart)
