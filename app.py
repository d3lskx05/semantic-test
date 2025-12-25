import streamlit as st
from utils import load_all_excels, semantic_search, keyword_search
import torch

st.set_page_config(page_title="Проверка фраз ФЛ", layout="centered")
st.title("🤖 Проверка фраз")

# ---------------- DATA ----------------
@st.cache_data
def get_data():
    return load_all_excels()

df = get_data()

# ---------------- TOPICS ----------------
def normalize_topic(t: str) -> str:
    return t.strip().lower()

# мапа: нормализованная → оригинальная (красивая)
topic_display_map = {}
for topics in df["topics"]:
    for topic in topics:
        norm = normalize_topic(topic)
        if norm not in topic_display_map:
            topic_display_map[norm] = topic.strip()

# уникальные нормализованные тематики
all_topics_norm = sorted(topic_display_map.keys())

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🔍 Поиск", "🚫 Не используем", "✅/❌ Да и Нет"])

# ================= TAB 1 =================
with tab1:
    selected_topics = st.multiselect(
        "Фильтр по тематикам:",
        options=all_topics_norm,
        format_func=lambda t: topic_display_map[t]
    )

    filter_search_by_topics = st.checkbox(
        "Искать только в выбранных тематиках",
        value=False
    )

    # -------- Фразы по тематикам --------
    if selected_topics:
        st.markdown("### 📂 Фразы по выбранным тематикам:")
        shown_phrases = set()
        filtered_df = df[
            df["topics"].apply(
                lambda topics: any(
                    normalize_topic(t) in selected_topics for t in topics
                )
            )
        ]

        for row in filtered_df.itertuples():
            if row.phrase_full in shown_phrases:
                continue
            shown_phrases.add(row.phrase_full)

            topics_pretty = [
                topic_display_map.get(normalize_topic(t), t)
                for t in row.topics
            ]

            with st.container():
                st.markdown(
                    f"""
                    <div style="border:1px solid #e0e0e0;
                                border-radius:12px;
                                padding:16px;
                                margin-bottom:12px;
                                background:#f9f9f9;">
                        <div style="font-size:18px;font-weight:600;">📝 {row.phrase_full}</div>
                        <div style="font-size:14px;color:#666;">
                            🔖 Тематики: <strong>{', '.join(topics_pretty)}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if row.comment and str(row.comment).strip().lower() != "nan":
                    with st.expander("💬 Комментарий"):
                        st.markdown(row.comment)

    # -------- Поиск --------
    query = st.text_input("Введите ваш запрос:")

    if query:
        try:
            search_df = df

            if filter_search_by_topics and selected_topics:
                search_df = df[
                    df["topics"].apply(
                        lambda topics: any(
                            normalize_topic(t) in selected_topics for t in topics
                        )
                    )
                ].copy()

                if search_df.empty:
                    search_df.attrs["phrase_embs"] = torch.empty((0, 384))

            if search_df.empty:
                st.warning("Нет данных для поиска по выбранным тематикам.")
            else:
                # ----- Semantic -----
                results = semantic_search(query, search_df)

                if results:
                    st.markdown("### 🔍 Умный поиск")
                    for score, phrase, topics, comment in results:
                        topics_pretty = [
                            topic_display_map.get(normalize_topic(t), t)
                            for t in topics
                        ]

                        st.markdown(
                            f"""
                            <div style="border:1px solid #e0e0e0;
                                        border-radius:12px;
                                        padding:16px;
                                        margin-bottom:12px;
                                        background:#f9f9f9;">
                                <div style="font-size:18px;font-weight:600;">🧠 {phrase}</div>
                                <div style="font-size:14px;color:#666;">
                                    🔖 Тематики: <strong>{', '.join(topics_pretty)}</strong>
                                </div>
                                <div style="font-size:13px;color:#999;">
                                    🎯 Релевантность: {score:.2f}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        if comment and str(comment).strip().lower() != "nan":
                            with st.expander("💬 Комментарий"):
                                st.markdown(comment)
                else:
                    st.info("Совпадений не найдено.")

                # ----- Keyword -----
                exact_results = keyword_search(query, search_df)

                if exact_results:
                    st.markdown("### 🧷 Точный поиск")
                    for phrase, topics, comment in exact_results:
                        topics_pretty = [
                            topic_display_map.get(normalize_topic(t), t)
                            for t in topics
                        ]

                        st.markdown(
                            f"""
                            <div style="border:1px solid #e0e0e0;
                                        border-radius:12px;
                                        padding:16px;
                                        margin-bottom:12px;
                                        background:#f9f9f9;">
                                <div style="font-size:18px;font-weight:600;">📌 {phrase}</div>
                                <div style="font-size:14px;color:#666;">
                                    🔖 Тематики: <strong>{', '.join(topics_pretty)}</strong>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        if comment and str(comment).strip().lower() != "nan":
                            with st.expander("💬 Комментарий"):
                                st.markdown(comment)

        except Exception as e:
            st.error(f"Ошибка: {e}")

# ================= TAB 2 =================
with tab2:
    st.markdown("### 🚫 Локалы, которые **не используем**")
    unused_topics = [
        "Local_Balance_Transfer", "Local_Friends", "Local_Next_Payment",
        "Local_Order_Cash", "Local_Other_Cashback", "Local_RemittanceStatus",
        "Подожди (Wait)", "Local_X5", "PassportChangeFirst",
        "PassportChangeSecond", "Меньше (Local_Less)", "Больше (Local_More)",
        "Рефинансирование под залог недвижимости",
        "Действующий займ", "General Мои кредитные предложения",
        "Настроить/Изменить/Восстановить",
        "Как сделать устройство доверенным",
        "Что такое доверенное устройство",
        "Что такое секретный код",
        "Новая карта", "Проблема с начислением кэшбэка"
    ]
    for t in unused_topics:
        st.markdown(f"- {t}")

# ================= TAB 3 =================
def render_phrases_grid(phrases, cols=3, color="#e0f7fa"):
    rows = [phrases[i:i+cols] for i in range(0, len(phrases), cols)]
    for row in rows:
        columns = st.columns(cols)
        for col, phrase in zip(columns, row):
            col.markdown(
                f"""
                <div style="background:{color};
                            padding:6px 10px;
                            border-radius:12px;
                            margin:4px;
                            font-size:14px;">
                    {phrase}
                </div>
                """,
                unsafe_allow_html=True
            )

with tab3:
    st.markdown("### ✅ Интерпретации «ДА»")
    render_phrases_grid(
        [
            "Да", "Ага", "Угу", "Можно", "Готов",
            "Подскажите", "Расскажи", "Скажи", "Проверь"
        ],
        color="#d1f5d3"
    )

    st.markdown("### ❌ Интерпретации «НЕТ»")
    render_phrases_grid(
        ["Не надо", "Не хочу", "Не готов", "Не интересно"],
        color="#f9d6d5"
    )
