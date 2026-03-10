# ============================================================
# Streamlit UI — interactive Vietnamese sentiment analysis demo
# ============================================================
import streamlit as st
import pandas as pd
from model import predict, load_model, LABEL_NAMES

SENTIMENT_EMOJI = {"Negative": "😠", "Neutral": "😐", "Positive": "😊"}
SENTIMENT_COLOR = {"Negative": "#e74c3c", "Neutral": "#f39c12", "Positive": "#2ecc71"}


@st.cache_resource
def init_model():
    load_model()
    return True


def main():
    st.set_page_config(
        page_title="Vietnamese Sentiment Analysis",
        page_icon="🇻🇳",
        layout="centered"
    )

    st.title("🇻🇳 Vietnamese Sentiment Analysis")
    st.caption("Powered by PhoBERT — fine-tuned on UIT-VSFC dataset")

    init_model()

    # --- Single text input ---
    st.subheader("Analyze Text")
    text_input = st.text_area(
        "Enter Vietnamese text:",
        placeholder="Ví dụ: Thầy giảng bài rất dễ hiểu, tôi rất thích môn này",
        height=100
    )

    if st.button("Analyze", type="primary"):
        if text_input.strip():
            with st.spinner("Analyzing..."):
                result = predict(text_input)

            if "error" in result:
                st.error(result["error"])
            else:
                label = result["predicted_label"]
                conf = result["confidence"]
                emoji = SENTIMENT_EMOJI[label]
                color = SENTIMENT_COLOR[label]

                st.markdown(f"### {emoji} {label}")
                st.progress(conf, text=f"Confidence: {conf:.1%}")

                col1, col2, col3 = st.columns(3)
                for col, lbl in zip([col1, col2, col3], LABEL_NAMES):
                    prob = result["probabilities"][lbl]
                    col.metric(
                        label=f"{SENTIMENT_EMOJI[lbl]} {lbl}",
                        value=f"{prob:.1%}"
                    )

                with st.expander("Preprocessed text"):
                    st.code(result["cleaned_text"])
        else:
            st.warning("Please enter some text.")

    # --- Batch analysis ---
    st.divider()
    st.subheader("Batch Analysis")

    examples = [
        "Thầy dạy rất hay và dễ hiểu",
        "Môn học này quá khó, không hiểu gì cả",
        "Bình thường, không có gì đặc biệt",
        "Giảng viên nhiệt tình, tài liệu đầy đủ",
        "Phòng học nóng, máy chiếu hỏng liên tục",
    ]

    if st.button("Run examples"):
        results = []
        progress = st.progress(0)
        for i, text in enumerate(examples):
            result = predict(text)
            results.append({
                "Text": text[:60] + "..." if len(text) > 60 else text,
                "Sentiment": f"{SENTIMENT_EMOJI[result['predicted_label']]} {result['predicted_label']}",
                "Confidence": f"{result['confidence']:.1%}"
            })
            progress.progress((i + 1) / len(examples))

        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()