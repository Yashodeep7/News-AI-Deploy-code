import streamlit as st
from langchain_helper import news_ai_agent, NewsRAG
from build_vector_store import VectorStoreBuilder

st.set_page_config(page_title="📰 News AI", layout="wide")
st.title("📰 News AI")

# Add persistent session state for vector store check
if 'vector_store_ready' not in st.session_state:
    st.session_state['vector_store_ready'] = False

# Button to trigger building the vector store
if st.button("Get Latest News"):
    builder = VectorStoreBuilder()
    success = builder.build_vector_store()
    if success:
        st.session_state['vector_store_ready'] = True
        st.success("✅ Vector store built successfully!")
    else:
        st.session_state['vector_store_ready'] = False
        st.error("❌ Failed to build vector store!")

# Input and querying
user_query = st.text_input("Ask me anything about the news:")

if st.button("Generate"):
    if not st.session_state['vector_store_ready']:
        st.warning("⚠️ Please click 'Get Latest News' to build the vector store first.")
    elif not user_query.strip():
        st.warning("⚠️ Please enter a question!")
    else:
        with st.spinner("Generating response..."):
            try:
                result = news_ai_agent(user_query.strip())
                st.markdown("### ✅ Answer")
                st.write(result["answer"])

                if result["sources"]:
                    st.markdown("### 🗂 Sources")
                    for i, source in enumerate(result["sources"], 1):
                        st.markdown(f"{i}. [{source['heading']}]({source['url']}) — {source['source']}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
