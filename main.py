import streamlit as st
from langchain_helper import news_ai_agent
from build_vector_store import VectorStoreBuilder

# Set page title
st.set_page_config(page_title="📰 News AI", layout="centered")
st.title("📰 News AI")

# Session state to avoid rebuilding vector store repeatedly
if "store_built" not in st.session_state:
    st.session_state.store_built = False

# Build vector store on button click
if st.button("Get Latest News"):
    builder = VectorStoreBuilder()
    with st.spinner("Scraping news and building vector store..."):
        success = builder.build_vector_store()
    if success:
        st.session_state.store_built = True
        st.success("✅ Vector store built successfully!")
        st.info("You can now ask questions about the latest news.")
    else:
        st.session_state.store_built = False
        st.error("❌ Failed to build vector store!")

# Input text box for queries
user_query = st.text_input("Enter your question:")

# Generate answer
if st.button("Generate"):
    if not st.session_state.store_built:
        st.warning("Please click 'Get Latest News' to build the vector store first.")
    elif not user_query.strip():
        st.warning("Please enter a question!")
    else:
        with st.spinner("Generating response..."):
            try:
                result = news_ai_agent(user_query.strip())
                st.subheader("Answer:")
                st.markdown(result.get("answer", "No answer generated."))

                if result.get("sources"):
                    st.subheader("Sources:")
                    for i, source in enumerate(result["sources"], 1):
                        st.markdown(f"{i}. [{source['heading']}]({source['url']}) - {source['source']}")

            except Exception as e:
                st.error(f"Error: {str(e)}")
