import streamlit as st
from langchain_helper import news_ai_agent
from build_vector_store import VectorStoreBuilder
import gc
import torch

# Set page title
st.set_page_config(page_title="📰 News AI", layout="centered")
st.title("📰 News AI")

# Initialize session state for managing instances
if 'builder_instance' not in st.session_state:
    st.session_state.builder_instance = None

# Function to cleanup memory
def cleanup_memory():
    """Force garbage collection and clear GPU memory if available"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Build vector store on button click
if st.button("Get Latest News"):
    with st.spinner("Scraping news and building vector store..."):
        try:
            builder = VectorStoreBuilder()
            success = builder.build_vector_store()
            if success:
                st.success("✅ Vector store built and updated in repository!")
                st.info("The latest news data has been updated for everyone.")
            else:
                st.error("❌ Failed to build vector store!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
        finally:
            cleanup_memory()

# Input text box for queries
user_query = st.text_input("Enter your question:")

# Generate answer with proper error handling and memory management
if st.button("Generate"):
    if not user_query.strip():
        st.warning("Please enter a question!")
    else:
        with st.spinner("Generating response..."):
            try:
                # Process query
                result = news_ai_agent(user_query.strip())
                
                # Display results
                st.subheader("Answer:")
                if result.get("answer"):
                    st.markdown(result["answer"])
                else:
                    st.warning("No answer generated.")
                
                # Display sources if available
                sources = result.get("sources", [])
                if sources:
                    st.subheader("Sources:")
                    for i, source in enumerate(sources, 1):
                        heading = source.get('heading', 'Unknown Title')
                        url = source.get('url', '#')
                        source_name = source.get('source', 'Unknown Source')
                        
                        # Safe URL handling
                        if url and url != '#':
                            st.markdown(f"{i}. [{heading}]({url}) - {source_name}")
                        else:
                            st.markdown(f"{i}. {heading} - {source_name}")
                
                # Display verification status for debugging
                if st.checkbox("Show Debug Info"):
                    st.json({
                        'verification_status': result.get('verification_status', 'unknown'),
                        'is_from_sources': result.get('is_from_sources', False),
                        'sources_count': len(sources)
                    })
                
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                st.info("Please try again with a different question or check if the vector store is properly built.")
            
            finally:
                # Always cleanup memory after processing
                cleanup_memory()

# Add memory usage info in sidebar
with st.sidebar:
    st.header("System Info")
    if st.button("Check Memory Usage"):
        import psutil
        memory_info = psutil.virtual_memory()
        st.metric("Memory Usage", f"{memory_info.percent}%")
        st.metric("Available Memory", f"{memory_info.available / (1024**3):.2f} GB")
    
    st.header("Tips")
    st.info("""
    - Use "Get Latest News" sparingly to avoid memory issues
    - If you get errors, try refreshing the page
    - For best results, ask specific questions about recent news
    """)
