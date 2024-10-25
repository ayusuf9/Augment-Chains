def process_pdf_from_s3(pdf_key):
    filename = os.path.splitext(os.path.basename(pdf_key))[0]
    faiss_index_dir = f"{filename}_faiss_index"

    try:
        # Try to load existing FAISS index
        if os.path.isdir(faiss_index_dir):
            vector_store = FAISS.load_local(faiss_index_dir, embeddings=get_embedding_model(), allow_dangerous_deserialization=True)
            st.success("Loaded existing FAISS index.")
            return vector_store
        
        # Get PDF from S3
        obj = s3_appdata_client_qh.get_object(pdf_key)
        pdf_data = obj.read()
        
        # Create a temporary file to store the PDF
        temp_pdf_path = f"/tmp/{filename}.pdf"
        with open(temp_pdf_path, 'wb') as f:
            f.write(pdf_data)
        
        with st.spinner(text="Processing PDF..."):
            # Use the temporary file path with PyPDFLoader
            loader = PyPDFLoader(temp_pdf_path)
            documents = loader.load()
            
            batch_size = 10
            all_texts = []
            all_embeddings = []

            progress_bar = st.progress(0)
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                texts = [doc.page_content for doc in batch_docs]
                embeddings = embed_texts(texts, get_embedding_model())
                all_texts.extend(batch_docs)
                all_embeddings.extend(embeddings)
                progress = min((i + batch_size) / len(documents), 1.0)
                progress_bar.progress(progress)
                time.sleep(0.1)
            
            pure_texts = [page.page_content for page in all_texts]
            textual_embeddings = list(zip(pure_texts, all_embeddings))
            vector_store = FAISS.from_embeddings(textual_embeddings, get_embedding_model())
            
            # Save the FAISS index
            vector_store.save_local(faiss_index_dir)
            st.success(f"Processed PDF and saved FAISS index.")
            
            # Clean up the temporary file
            os.remove(temp_pdf_path)
            
            return vector_store
            
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None