def get_summary(prompt, url):
    pdf_pages = PyPDFLoader(url).load()
    vectorstore = process_pdf_with_metadata(pdf_pages)

    if vectorstore is None:
        return "Failed to process the PDF.", None

    retrieved_docs = retrieve_documents(vectorstore, prompt)
    print_retrieved_documents(retrieved_docs)

    decision = input("Do you want to modify the documents or proceed with processing? (modify/proceed): ").lower()

    if decision == 'modify':
        modified_docs = modify_documents(retrieved_docs)
    elif decision == 'proceed':
        modified_docs = retrieved_docs
    else:
        return None, "Invalid decision"

    # Generate summary from modified or original documents
    deploy_name = llm.GPT_4_OMNI_MODEL
    llm_instance = AzureChatOpenAI(
        deployment_name=deploy_name,
        openai_api_version="2023-07-01-preview",
        openai_api_key=openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        temperature=0.0,
        max_tokens=1000
    )

    docs_content = "\n\n".join([doc.page_content for doc in modified_docs])
    summary_prompt = f"{prompt}\n\nDocuments:\n{docs_content}"

    from langchain.callbacks import get_openai_callback

    with get_openai_callback() as cb:
        summary = llm_instance([HumanMessage(content=summary_prompt)])

    output_cost = (cb.completion_tokens / 1000) * OUTPUT_PRICE_PER_1K_TOKENS

    return summary.content, output_cost

# Main execution remains the same
url1 = 'https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20240731.pdf'
url2 = 'https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20240918.pdf'

prompt = "Summarize the participants' views on current conditions and the economic outlook."

summary1, cost1 = get_summary(prompt, url1)
print(f"Summary For First PDF:\n{summary1}")
print(f"Cost for summary 1: ${cost1:.4f}")

summary2, cost2 = get_summary(prompt, url2)
print(f"Summary For Second PDF:\n{summary2}")
print(f"Cost for summary 2: ${cost2:.4f}")

comparison_prompt = "Compare and contrast the views expressed in these two sets of minutes, focusing on any shifts in economic outlook or policy stance."

comparison_result, comparison_cost = compare(comparison_prompt, summary1, summary2)
print(f"\nComparison Result:\n{comparison_result}")
print(f"Cost for comparison: ${comparison_cost:.4f}")

total_cost = cost1 + cost2 + comparison_cost
print(f"\nTotal Cost: ${total_cost:.4f}")