import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing Component...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-5.2")

vector_store = PineconeVectorStore(
    embedding=embeddings,
    index_name=os.getenv("INDEX_NAME")
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def retrival_chain_without_LCEL(query):
    """Simple retrieval chain without using LCELLM.
    Limitations:
    - Manual Step by step execution.
    - No built in streaming support
    - No Async support without additional code.
    - Harder to compose with other chains
    - More verbose and error prone.
    """
    prompt_template = """Answer the question based on the following context: 
    {context}
    Question: {question}
    Provide a concise and accurate detail answer."""

    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format the retrieved documents into a context string
    context = format_docs(docs)

    # Step 3: Create the prompt by filling in the context and question
    message = prompt_template.format(context=context, question=query)

    # Step 4: Call the LLM with the formatted prompt
    response = llm.invoke(message)

    return response.content

def retrival_chain_with_LCEL():
    """Retrieval chain using LCELLM.
    Benefits:
    - Declarative and composable: Easy to chain operations with pipe operator(|).
    - Built in support for streaming and async operations.
    - Batch processing:chain.batch() for multiple inputs
    - Type Safety: Better integration with Langchain's type system
    - Less code: More concise and readable
    - Resusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: Langchain provides better observability tools.
    """

    prompt_template = ChatPromptTemplate.from_template("""Answer the question based on the following context: 
    {context}
    Question: {question}
    Provide a concise and accurate detail answer.""")

    # retrieval_chain = retriever | format_docs | prompt_template | llm | StrOutputParser()

    retrieval_chain = (
        RunnableParallel({
            "context":retriever | format_docs,
            "question":RunnablePassthrough()
        }) | prompt_template | llm | StrOutputParser())

    # retrieval_chain = (
    #     RunnablePassthrough.assign(
    #         context = itemgetter[str]("question") | retriever | format_docs
    #     ) | prompt_template | llm | StrOutputParser())

    return retrieval_chain

if __name__ == "__main__":
    print("Retrieving relevant documents...")
    print("Answering question without LCELLM...")
    
    query = "What is vLLM?"

    answer = retrival_chain_without_LCEL(query)
    print(f"Answer: {answer}")

    print("\nAnswering question with LCELLM...")
    answer = retrival_chain_with_LCEL().invoke(query)
    print(f"Answer: {answer}")
