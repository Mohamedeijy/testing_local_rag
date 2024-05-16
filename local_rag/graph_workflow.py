from typing import TypedDict, List

import langchain_core.documents
from langchain_community.llms import BaseLLM
from langchain_nomic.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import END, StateGraph
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        query: user query
        answer: LLM generation
        documents: retrieved documents from our rag and the internet
        web_search_needed: whether we need a websearch or not
        answer_grade: whether the final answer was judged relevant by our grader, True if yes

    """
    query: str
    answer: str
    documents: List[str]
    relevant_answer: bool


def get_workflow(model: BaseLLM, retriever: VectorStoreRetriever, embed_model: Embeddings) -> StateGraph:
    # ---- TEMPLATES AND CHAINS ---- #
    # Router
    router_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are dedicated to routing a query inputted by a user, either to a vectorstore or a web search.
    The vectorstore contains information related to statistical analysis and machine learning concepts and practices.
    Queries are routed to the vectorstore when they are questions on the aforementioned subjects. Otherwise, route the
    query to web search. You strictly return a binary choice between 'vectorstore' and 'websearch'. Your output should be a JSON
    with the only key being 'context_source', nothing else. <|eot_id|>
    
    <|start_header_id|>user<|end_header_id|>
    Here is the query you need to route: '{query}' <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    router_prompt = PromptTemplate.from_template(router_template)
    router = router_prompt | model | JsonOutputParser()

    # Rag
    rag_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a virtual assistant strictly designed to provide knowledge based on provided context from a database of documents.
    Answer the question based on the context below.
    If you have evaluated that the given context cannot help you provide a related answer, reply 'I don't know' and absolutely nothing else.<|eot_id|>
    
    <|start_header_id|>user<|end_header_id|>
    Query: '{query}'
    Context: {context}
    Answer:<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    rag_prompt = PromptTemplate.from_template(rag_template)
    rag = rag_prompt | model | StrOutputParser()

    # Grader
    grader_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader responsible for evaluating whether an answer is relevant to a given query.
    You strictly return a binary choice between 'yes' and 'no'. Your output should be a JSON
    with the only key being 'relevant_answer', nothing else.<|eot_id|>
    
    <|start_header_id|>user<|end_header_id|>
    Here is the answer:
    \n ------- \n
    {answer}
    \n ------- \n
    Here is the query: '{query}'<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    grader_prompt = PromptTemplate.from_template(grader_template)
    answer_grader = grader_prompt | model | JsonOutputParser()

    # ---- WEB SEARCH TOOL ---- #
    # Web search
    web_search_tool = TavilySearchAPIRetriever(k=3)

    # ---- ADDING NODES ---- #
    # Nodes
    def retrieve_documents(state: dict) -> dict:
        """
        Retrieve document from vector store.

        :param state: Current state graph
        :return: State graph with documents key added
        """
        print("---RETRIEVING DOCUMENTS FROM VECTOR STORE---")
        query = state["query"]

        documents: list[langchain_core.documents.Document] = retriever.invoke(query)
        return {"query": query, "documents": documents}

    def web_search(state: dict) -> dict:
        """
        Search for the web with a query and retrieve top k documents.

        :param state: Current state graph
        :return: State graph with websearch key added
        """
        print("---WEB SEARCH FOR ADDITIONAL INFO---")
        query = state["query"]
        documents = state["documents"]

        web_results: list[langchain_core.documents.Document] = web_search_tool.invoke(query)
        if documents is None:
            documents = web_results
        else:
            documents.extend(web_results)

        return {"query": query, "documents": documents}

    def generate_answer(state: dict) -> dict:
        """
        Call LLM to generate answer based on query and context.

        :param state: Current state graph
        :return: State graph with websearch key added
        """
        print("---GENERATING LLM ANSWER BASED ON RETRIEVED CONTEXT---")
        query = state["query"]
        documents = state["documents"]

        answer = rag.invoke(input={"query": query, "context": documents})

        return {"query": query, "documents": documents, "answer": answer}

    def grade_answer(state: dict) -> dict:
        """
        Call LLM to determine if answer is relevant to context or no.

        :param state: Current state graph
        :return: State graph with websearch key added
        """
        print("---GRADING ANSWER BASED ON RELEVANCE TO CONTEXT---")
        query = state["query"]
        documents = state["documents"]
        answer = state["answer"]

        grade: dict = answer_grader.invoke(input={"query": query, "answer": answer})
        assert 'relevant_answer' in grade and grade[
            'relevant_answer'] == 'yes' or 'no', 'LLM response should be a relevant_answer JSON key with yes or no as value'
        relevant_answer = grade['relevant_answer'] == 'yes'

        print(f"User query: '{query}'\n")
        print(f"""Final answer:
        \n ------- \n
        {answer}
        \n ------- \n
        Was this answer judged relevant ? : {grade['relevant_answer']}
        """)
        return {"query": query, "documents": documents, "answer": answer, "relevant_answer": relevant_answer}
        # return query

    # ---- CONDITIONAL EDGES ---- #
    def route_to_research_or_rag(state: dict) -> str:
        print("---ROUTE QUERY---")
        query = state['query']
        print(f"Given query: {query}\n")
        decision = router.invoke(input={"query": query})
        print(f"Decision: {decision}")
        assert 'context_source' in decision and decision['context_source'] == 'vectorstore' or 'websearch', \
            "LLM response should be a context_source JSON key with vectorstore or websearch as value"
        route = decision['context_source']
        print(f"---ROUTE QUERY TO  {str.upper(route)}---")
        return route

    # ---- BUILDING WORKFLOW GRAPH ---- #
    workflow = StateGraph(GraphState)
    # Define the nodes
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("grade_answer", grade_answer)

    # add nodes
    workflow.set_conditional_entry_point(route_to_research_or_rag,
                                         {
                                             "websearch": "web_search",
                                             "vectorstore": "retrieve_documents",
                                         })

    workflow.add_edge("retrieve_documents", "generate_answer")
    workflow.add_edge("web_search", "generate_answer")
    workflow.add_edge("generate_answer", "grade_answer")
    workflow.add_edge("grade_answer", END)
    return workflow
