import os
from operator import itemgetter

import uvicorn
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langserve import add_routes
from graph_workflow import get_workflow
from model_chain import get_model, get_embed_model, get_retriever
from graph_workflow import get_workflow
from langgraph.graph import StateGraph
from langchain_core.runnables import chain
from dotenv import load_dotenv, dotenv_values


def main():
    load_dotenv()
    model = get_model()
    embed_model = get_embed_model()
    retriever = get_retriever(embed_model=embed_model)

    workflow: StateGraph = get_workflow(model=model, embed_model=embed_model, retriever=retriever)
    graph = workflow.compile()

    @chain
    def custom_chain(text):
        inputs = {"query": f"{text}"}
        return graph.invoke(inputs)["answer"]

    app = FastAPI(
        title="Local Assistant",
        version="1.0",
        description="Local Assistant",
    )

    add_routes(
        app,
        custom_chain.with_types(input_type=dict,output_type=str),
        # custom_chain.with_types(input_type=str,output_type=str), # endpoint type change for default playground
        # graph.with_types(input_type=str),
        path="/local_assistant",
        playground_type="chat",
        # enable_feedback_endpoint=True,
        # enable_public_trace_link_endpoint=True,
    )
    uvicorn.run(app, host="localhost", port=8000)


if __name__ == '__main__':
    main()
