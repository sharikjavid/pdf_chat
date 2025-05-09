from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from .document_parser import DocumentParser
from .prompt_builder import PromptBuilder
from .resource_loader import ResourceLoader 


class RAGChainManager:
    def __init__(self, resource_loader: ResourceLoader):
        self.resource_loader = resource_loader
        self.retriever = resource_loader.get_retriever()
        self.llm = resource_loader.get_llm()
        self._chain = self._build_chain()

    def _build_chain(self):
        def prepare_context_and_question(input_dict):
            parsed_docs = DocumentParser.parse_docs(input_dict["retrieved_docs"])
            return {"context": parsed_docs, "question": input_dict["question"]}
        def build_prompt_from_prepared(input_dict):
            return PromptBuilder.build_prompt(input_dict["context"], input_dict["question"])

        chain = (
            {
                "retrieved_docs": self.retriever, 
                "question": RunnablePassthrough() 
            }
            | RunnableLambda(prepare_context_and_question) # Output: {"context": {"images": [], "texts": []}, "question": ...}
            | RunnableLambda(build_prompt_from_prepared) 
            | self.llm
            | StrOutputParser()
        )
        return chain

    def invoke(self, question: str):
        return self._chain.invoke(question)

    def retrieve_documents(self, question: str):
        raw_docs = self.retriever.invoke(question)
        return DocumentParser.parse_docs(raw_docs)