import os
import time
from dotenv import load_dotenv
from typing import List, Dict, Any
from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from config import AppConfig
from rag_components.resource_loader import ResourceLoader
from rag_components.retrieval_chain import RAGChainManager
from langchain.schema.document import Document # For type hinting

os.environ["OPENAI_API_KEY"] = os.env[OPENAI_API_KEY]

def format_retrieval_context_for_deepeval(parsed_docs: Dict[str, List[Any]]) -> List[str]:
    context_strings: List[str] = []
    if parsed_docs and "texts" in parsed_docs:
        for doc in parsed_docs["texts"]:
            if isinstance(doc, Document):
                context_strings.append(doc.page_content)
            elif isinstance(doc, str): 
                context_strings.append(doc)
    print(len(context_strings))
    return context_strings


def run_evaluation():
    load_dotenv()
    print("☑️ Environment variables loaded.")

    if not os.getenv("OPENAI_API_KEY"):
        print("🔴 WARNING: OPENAI_API_KEY environment variable not set.")
        print("   DeepEval metrics using OpenAI models (e.g., gpt-4o) will likely fail.")
        print("   Please set it in your .env file or your environment.")
    print("\n🔄 Initializing RAG system...")
    try:
        config = AppConfig()
        resource_loader = ResourceLoader(config)
        resource_loader.load_all()  # Explicitly load all resources
        rag_manager = RAGChainManager(resource_loader)
        print("✅ RAG system initialized successfully.")
    except Exception as e:
        print(f"🔴 FATAL: Error initializing RAG system: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n📚 Defining test dataset...")

    golden_dataset = [
 
  
{
    "input": "What is machine learning according to the document?",
    "expected_output": "Machine learning is programming computers to optimize a performance criterion using example data or past experience. [cite: 1545, 1546] It is the field of study that gives computers the ability to learn without being explicitly programmed. [cite: 1549]"
  },
 
  {
    "input": "Explain the concept of abstraction in the learning process.",
    "expected_output": "Abstraction is the process of extracting knowledge about stored data by creating general concepts about the data as a whole. [cite: 1561, 1562] This involves applying known models and creating new ones, with fitting a model to a dataset being known as training. [cite: 1563, 1564]"
  },

  {
    "input": "How is evaluation defined in the learning process?",
    "expected_output": "Evaluation is the process of providing feedback to the user to measure the usefulness of the learned knowledge, which is then used to improve the overall learning process. [cite: 1570, 1571]"
  },
  {
    "input": "What are the three main categories of learning models discussed in the document?",
    "expected_output": "The three main categories of learning models are Logical models, Geometric models, and Probabilistic models. [cite: 1587, 1588, 1589]"
  },

  {
    "input": "How do Geometric models define similarity?",
    "expected_output": "Geometric models define similarity by considering the geometry of the instance space, where features can be described as points in a multi-dimensional space. [cite: 1616, 1617] Similarity can be imposed using geometric concepts like lines or planes to segment the space (Linear models) or using the geometric notion of distance (Distance-based models). [cite: 1620, 1621, 1622, 1623]"
  },
  {
    "input": "Explain Linear models.",
    "expected_output": "Linear models are a type of Geometric model where the function is represented as a linear combination of its inputs. [cite: 1624, 1625] They are parametric models with a fixed form and a small number of numeric parameters to be learned from data, unlike tree or rule models where the structure is not fixed. [cite: 1628, 1629, 1630] Linear models are stable and less likely to overfit but more likely to underfit. [cite: 1631, 1633, 1634]"
  },

   
]

    if not golden_dataset or not golden_dataset[0]["input"].startswith("What is the main topic"):
        print("⚠️ WARNING: Please replace the placeholder golden_dataset with questions and answers relevant to YOUR PDF content for meaningful evaluation!")
   



    print("\n📊 Initializing DeepEval metrics...")
  
    correctness_metric = GEval(
        name="Correctness (vs Expected)",
        model="gpt-3.5-turbo", 
        evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_steps=[
            "Examine the 'Actual Output' and the 'Expected Output'.",
            "Determine if the 'Actual Output' is factually consistent with the 'Expected Output'.",
            "Assess if the 'Actual Output' omits any critical information from, or introduces any inaccuracies not present in, the 'Expected Output'.",
            "The score should reflect the degree of factual alignment and completeness of the 'Actual Output' in relation to the 'Expected Output'."
        ]
    )

    # Faithfulness: Is the answer faithful to the retrieved context?
    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7, 
        model="gpt-3.5-turbo",
        include_reason=True 
    )

    # Contextual Relevancy: Is the retrieved context relevant to the input question?
    contextual_relevancy_metric = ContextualRelevancyMetric(
        threshold=0.7,
        model="gpt-3.5-turbo",
        include_reason=True
    )
    
    # List of all metrics to run
    all_metrics = [correctness_metric, faithfulness_metric, contextual_relevancy_metric]
    print("✅ DeepEval metrics initialized.")

    # 4. Generate LLMTestCases by running the RAG system for each item in the golden dataset
    print("\n⚙️ Generating LLMTestCases by querying the RAG system...")
    test_cases: List[LLMTestCase] = []
    for i, item in enumerate(golden_dataset):
        question = item["input"]
        expected_answer = item["expected_output"]
        
        
        try:
            start_time = time.time()
            # Get actual output from the RAG chain
            actual_answer = rag_manager.invoke(question)
            latency = time.time() - start_time

            # Get retrieved context
            # retrieve_documents() from our RAGChainManager returns a dict: {"images": ..., "texts": ...}
            parsed_retrieved_docs = rag_manager.retrieve_documents(question)
            retrieval_context_str_list = format_retrieval_context_for_deepeval(parsed_retrieved_docs)

            test_case = LLMTestCase(
                input=question,
                actual_output=actual_answer,
                expected_output=expected_answer,
                retrieval_context=retrieval_context_str_list,

            )
            test_cases.append(test_case)
            print(f"    ⏱️ Latency: {latency:.2f}s")

        except Exception as e:
            print(f"    🔴 Error processing test case '{question}': {e}")
            # Create a failed test case or skip
            test_cases.append(LLMTestCase(
                input=question,
                actual_output=f"Error during generation: {e}",
                expected_output=expected_answer,
                retrieval_context=[] 
            ))
    print(f"✅ LLMTestCases generated ({len(test_cases)} total).")

    # 5. Run Evaluation using DeepEval
    if not test_cases:
        print("\n🔴 No test cases were successfully generated. Skipping DeepEval evaluation.")
        return

    print("\n🚀 Running evaluation with DeepEval...")
    print("   This may take some time as it involves calls to the evaluation LLM (e.g., GPT-4o).")
    try:

        evaluation_results_list = evaluate(test_cases=test_cases, metrics=all_metrics)
      
        

        
    except Exception as e:
        print(f"🔴 An error occurred during DeepEval's `evaluate` call: {e}")
        print("   This might be due to issues with API keys for evaluation models (e.g., OpenAI for GPT-4o),")
        print("   network problems, or misconfiguration of DeepEval metrics or test cases.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_evaluation()