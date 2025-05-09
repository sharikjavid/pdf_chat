from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from typing import Dict, List, Union
from langchain.schema.document import Document
from .prompt.prompts import generation_template

class PromptBuilder:
    @staticmethod
    def build_prompt(context_docs: Dict[str, List[Union[str, Document]]], user_question: str) -> ChatPromptTemplate:

        context_text = ""
        if context_docs.get("texts"):
            for text_element in context_docs["texts"]:
                if isinstance(text_element, Document):
                    context_text += text_element.page_content + "\n\n"
                else: # Should not happen if parse_docs works correctly
                    context_text += str(text_element) + "\n\n"

        prompt_template_text = generation_template.format(
            context_placeholder=context_text.strip(), 
            user_question=user_question
        )

        prompt_content = [{"type": "text", "text": prompt_template_text}]

        if context_docs.get("images"):
            # print(f"[DEBUG] prompt_builder.py: Adding {len(context_docs['images'])} images to prompt.")
            for image_b64 in context_docs["images"]:
                prompt_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    }
                )
        
        return ChatPromptTemplate.from_messages(
            [
                HumanMessage(content=prompt_content),
            ]
        )