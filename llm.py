from operator import itemgetter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import format_document
from langchain.prompts.prompt import PromptTemplate
from langdetect import detect

condense_question = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question)

answer_template = {
    "en": """
### Instruction:
You are a helpful research assistant who answers questions based on the provided research in a clear and easy-to-understand manner.
If there is no research, or the research is not relevant to answer the question, simply state that you cannot answer.
Please answer with a detailed response and list your sources and the paragraphs where the information is derived from. If you are unable to answer the question, do not mention sources.

Please provide the response in English.

## Research:
{context}

## Question:
{question}

## Answer:
""",
    "fr": """
### Instruction:
Vous êtes un assistant de recherche utile, qui répond aux questions basées sur les recherches fournies de manière claire et facile à comprendre.
S'il n'y a pas de recherche, ou si la recherche n'est pas pertinente pour répondre à la question, répondez simplement que vous ne pouvez pas répondre.
Veuillez répondre simplement avec la réponse détaillée et listez vos sources et les paragraphes d'où proviennent les informations. Si vous n'êtes pas en mesure de répondre à la question, ne mentionnez pas les sources.

Veuillez fournir la réponse en français.

## Recherche:
{context}

## Question:
{question}

## Réponse:
"""
}

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="Source Document: {source}, Page {page}:\n{page_content}"
)

def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    # Format each document with the given prompt template, including source and page
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)



def getStreamingChain(question: str, memory, llm, db):
    # Detect the language of the question
    language = detect(question)
    language_code = "fr" if language == "fr" else "en"

    # Select the appropriate answer prompt template
    answer_prompt = ChatPromptTemplate.from_template(answer_template[language_code])

    retriever = db.as_retriever(search_kwargs={"k": 10})
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(
            lambda x: "\n".join(
                [f"{item['role']}: {item['content']}" for item in x["memory"]]
            )
        ),
    )

    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
    }

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
        "sources": lambda x: ", ".join(set([doc.metadata["source"] for doc in x["docs"]]))  # Collecter les noms des documents
    }

    # Retourner la réponse avec les sources
    answer = final_inputs | answer_prompt | llm

    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    return final_chain.stream({"question": question, "memory": memory})


def getChatChain(llm, db):
    retriever = db.as_retriever(search_kwargs={"k": 10})

    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables)
        | itemgetter("history"),
    )

    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
    }

    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }

    # Select the appropriate answer prompt template based on language detection
    def get_answer_prompt(question):
        language = detect(question)
        language_code = "fr" if language == "fr" else "en"
        return ChatPromptTemplate.from_template(answer_template[language_code])

    # And finally, we do the part that returns the answers
    answer = {
        "answer": final_inputs
        | get_answer_prompt(itemgetter("question"))
        | llm.with_config(callbacks=[StreamingStdOutCallbackHandler()]),
        "docs": itemgetter("docs"),
    }

    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    def chat(question: str):
        inputs = {"question": question}
        result = final_chain.invoke(inputs)
        memory.save_context(inputs, {"answer": result["answer"]})

    return chat
