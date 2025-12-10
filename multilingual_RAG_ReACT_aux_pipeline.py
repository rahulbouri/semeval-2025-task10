import os
import ast
import numpy as np
import pandas as pd
from openai import AzureOpenAI
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from bert_score import score
from tqdm import tqdm

import time
import random
from azure.core.exceptions import ServiceRequestError, AzureError

from huggingface_hub import login

login("")

import logging
log_path = ""

logging.basicConfig(
    filename=log_path,
    level=logging.INFO, 
    format="%(asctime)s - %(message)s",
    force=True  # Ensures no previous logging settings interfere
)

# Disable logging from external libraries
for logger_name in ["openai", "azure", "httpx", "http.client", "urllib3"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)  # Suppress lower-level logs
    logging.getLogger(logger_name).disabled = True

# Suppress Azure SDK HTTP logging (this is where the HTTP request logs originate)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.CRITICAL)
logging.getLogger("azure").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)

pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)


os.environ["AZURE_OPENAI_KEY"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["AZURE_API_VERSION"] = ""
os.environ["AZURE_DEPLOYMENT_ID"] = ""
os.environ["AWS_ACCESS_KEY"] = ""
os.environ["AWS_SECRET_KEY"] = ""
os.environ["AWS_SESSION_TOKEN"] = ""
os.environ["AWS_REGION"] = "ap-south-1"
model_name = "gpt-4o-mini"

client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_ID")
)



openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_type="azure",
                api_version=os.getenv("AZURE_API_VERSION"),
                model_name="text-embedding-3-large"
            )


def clean_and_parse(x):
    import ast 

    if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
        cleaned = x.replace('""', '"')
        cleaned = cleaned.replace('\n', ' ').replace('\r', '')  # Remove line breaks, if any

        try:
            return ast.literal_eval(cleaned)
        except Exception as e:
            print(f"Error parsing string: {cleaned}\n{e}")
            return x
    else:
        return x

def get_main_taxonomy_examples(dom: str, mode: str) -> str:

    if mode == "CC":
    
        df = pd.read_csv("auxilliary_knowledge_base/cc_taxonomy.csv")
        df = df.astype(str)
        df['Main Narrative Example'] = df['Main Narrative Example'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else x)
                
        taxonomy = ''
        examples = ''

        dom = dom.split(":")[0] if len(dom.split(":")) > 0 else dom

        if dom in df['Main Narrative'].values:
            temp = df[df['Main Narrative'] == dom].reset_index()         
            taxonomy = taxonomy + f"Category: {temp.loc[0, 'Main Narrative']} | Definition: {temp.loc[0, 'Main Narrative Definition']}\n"
            if isinstance(temp.loc[0, 'Main Narrative Example'], list):
                for item in temp.loc[0, 'Main Narrative Example']:
                    examples = examples + f"{item} => {temp.loc[0, 'Main Narrative']}\n"
                if not temp.loc[0, 'Detail Instructions Main Narrative'] == 'nan':
                    examples = examples + f"Note: {temp.loc[0, 'Detail Instructions Main Narrative']}\n"
            else:
                pass


        return (taxonomy, examples)

    else:

        df = pd.read_csv("auxilliary_knowledge_base/urw_taxonomy.csv")
        df = df.astype(str)
        df['Main Narrative Example'] = df['Main Narrative Example'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else x)
          
        taxonomy = ''
        examples = ''

        dom = dom.split(":")[0] if len(dom.split(":")) > 0 else dom

        if dom in df['Main Narrative'].values:
            temp = df[df['Main Narrative'] == dom].reset_index()           
            taxonomy = taxonomy + f"Category: {temp.loc[0, 'Main Narrative']} | Definition: {temp.loc[0, 'Main Narrative Definition']}\n"
            if isinstance(temp.loc[0, 'Main Narrative Example'], list):
                for item in temp.loc[0, 'Main Narrative Example']:
                    examples = examples + f"{item} => {temp.loc[0, 'Main Narrative']}\n"
                if not temp.loc[0, 'Detail Instructions Main Narrative'] == 'nan':
                    examples = examples + f"Note: {temp.loc[0, 'Detail Instructions Main Narrative']}\n"
            else:
                pass
            
        return (taxonomy, examples)

def get_sub_taxonomy_examples(sub: str, mode: str) -> str:

    if mode == "CC":
        df = pd.read_csv("auxilliary_knowledge_base/cc_taxonomy.csv")
        df = df.astype(str)
        df['Sub Narrative Example'] = df['Sub Narrative Example'].apply(clean_and_parse)

        taxonomy = ''
        examples = ''

        sub = sub.split(":")[2][1:] if len(sub.split(":")) > 1 else sub

        if sub in df['Sub Narrative'].values:
            temp = df[df['Sub Narrative'] == sub].reset_index()
            taxonomy = taxonomy + f"Category: {temp.loc[0, 'Sub Narrative']} | Definition: {temp.loc[0, 'Sub Narrative Definition']}\n"
            if isinstance(temp.loc[0, 'Sub Narrative Example'], list):
                for item in temp.loc[0, 'Sub Narrative Example']:
                    examples = examples + f"{item} => {temp.loc[0, 'Sub Narrative']}\n"
                if not temp.loc[0, 'Detail Instructions Sub Narrative'] == 'nan':
                    examples = examples + f"Note: {temp.loc[0, 'Detail Instructions Sub Narrative']}\n"
            else:
                pass
        return (taxonomy, examples)

    else:
        df = pd.read_csv("auxilliary_knowledge_base/urw_taxonomy.csv")
        df = df.astype(str)
        df['Sub Narrative Example'] = df['Sub Narrative Example'].apply(clean_and_parse)

        taxonomy = ''
        examples = ''

        sub = sub.split(":")[2][1:] if len(sub.split(":")) > 1 else sub

        if sub in df['Sub Narrative'].values:
            temp = df[df['Sub Narrative'] == sub].reset_index()
            taxonomy = taxonomy + f"Category: {temp.loc[0, 'Sub Narrative']} | Definition: {temp.loc[0, 'Sub Narrative Definition']}\n"
            if isinstance(temp.loc[0, 'Sub Narrative Example'], list):
                for item in temp.loc[0, 'Sub Narrative Example']:
                    examples = examples + f"{item} => {temp.loc[0, 'Sub Narrative']}\n"
                if not temp.loc[0, 'Detail Instructions Sub Narrative'] == 'nan':
                    examples = examples + f"Note: {temp.loc[0, 'Detail Instructions Sub Narrative']}\n"
            else:
                pass
        return (taxonomy, examples)


def create_prompt(res, dom, sub, mode, language):
    # Helper function replacing quotation marks in the text:


    if mode == "CC":
        # Update predicted_labels by slicing from the 4th character
        main_taxonomy, _ = get_main_taxonomy_examples(dom, mode)
        sub_taxonomy, _ = get_sub_taxonomy_examples(sub, mode)
        context = f"""You will be given the dominant narrative and sub narrative associated with an article along with a list of sentences supporting the dominant narrative from the article.
        
        GOAL: Justify the choice of dominant and sub narratives assigned to the article using the sentences given to you. Provide reasoning and quote relevant text from the original ssentences as to why these are the correct choice of dominant and sub narratives for the article using the list of sentences given to you.

            INSTRUCTIONS:
            Read the provided text carefully.
            You will be given: 
            1.Taxonomy - definition of the narrative
            2.Any additional idenyifying information for the narratives.
            Based on the given information give an explanation as to why the dominant and sub narratives are the correct choice for the article using the list of sentences given to you.
            
            Note: Keep the output concise and to the point - use relevant textual examples directly from the list of sentences given.

            DOMINANT NARRATIVE: {dom[4:]}
            TAXONOMY: {main_taxonomy}

            SUB NARRATIVE: {sub[4:]}
            TAXONOMY: {sub_taxonomy}
        """

        prompt = f'''{context}
        -------------------------------------------------------
        Based on the given Instructions and Taxonomies: Justify the choice of dominant and sub narratives assigned to the Climate Change article using the given list of sentences supporting the dominant narrative taken fron the article. 

        Output Format: Return text in {language} with a paragraph format in strictly 75 words or under. Focus on quoting sentences directly.

        LIST OF SENTENCES SUPPORTING DOMINANT NARRATIVE FROM ARTICLE: "{res}" => '''
        
        return {
            "role": "user",
            "content": prompt
        }
        
    else:
        # Update predicted_labels by slicing from the 5th character
        main_taxonomy, _ = get_main_taxonomy_examples(dom, mode)
        sub_taxonomy, _ = get_sub_taxonomy_examples(sub, mode)

        context = f"""You will be given the dominant narrative and sub narrative associated with an article along with a list of sentences supporting the dominant narrative from the article.
        
        GOAL: Justify the choice of dominant and sub narratives assigned to the article using the sentences given to you. Provide reasoning and quote relevant text from the original ssentences as to why these are the correct choice of dominant and sub narratives for the article using the list of sentences given to you.

            INSTRUCTIONS:
            Read the provided text carefully.
            You will be given: 
            1.Taxonomy - definition of the narrative
            2.Any additional idenyifying information for the narratives.
            Based on the given information give an explanation as to why the dominant and sub narratives are the correct choice for the article using the list of sentences given to you.
            
            Note: Keep the output concise and to the point - use relevant textual examples directly from the list of sentences given.

            DOMINANT NARRATIVE: {dom[4:]}
            TAXONOMY: {main_taxonomy}

            SUB NARRATIVE: {sub[4:]}
            TAXONOMY: {sub_taxonomy}
        """

        prompt = f'''{context}
        -------------------------------------------------------
        Based on the given Instructions and Taxonomies: Justify the choice of dominant and sub narratives assigned to the Ukraine Russia War article using the given list of sentences supporting the dominant narrative taken fron the article. 
        Output Format: Return text in {language} with a paragraph format in strictly 75 words or under. Focus on quoting sentences directly.

        LIST OF SENTENCES SUPPORTING DOMINANT NARRATIVE FROM ARTICLE: "{res}" => '''
        
        return {
            "role": "user",
            "content": prompt
        }

def get_embedded_json(embedded_str):
    import re
    try:
        # Extract only the list content using regex
        match = re.search(r"\[.*\]", embedded_str)
        if not match:
            return []  # Return empty list if no valid list is found
        
        cleaned_str = match.group(0)  # Extract the matched list portion

        # Convert to Python list safely
        return ast.literal_eval(cleaned_str)
    
    except (ValueError, SyntaxError):
        return []  # Return empty list if parsing fails
    

def add_documents_with_retry(collection, documents, ids, max_retries=3):
    assert len(documents) == len(ids), "The number of documents must match the number of IDs."

    # Wrap tqdm around the loop
    for doc, doc_id in tqdm(zip(documents, ids), total=len(documents), desc="Adding documents to chromadb", unit="doc"):
        retries = 0
        while retries < max_retries:
            try:
                collection.add(documents=[doc], ids=[doc_id])
                break  
            except AzureError as e:
                if "rate limit" in str(e).lower() or isinstance(e, ServiceRequestError):
                    retries += 1
                    wait_time = 3
                    print(f"\nRate limit exceeded. Retrying in {wait_time:.2f} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"\nFailed to add document with ID {doc_id}: {e}")
                    break  # Non-retryable error
            except Exception as e:
                print(f"\nUnexpected error while adding document with ID {doc_id}: {e}")
                break  # Unexpected error, stop retrying

        if retries == max_retries:
            print(f"\nExceeded maximum retries for document with ID {doc_id}. Skipping...")
    
    total_docs = collection.count()
    logging.info(f"Total documents in ChromaDB after addition: {total_docs}")


def split_and_retrieve_sentences(text: str, narrative: str, subnarrative: str, threshold: float):
    sentences = [s.strip() for s in text.replace('\n', '').split('.') if s.strip()]
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="text-embedding-3-large", embedding_function=openai_ef, metadata={"hnsw:space": "cosine"})
    ids = [f"sentence_{i}" for i in range(len(sentences))]

    add_documents_with_retry(collection=collection, documents=sentences, ids=ids, max_retries=10)
    
    search_results = collection.query(
        query_texts=[narrative],
        n_results=5
    )
    top_documents = search_results['documents'][0].copy()
    top_scores = [1 - distance for distance in search_results['distances'][0]].copy()

    sub_search_results = collection.query(
        query_texts=[subnarrative],
        n_results=5
    )
    
    # Check if any result exceeds the threshold and add to search_results
    for doc, score, doc_id in zip(sub_search_results['documents'][0], sub_search_results['distances'][0], sub_search_results['ids'][0]):
        if 1 - score > threshold:
            # Add to search results if score exceeds the threshold
            top_documents.append(doc)
            top_scores.append(score)
            
    chroma_client.delete_collection("text-embedding-3-large")

    docs_df = pd.DataFrame({"document":top_documents, "score":top_scores})

    docs_df = docs_df.sort_values(by='score', ascending=False).reset_index(drop=True)

    top_docs = docs_df['document'].to_list()

    return top_docs

def generate_response(text:str, dom, sub, mode, temp, language):
    system_main = \
f'''You are an expert trained to analyse and justify the choice of dominant and sub narratives assigned to a given article within 80 words.

Instructions:
Write in {language} only.
Use ReACT (Reasoning and Contextual Text) to justify the choice of dominant and sub narratives assigned to the article.

Example ReACT Flow:
1. Identify the central claim i.e dominant narrative.
  - Thought: The text discusses the <dominant narrative> as the central claim.
  - Action: Identify the supporting evidence.
  - Observation: The text provides <evidence> to support the <dominant narrative>.  
2. Identify the supporting claim i.e sub narrative.
    - Thought: The text discusses the <sub narrative> as the supporting claim.
    - Action: Identify the supporting evidence.
    - Observation: The text provides <evidence> to support the <sub narrative>.

Use the "Observations" obtained from ReACT Flow to justify the choice of dominant and sub narratives assigned to the article.
Provide reasoning and quote relevant text from the original text as to why these are the correct choice of dominant and sub narratives for the text file.
Keep the output concise and to the point—ideally find relevant textual examples.

Categorization Rules:
Use the help of the provided taxonomy and examples to justify the choice of dominant and sub narratives assigned to the article.

OUTPUT FORMAT:
Return {language} language text in paragraph(s) format within 80 words.
'''

    relevant_sentences = split_and_retrieve_sentences(text, dom, sub, 0.25)

    prompt = create_prompt(relevant_sentences, dom, sub, mode, language)

    messages = [
        {"role": "system", "content": system_main},
        {"role": "user", "content": prompt.get("content", "")}
    ]
    
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=temp,
    max_tokens=100
    )   

    output = str(response.choices[0].message.content)
    return output




mapping = {
    "HINDI": ("<path>", "hi"),
    "ENGLISH": ("<path>", "en"),
    "PORTUGUESE": ("<path>", "pt"),
    "RUSSIAN": ("<path>", "ru"),
    "BULGARIAN": ("<path>", "bg")
}

data_list = []

for key, value in tqdm(mapping.items(), total=len(mapping)):
    try:
        logging.info(f"Processing started for language: {key}")
        print("Processing for language: ", key)

        df = pd.read_csv(value[0])
        df['mode'] = df['dominant_narrative'].apply(lambda x: 'CC' if x.split(':')[0] == 'CC' else 'URW')

        temp = 0.3

        tqdm.pandas()
        df['output'] = df.progress_apply(lambda x: generate_response(x['text'], x['dominant_narrative'], 
                                                                     x['sub_narratives'], x['mode'], temp, 
                                                                     language=value[0]), axis=1)
        
        predictions = df['output'].tolist()
        references = df['ground_truth'].tolist()

        P, R, F1 = score(predictions, references, lang=value[1])

        df['precision'] = P.tolist()
        df['recall'] = R.tolist()
        df['f1'] = F1.tolist()

        mean_precision = df['precision'].mean()
        mean_recall = df['recall'].mean()
        mean_f1 = df['f1'].mean()

        logging.info(f"Results for {key}: Precision={mean_precision:.4f}, Recall={mean_recall:.4f}, F1={mean_f1:.4f}")

        data = {"Language": key,
                "Mean Precision": mean_precision,
                "Mean Recall": mean_recall,
                "Mean F1": mean_f1}
        
        data_list.append(data)

        logging.info(f"Processing completed for language: {key}")

    except Exception as e:
        logging.error(f"Error processing {key}: {str(e)}", exc_info=True)

final = pd.DataFrame(data_list)
final.to_csv("<final_results>", index=False)
logging.info("Final results saved to CSV.")

    

