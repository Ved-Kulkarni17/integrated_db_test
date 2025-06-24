# from mlflow.deployments import get_deploy_client

# def _query_endpoint(endpoint_name: str, messages: list[dict[str, str]], max_tokens) -> list[dict[str, str]]:
#     """Calls a model serving endpoint."""
#     res = get_deploy_client('databricks').predict(
#         endpoint=endpoint_name,
#         inputs={'messages': messages, "max_tokens": max_tokens},
#     )
#     if "messages" in res:
#         return res["messages"]
#     elif "choices" in res:
#         return [res["choices"][0]["message"]]
#     raise Exception("This app can only run against:"
#                     "1) Databricks foundation model or external model endpoints with the chat task type (described in https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models#chat-completion-model-query)"
#                     "2) Databricks agent serving endpoints that implement the conversational agent schema documented "
#                     "in https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent")

# def query_endpoint(endpoint_name, messages, max_tokens):
#     """
#     Query a chat-completions or agent serving endpoint
#     If querying an agent serving endpoint that returns multiple messages, this method
#     returns the last message
#     ."""
#     return _query_endpoint(endpoint_name, messages, max_tokens)[-1]

import json
import logging
from mlflow.deployments import get_deploy_client
from typing import Union, Dict, List

logger = logging.getLogger(__name__)

def query_endpoint(endpoint_name: str, **kwargs) -> Union[str, Dict]:
    """
    Dispatches query to the appropriate format (structured vs unstructured).
    """

    if "encoded_pdf" in kwargs:
        return _query_unstructured(endpoint_name, kwargs)

    elif "question" in kwargs and "generate_insights" in kwargs:
        return _query_structured(endpoint_name, kwargs)

    raise ValueError("Unsupported input format to query_endpoint")

# 👇 Unstructured (PDF chat)
def _query_unstructured(endpoint_name: str, args: Dict) -> str:
    client = get_deploy_client("databricks")
    payload = {
        "dataframe_records": [
            {
                "encoded_pdf": args["encoded_pdf"],
                "file_name": args["file_name"],
                "question": args["question"],
                "chat_history": json.dumps(args.get("history", []), ensure_ascii=False)
            }
        ]
    }
    logger.debug(f"Unstructured Payload: {json.dumps(payload, indent=2)}")

    response = client.predict(endpoint=endpoint_name, inputs=payload)
    predictions = response.get("predictions", [])
    if not predictions:
        raise ValueError("No predictions returned.")
    
    pred = predictions[0]
    return next(iter(pred.values())) if isinstance(pred, dict) else str(pred)

def _query_structured(endpoint_name: str, args: Dict) -> Dict:
    client = get_deploy_client("databricks")
    payload = {
        "inputs": [{
            "question": args["question"],
            "generate_insights": args["generate_insights"]
        }]
    }
    logger.debug(f"Structured Payload: {json.dumps(payload, indent=2)}")

    response = client.predict(endpoint=endpoint_name, inputs=payload)

    prediction_obj = response.get("predictions", [{}])[0]
    json_str = prediction_obj.get("0")
    if json_str:
        return json.loads(json_str)

    raise ValueError(f"Unexpected response format: {response}")
