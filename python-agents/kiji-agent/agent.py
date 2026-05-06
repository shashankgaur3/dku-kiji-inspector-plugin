import dataiku
import json
import requests
import logging
from dataiku.llm.python import BaseLLM
from kiji_agent.utils import create_payload, parse_response

logger = logging.getLogger(__name__)

# These will be set from agent configuration parameters
FEATURES_DATASET = None
POSTGRES_CONNECTION = None
LLM_ENDPOINT = None
LLM_MODEL_NAME = None


def _extract_explanations(response_data: dict):
    """
    Extract feature explanations from the LLM response.

    Args:
        response_data: The response from the LLM API

    Returns:
        dict: Dictionary mapping layer IDs to lists of feature explanations
              Each explanation contains: feature_id, description, activation
    """
    choices = response_data.get('choices', [])
    if not choices:
        return {}
    explanations = choices[0].get('explanations', {})
    return explanations


def _get_or_create_physical_table_name():
    try:
        dku_client = dataiku.api_client()
        project = dku_client.get_default_project()
        builder = project.new_managed_dataset(FEATURES_DATASET)

        ds_exist = builder.already_exists()
        if not ds_exist:
            logger.info(f"DS {FEATURES_DATASET} doesn't exist. Creating one.")

            builder.with_store_into(POSTGRES_CONNECTION)
            dataset = builder.create()
            ds_settings = dataset.get_settings()
            ds_settings.add_raw_schema_column({"name": "timestamp_ms", "type": "bigint"})
            ds_settings.add_raw_schema_column({"name": "messages", "type": "string"})
            ds_settings.add_raw_schema_column({"name": "settings", "type": "string"})
            ds_settings.add_raw_schema_column({"name": "response", "type": "string"})
            ds_settings.add_raw_schema_column({"name": "layer_id", "type": "string"})
            ds_settings.add_raw_schema_column({"name": "feature_id", "type": "int"})
            ds_settings.add_raw_schema_column({"name": "description", "type": "string"})
            ds_settings.add_raw_schema_column({"name": "activation", "type": "double"})
            ds_settings.save()

            dss_dataset_obj = dataiku.Dataset(FEATURES_DATASET)
            with dss_dataset_obj.get_writer() as writer:
                pass

            logger.info(f"DS {FEATURES_DATASET} created successfully.")
            physical_table_name = dss_dataset_obj.get_location_info()["info"]["table"]

            logger.info(f"Physical table {physical_table_name} created successfully.")
            return physical_table_name

        dss_dataset_obj = dataiku.Dataset(FEATURES_DATASET)
        location = dss_dataset_obj.get_location_info()
        info = location.get("info", {})
        table = info.get("table")
        if not table:
            logger.error("Dataset exists but has no physical table")
            return None
        return table
    except Exception:
        logger.exception("Failed to get or create underlying table name")
        return None

def _write_features_rows(query: dict, settings: dict, response_data: dict, parsed_response: dict, trace=None):
    """
    Write feature explanations to the database.

    Each feature from each layer is written as a separate row.

    Args:
        query: The original query containing messages
        settings: LLM settings used
        response_data: Raw response from the LLM API
        parsed_response: Parsed response
        trace: Optional trace object for logging
    """
    try:
        from datetime import datetime, timezone
        # Get current timestamp in milliseconds since epoch (Unix timestamp)
        current_ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        # Extract explanations from the LLM response
        explanations = _extract_explanations(response_data)

        # Early return if no explanations available
        if not explanations:
            logger.warning("No explanations to write - skipping database write")
            return

        logger.info(f"Extracted explanations from {len(explanations)} layer(s)")

        # Add trace for feature explanations
        if trace:
            try:
                # Set trace span name
                trace.span["name"] = "FEATURE_EXPLANATION_EXTRACTION"

                # Add input - the LLM response
                trace.inputs["llm_response_text"] = parsed_response.get("text", "")
                trace.inputs["llm_response_tool_calls"] = json.dumps(parsed_response.get("toolCalls", []))
                trace.inputs["llm_finish_reason"] = parsed_response.get("finishReason", "")

                # Add output - the full explanations
                trace.outputs["feature_explanations"] = json.dumps(explanations)

                total_features = sum(len(features) for features in explanations.values())
                logger.info(f"Added trace for feature explanation extraction with {total_features} total features")
            except Exception as trace_error:
                logger.warning(f"Failed to add trace for feature explanations: {trace_error}")

        # Get table name
        table = _get_or_create_physical_table_name()
        if not table:
            logger.error("Failed to get table name - Dataset is not SQL-backed or missing connection/table info")
            raise ValueError("Dataset is not SQL-backed or missing connection/table info")

        logger.info(f"Writing to table: {table}")

        # Prepare rows - one row per feature
        rows = []
        for layer_id, features in explanations.items():
            for feature in features:
                row = {
                    "timestamp_ms": current_ts_ms,
                    "messages": json.dumps(query.get("messages", [])),
                    "settings": json.dumps(settings or {}),
                    "response": json.dumps(parsed_response or {}),
                    "layer_id": str(layer_id),
                    "feature_id": feature.get("feature_id"),
                    "description": feature.get("description", ""),
                    "activation": feature.get("activation", 0.0)
                }
                rows.append(row)

        if not rows:
            logger.warning("No feature rows to write")
            return

        logger.info(f"Preparing to write {len(rows)} feature row(s) with timestamp: {current_ts_ms}")

        # Use SQL INSERT to append rows to Postgres table
        def _sql_literal(value):
            if value is None:
                return "NULL"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return str(value)
            # Escape single quotes by doubling them
            return "'" + str(value).replace("'", "''") + "'"

        # Build multi-row INSERT statement
        columns = list(rows[0].keys())
        values_list = []
        for row in rows:
            values = [_sql_literal(row[c]) for c in columns]
            values_list.append(f"({', '.join(values)})")

        insert_stmt = f"""
        INSERT INTO "{table}" (
            {", ".join(columns)}
        ) VALUES
            {", ".join(values_list)}
        """

        logger.debug(f"Executing SQL INSERT for {len(rows)} rows: {insert_stmt[:200]}...")

        try:
            executor = dataiku.SQLExecutor2(dataset=FEATURES_DATASET)
            # Execute INSERT directly with post_queries COMMIT - this is the correct pattern
            executor.query_to_df(insert_stmt, post_queries=["COMMIT"])
            logger.info(f"Successfully appended {len(rows)} feature row(s) to {FEATURES_DATASET}")
        except Exception as sql_error:
            logger.error(f"SQL execution failed: {sql_error}", exc_info=True)
            raise

    except Exception as e:
        logger.error(f"Failed to write feature rows: {e}", exc_info=True)
        # Don't re-raise to avoid breaking the main flow
        # But log the full error for debugging


class MyLLM(BaseLLM):
    def __init__(self):
        """
        Initialize the custom LLM agent.
        Configuration will be set via the set_config method called by Dataiku framework.
        """
        pass

    def set_config(self, config, plugin_config):
        """
        Set configuration parameters for the agent.
        This method is called by Dataiku framework after instantiation.

        Args:
            config (dict): Configuration dictionary containing:
                - llm_endpoint: LLM API endpoint URL
                - llm_model_name: LLM model name/identifier
                - postgres_connection: Postgres connection name
                - activations_table: Dataset name for storing feature explanations
        """
        global FEATURES_DATASET, POSTGRES_CONNECTION, LLM_ENDPOINT, LLM_MODEL_NAME

        # Set global configuration from parameters
        LLM_ENDPOINT = config.get('llm_endpoint', 'http://127.0.0.1:8000/v1/chat/completions')
        LLM_MODEL_NAME = config.get('llm_model_name', 'google/gemma-4-E4B-it')
        POSTGRES_CONNECTION = config.get('postgres_connection', 'postgres-rds')
        FEATURES_DATASET = config.get('activations_table', 'features_master')

        logger.info(f"Initialized MyLLM with config:")
        logger.info(f"  LLM Endpoint: {LLM_ENDPOINT}")
        logger.info(f"  LLM Model: {LLM_MODEL_NAME}")
        logger.info(f"  Postgres Connection: {POSTGRES_CONNECTION}")
        logger.info(f"  Features Dataset: {FEATURES_DATASET}")

    def process(self, query, settings, trace):
        """
        Processes a query for a non-streaming response.

        Args:
            query (dict): The input query, typically containing a 'messages' list.
            settings (dict): A dictionary of settings for the API call (e.g., temperature).
            trace: A tracing object for logging.

        Returns:
            dict: The complete parsed response from the LLM.

        Raises:
            ConnectionError: If the API request fails.
        """
        logger.info(f"Processing non-streaming request")
        # This method handles non-streaming requests.
        # We explicitly set stream to False in the payload.
        settings['stream'] = False
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = create_payload(LLM_MODEL_NAME, query, settings)

        logger.debug(f"Sending request with query: {query}")
        logger.debug(f"Sending request with settings: {settings}")
        logger.debug(f"Using LLM endpoint: {LLM_ENDPOINT}")

        try:
            response = requests.post(
                LLM_ENDPOINT, headers=headers, json=payload, stream=False
            )
            response.raise_for_status()
            response_data = response.json()
            logger.info("Successfully received response from API #### ")
            logger.debug(f"Response keys: {response_data.keys()}")

            parsed_response = parse_response(response_data, 0, 0)

            # Check if response contains explanations before attempting to write
            if response_data.get('choices', [{}])[0].get('explanations'):
                # Attempt to write feature explanations - log any errors but don't fail the request
                try:
                    logger.info("Attempting to write feature explanations to database")
                    _write_features_rows(query, settings, response_data, parsed_response, trace)
                except Exception as write_error:
                    logger.error(f"Failed to write feature explanations (non-fatal): {write_error}", exc_info=True)
            else:
                logger.info("No explanations found in response - skipping feature processing")

            return parsed_response

        except requests.exceptions.RequestException as e:
            # Handle exceptions for network errors, non-200 responses, etc.
            logger.error(f"API request failed: {e}", exc_info=True)
            raise ConnectionError(f"API request failed: {e}")
