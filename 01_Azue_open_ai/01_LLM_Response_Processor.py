from fastapi.responses import JSONResponse


class LLMResponseProcessor:

    def is_meaningful_response(llm_output: str) -> bool:
        fallback_phrases = [
            "does not contain any data",
            "I don't have enough information",
            "I'm not sure",
            "cannot provide a response",
        ]
        return not any(phrase in llm_output.lower() for phrase in fallback_phrases)

    def process_llm_response(trace_id: str) -> JSONResponse:
        try:
            # Simulate LLM call
            llm_output = call_llm()  # LLM Call placeholder

            if not is_meaningful_response(llm_output):
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "success",
                        "message": "LLM did not generate meaningful output.",
                        "traceId": trace_id,
                        "data": {"response": llm_output},
                    },
                )
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": "LLM response generated successfully.",
                    "traceId": trace_id,
                    "data": {"response": llm_output},
                },
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Failed to generate response from LLM.",
                    "traceId": trace_id,
                    "data": None,
                },
            )
