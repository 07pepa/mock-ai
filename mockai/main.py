import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiofiles

from mockai.anthropic.router import anthropic_router
from mockai.dependencies import ResponseFile
from mockai.models import PreDeterminedResponse
from mockai.openai.router import openai_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    mockai_embedding_size = os.getenv("MOCKAI_EMBEDDING_SIZE", "1536")

    app.state.embedding_size = int(mockai_embedding_size)

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(openai_router)
app.include_router(anthropic_router)


@app.get("/api/responses/create")
async def create_response(responses: ResponseFile):
    if responses is None:
        raise HTTPException(400, "No response file to update.")

    responses.append(PreDeterminedResponse(type="text", input="", output=""))

    if file := os.getenv("MOCKAI_RESPONSES"):
        async with aiofiles.open(file, "w") as f:
            await f.write(responses.model_dump_json(indent=2))
        return responses
    else:
        raise ValueError("No response file set.")


@app.get("/api/responses/read")
async def get_responses(file: ResponseFile):
    return file


class ResponseUpdate(BaseModel):
    number: int
    new_response: PreDeterminedResponse


@app.post("/api/responses/update")
async def update_response(data: ResponseUpdate, responses: ResponseFile):
    if responses is None:
        raise HTTPException(400, "No response file to update.")

    responses[data.number - 1] = data.new_response

    if file := os.getenv("MOCKAI_RESPONSES"):
        async with aiofiles.open(file, "w") as f:
            await f.write(responses.model_dump_json(indent=2))
        return responses
    else:
        raise ValueError("No response file set.")


@app.delete("/api/responses/delete")
async def delete_response(number: int, responses: ResponseFile):
    if responses is None:
        raise HTTPException(400, "No response file to update.")

    responses.pop(number - 1)

    if file := os.getenv("MOCKAI_RESPONSES"):
        async with aiofiles.open(file, "w") as f:
            await f.write(responses.model_dump_json(indent=2))
        return responses
    else:
        raise ValueError("No response file set.")
