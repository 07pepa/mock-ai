import json
import os
import subprocess

import click
import pydantic
import orjson

from mockai.models.json_file.models import PreDeterminedResponses

dir_path = os.path.dirname(os.path.realpath(__file__))


@click.group()
def cli():
    pass


@cli.command()
@click.argument("responses", type=click.File("rb"), required=False)
@click.option("--embedding-size", "-E", default=1536)
@click.option("--host", "-h", default="127.0.0.1")
@click.option("--port", "-p", default=8100)
def server(responses, embedding_size, host, port):
    if responses:
        print(f"Reading pre-determined responses from {responses.name}.")

        try:
            responses_data = orjson.loads(responses.read())
        except json.JSONDecodeError:
            raise click.BadParameter("Error reading JSON file: Is it valid JSON?")

        if responses_data == {}:
            raise click.BadParameter(f"Error reading JSON file: file is probably empty '{responses.read().decode()}'")

        try:
            PreDeterminedResponses.model_validate(responses_data)
        except pydantic.ValidationError as e:
            error = e.errors()[0]
            raise click.BadParameter(
                f"Error validating responses. Make sure they follow the proper structure.\n"
                f"Problematic input: {error['input']}\n"
                f"Fix: {error['msg']}\n"
                f"parsed json data was: {responses_data}"
            )
        os.environ["MOCKAI_RESPONSES"] = responses.name

    os.environ["MOCKAI_EMBEDDING_SIZE"] = str(embedding_size)

    print(f"Starting MockAI server ...")
    subprocess.run(
        [
            "uvicorn",
            "--app-dir",
            f"{dir_path}",
            "main:app",
            "--host",
            host,
            "--port",
            str(port),
            "--log-config",
            f"{dir_path}/logging_conf.yaml",
        ]
    )
