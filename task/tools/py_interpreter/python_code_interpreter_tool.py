import base64
import json
from typing import Any, Optional

from aidial_client import Dial
from aidial_sdk.chat_completion import Message, Attachment
from pydantic import StrictStr, AnyUrl

from task.tools.base import BaseTool
from task.tools.py_interpreter._response import _ExecutionResult
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool_model import MCPToolModel
from task.tools.models import ToolCallParams


class PythonCodeInterpreterTool(BaseTool):
    """
    Uses https://github.com/khshanovskyi/mcp-python-code-interpreter PyInterpreter MCP Server.

    ⚠️ Pay attention that this tool will wrap all the work with PyInterpreter MCP Server.
    """

    def __init__(
            self,
            mcp_client: MCPClient,
            mcp_tool_models: list[MCPToolModel],
            tool_name: str,
            dial_endpoint: str,
    ):
        """
        :param tool_name: it must be actual name of tool that executes code. It is 'execute_code'.
            https://github.com/khshanovskyi/mcp-python-code-interpreter/blob/main/interpreter/server.py#L303
        """
        #TODO:
        # 1. Set dial_endpoint
        self.dial_endpoint = dial_endpoint
        # 2. Set mcp_client
        self.mcp_client = mcp_client
        # 3. Set _code_execute_tool: Optional[MCPToolModel] as None at start, then iterate through `mcp_tool_models` and
        #    if any of tool model has the same same as `tool_name` then set _code_execute_tool as tool model
        for tool_model in mcp_tool_models:
            if tool_model.name == tool_name:
                self._code_execute_tool = tool_model
                break
        if self._code_execute_tool is None:
            raise ValueError(f"Tool model with name {tool_name} not found in {mcp_tool_models}")
        self.mcp_tool_model = self._code_execute_tool

        # 4. If `_code_execute_tool` is null then raise error (We cannot set up PythonCodeInterpreterTool without tool that executes code)

    @classmethod
    async def create(
            cls,
            mcp_url: str,
            tool_name: str,
            dial_endpoint: str,
    ) -> 'PythonCodeInterpreterTool':
        """Async factory method to create PythonCodeInterpreterTool"""
        #TODO:
        # 1. Create MCPClient
        mcp_client = await MCPClient.create(mcp_server_url=mcp_url)
        # 2. Get tools
        mcp_tool_models = await mcp_client.get_tools()
        # 3. Create PythonCodeInterpreterTool instance and return it
        return cls(
            mcp_client=mcp_client,
            mcp_tool_models=mcp_tool_models,
            tool_name=tool_name,
            dial_endpoint=dial_endpoint
        )

    @property
    def show_in_stage(self) -> bool:
        # TODO: set as False since we will have custom variant of representation in Stage
        return False

    @property
    def name(self) -> str:
        # TODO: provide `_code_execute_tool` name
        return self._code_execute_tool.name

    @property
    def description(self) -> str:
        # TODO: provide `_code_execute_tool` description
        return self._code_execute_tool.description

    @property
    def parameters(self) -> dict[str, Any]:
        # TODO: provide `_code_execute_tool` parameters
        return self._code_execute_tool.parameters

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        #TODO:
        # 1. Load arguments with `json`
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        # 2. Get `code` from arguments
        code = arguments.get("code")
        # 3. Get `session_id` from arguments (it is optional parameter, use get method)
        session_id = arguments.get("session_id")
        # 4. Get stage from `tool_call_params`
        stage = tool_call_params.stage
        # 5. Append content to stage: "## Request arguments: \n"
        stage.append_content("## Request arguments: \n")
        # 6. Append content to stage: `"```python\n\r{code}\n\r```\n\r"` it will show code in stage as python markdown
        stage.append_content(f"```python\n\r{code}\n\r```\n\r")
        # 7. Append session to stage:
        #       - if `session_id` is present and not 0 then append to stage `f"**session_id**: {session_id}\n\r"`
        #       - otherwise append "New session will be created\n\r"
        if session_id and session_id != "0":
            stage.append_content(f"**session_id**: {session_id}\n\r")
        else:
            stage.append_content("New session will be created\n\r")
        # 8. Make tool call
        result = await self.mcp_client.call_tool(self.mcp_tool_model.name, arguments)
        execution_result = _ExecutionResult.model_validate_json(result)
        # 9. Load retrieved response as json (️⚠️ here can be potential issues if you didn't properly implemented
        #    MCPClient tool call, it must return string)
        result = json.loads(result)
        # 10. Validate result with _ExecutionResult (it is full copy of https://github.com/khshanovskyi/mcp-python-code-interpreter/blob/main/interpreter/models.py)
        # 11. If execution_result contains files we need to pool files from PyInterpreter and upload them to DIAL bucked:
        #       - Create Dial client
        #       - Get with client `my_appdata_home` path as `files_home`
        #       - Iterated through files and:
        #           - get file name and mime_type and assign to appropriate variables
        #           - get resource with mcp client by URL from file (https://github.com/khshanovskyi/mcp-python-code-interpreter/blob/main/interpreter/server.py#L429)
        #           - according to MCP binary resources must be encoded with base64 https://modelcontextprotocol.io/specification/2025-06-18/server/resources#binary-content
        #             Check if mime_type starts with `text/` or some of 'application/json', 'application/xml', is yes
        #             then encode resource with 'utf-8' format (text will be present as bytes to upload to DIAL bucket).
        #             Otherwise (binary file) decode it with `b64decode`
        #           - Prepare URL to upload downloaded file: f"files/{(files_home / file_name).as_posix()}"
        #           - Upload file with DIAL client
        #           - Prepare Attachment with url, type (mime_type), and title (file_name)
        #           - Add attachment to stage and also add this attachment to choice (it will be chown in both stage and choice)
        #       - Add to execution_result json addition
        dial_client = Dial(base_url=self.dial_endpoint, api_key=tool_call_params.api_key)
        files_home = dial_client.my_appdata_home
        for file in execution_result.files:
            file_name = file.name
            mime_type = file.mime_type
            resource = await self.mcp_client.get_resource(file.uri)

            if mime_type.startswith("text/") or mime_type in ["application/json", "application/xml"]:
                resource = resource.decode("utf-8")
            else:   
                resource = base64.b64decode(resource)

            url = f"files/{(files_home / file_name).as_posix()}"
            await dial_client.upload_file(url, resource, mime_type)
            attachment = Attachment(url=url, type=mime_type, title=file_name)

            stage.append_attachment(attachment)
            tool_call_params.choice.append_attachment(attachment)

        if execution_result.output:
            execution_result.output = execution_result.output[:1000]
        stage.append_content(f"```json\n\r{execution_result.model_dump_json(indent=2)}\n\r```\n\r")
        return execution_result.model_dump_json()

        # 12. Check if execution_result output present and if yes iterate through all output results and cut it length
        #     to 1000 chars, it is needed to avoid high costs and context window overload
        # 13. Append to stage response f"```json\n\r{execution_result.model_dump_json(indent=2)}\n\r```\n\r"
        # 14. Return execution result as string (model_dump_json method)
