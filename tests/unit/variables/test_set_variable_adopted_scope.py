"""Set Variable adopts an existing variable at its own scope, and every path agrees which one.

``SetVariable`` decides "does this variable already exist" at the node's declared scope
(hierarchical by default) in three places: the eager create when ``variable_name`` is set,
the type sync when something connects to ``value``, and the write in ``aprocess``. These
tests pin that the three agree, both for a global the node adopts from the top level and
for a flow variable adopted from inside a loop body.
"""

from collections.abc import Generator

import pytest
from griptape_nodes.retained_mode.engine import Engine
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest, CreateConnectionResultSuccess
from griptape_nodes.retained_mode.events.execution_events import StartFlowRequest, StartFlowResultSuccess
from griptape_nodes.retained_mode.events.flow_events import (
    CreateFlowRequest,
    CreateFlowResultSuccess,
    DeleteFlowRequest,
)
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest, CreateNodeResultSuccess
from griptape_nodes.retained_mode.events.parameter_events import SetParameterValueRequest
from griptape_nodes.retained_mode.events.variable_events import (
    CreateVariableRequest,
    CreateVariableResultSuccess,
    DeleteVariableRequest,
    GetVariableTypeRequest,
    GetVariableTypeResultSuccess,
    GetVariableValueRequest,
    GetVariableValueResultSuccess,
    HasVariableRequest,
    HasVariableResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.variable_types import VariableScope

FLOW_NAME = "canvas"
VARIABLE_NAME = "shared_name"


@pytest.fixture
def flow(griptape_nodes: Engine) -> Generator[str, None, None]:  # noqa: ARG001
    """Create a fresh top-level flow for each test and drop any global the test created."""
    context_manager = GriptapeNodes.ContextManager()
    context_manager.push_workflow(workflow_name="test_set_variable_adopted_scope_workflow")
    try:
        result = GriptapeNodes.handle_request(CreateFlowRequest(parent_flow_name=None, flow_name=FLOW_NAME))
        assert isinstance(result, CreateFlowResultSuccess)
        yield FLOW_NAME
        GriptapeNodes.handle_request(DeleteFlowRequest(flow_name=FLOW_NAME))
        GriptapeNodes.handle_request(DeleteVariableRequest(name=VARIABLE_NAME, lookup_scope=VariableScope.GLOBAL_ONLY))
    finally:
        context_manager.pop_workflow()


def _create_node(
    node_type: str, node_name: str, *, parent_flow: str | None = None, parent_group: str | None = None
) -> None:
    result = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type=node_type,
            node_name=node_name,
            override_parent_flow_name=parent_flow,
            parent_group_name=parent_group,
            metadata={"position": {"x": 0, "y": 0}},
        )
    )
    assert isinstance(result, CreateNodeResultSuccess), result.result_details


def _set_param(node_name: str, parameter_name: str, value: object) -> None:
    GriptapeNodes.handle_request(
        SetParameterValueRequest(node_name=node_name, parameter_name=parameter_name, value=value)
    )


def _flow_has_variable(flow: str) -> bool:
    result = GriptapeNodes.handle_request(
        HasVariableRequest(name=VARIABLE_NAME, lookup_scope=VariableScope.CURRENT_FLOW_ONLY, starting_flow=flow)
    )
    assert isinstance(result, HasVariableResultSuccess)
    return result.exists


class TestSetVariableAdoptsGlobal:
    """A top-level Set Variable whose name matches a global writes that global, not a local copy."""

    @pytest.mark.asyncio
    async def test_write_reaches_the_global(self, flow: str) -> None:
        create_result = GriptapeNodes.handle_request(
            CreateVariableRequest(name=VARIABLE_NAME, type="str", is_global=True, value="global-seed")
        )
        assert isinstance(create_result, CreateVariableResultSuccess), create_result.result_details

        _create_node("SetVariable", "Set", parent_flow=flow)
        _set_param("Set", "variable_name", VARIABLE_NAME)
        _set_param("Set", "value", "from-node")

        # Picking the global's name must not mint a flow-local shadow.
        assert not _flow_has_variable(flow)

        start_result = await GriptapeNodes.ahandle_request(StartFlowRequest(flow_name=flow))
        assert isinstance(start_result, StartFlowResultSuccess), start_result.result_details

        value_result = GriptapeNodes.handle_request(
            GetVariableValueRequest(name=VARIABLE_NAME, lookup_scope=VariableScope.GLOBAL_ONLY)
        )
        assert isinstance(value_result, GetVariableValueResultSuccess)
        assert value_result.value == "from-node"
        assert not _flow_has_variable(flow)


class TestSetVariableTypeSyncInsideLoop:
    """Connecting a typed source to ``value`` updates the adopted outer variable's type."""

    def test_connection_updates_the_outer_variables_type(self, flow: str) -> None:
        create_result = GriptapeNodes.handle_request(
            CreateVariableRequest(name=VARIABLE_NAME, type="str", is_global=False, value="seed", owning_flow=flow)
        )
        assert isinstance(create_result, CreateVariableResultSuccess), create_result.result_details

        _create_node("ForLoopGroupNode", "Loop", parent_flow=flow)
        _create_node("SetVariable", "Set", parent_group="Loop")
        _set_param("Set", "variable_name", VARIABLE_NAME)
        _create_node("IntegerInput", "Int", parent_group="Loop")

        connect_result = GriptapeNodes.handle_request(
            CreateConnectionRequest(
                source_node_name="Int",
                source_parameter_name="integer",
                target_node_name="Set",
                target_parameter_name="value",
            )
        )
        assert isinstance(connect_result, CreateConnectionResultSuccess), connect_result.result_details

        type_result = GriptapeNodes.handle_request(
            GetVariableTypeRequest(name=VARIABLE_NAME, lookup_scope=VariableScope.CURRENT_FLOW_ONLY, starting_flow=flow)
        )
        assert isinstance(type_result, GetVariableTypeResultSuccess)
        assert type_result.type == "int"
