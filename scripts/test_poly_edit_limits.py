"""Offline vertex-limit regressions; never connects to 3ds Max.

Run each registration profile in a separate process:
    python scripts/test_poly_edit_limits.py --profile core
    python scripts/test_poly_edit_limits.py --profile full
    python scripts/test_poly_edit_limits.py --profile progressive
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PROFILE = "core"


class VertexLimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        environment = patch.dict(os.environ, {"MCP_TOOL_PROFILE": PROFILE})
        environment.start()
        cls.addClassCleanup(environment.stop)
        from maxmcp.max_client import MaxClient

        # Patch before server/tool imports, so an accidental bridge call fails closed.
        bridge = patch.object(
            MaxClient, "send_command", side_effect=AssertionError("Unexpected bridge call")
        )
        cls.bridge = bridge.start()
        cls.addClassCleanup(bridge.stop)
        from maxmcp.server import mcp

        cls.mcp = mcp

    def setUp(self):
        self.bridge.reset_mock()
        self.bridge.side_effect = AssertionError("Unexpected bridge call")

    def call(self, tool, arguments):
        response = asyncio.run(self.mcp.call_tool(tool, arguments))
        if isinstance(response, tuple):
            return response[1]
        if isinstance(response, dict):
            return response
        return json.loads(response[0].text)

    def edit(self, action="get", **arguments):
        args = {"action": action, "name": "Offline mesh", **arguments}
        if PROFILE == "progressive":
            return self.call("call_tool", {"name": "edit_vertices", "arguments": args})
        return self.call("edit_vertices", args)

    def prepare(self, action, count):
        self.bridge.reset_mock()
        self.bridge.side_effect = None
        if action == "get":
            reply, args = "META|10000|0|0|0\n", {}
        elif action == "move":
            reply, args = "OK|1|0|false", {"offset": [1, 2, 3]}
        elif action == "set":
            reply, args = f"OK|{count}|false", {"positions": [[0, 0, 0]] * count}
        else:
            reply, args = "OK|1|0|spline|false", {"target": "Offline spline"}
        self.bridge.return_value = {"result": reply}
        return args

    def sent_indices(self):
        self.bridge.assert_called_once()
        script = self.bridge.call_args.args[0]
        match = re.search(r"local (?:rawIdxs|idxs) = #\(([^)]*)\)", script)
        self.assertIsNotNone(match, "Expected explicit indices in the bridge command")
        return [int(value.strip()) for value in match[1].split(",")]

    def test_published_limits(self):
        tools = asyncio.run(self.mcp.list_tools())
        if PROFILE == "progressive":
            self.assertNotIn("edit_vertices", [tool.name for tool in tools])
            described = self.call("describe_toolset", {"toolset": "modeling"})
            self.assertTrue(described["ok"], described)
            tool = next(t for t in described["result"]["tools"] if t["name"] == "edit_vertices")
            description, schema = tool["description"], tool["input_schema"]
        else:
            tool = next(t for t in tools if t.name == "edit_vertices")
            description, schema = tool.description, tool.inputSchema
        for number in ("5000", "2000", "500"):
            self.assertIn(number, description.replace(",", ""))
        props = schema["properties"]
        self.assertIn("5000", props["indices"]["description"].replace(",", ""))
        self.assertIn("2000", props["max_verts"]["description"].replace(",", ""))
        self.assertEqual(props["max_verts"]["default"], 500)
        self.bridge.assert_not_called()

    def test_static_catalog_preserves_types_and_limit_metadata(self):
        from scripts.gen_tool_catalog import collect_tools

        generated = next(tool for tool in collect_tools() if tool["name"] == "edit_vertices")
        saved_catalog = json.loads((ROOT / "tool_playground" / "catalog.json").read_text(encoding="utf-8"))
        saved = next(tool for tool in saved_catalog["tools"] if tool["name"] == "edit_vertices")
        self.assertEqual(saved["schema"], generated["schema"])
        self.assertEqual(saved["description"], generated["description"])
        props = generated["schema"]["properties"]
        self.assertEqual(props["indices"]["type"], "array")
        self.assertEqual(props["indices"]["items"], {"type": "integer"})
        self.assertIn("5000", props["indices"]["description"].replace(",", ""))
        self.assertEqual(props["max_verts"]["type"], "integer")
        self.assertIn("2000", props["max_verts"]["description"].replace(",", ""))
        self.bridge.assert_not_called()

    def test_index_limit_boundaries_for_every_action(self):
        for action in ("get", "move", "set", "conform"):
            for count in (4999, 5000, 5001):
                with self.subTest(action=action, count=count):
                    args = self.prepare(action, count)
                    indices = list(range(1, count + 1))
                    result = self.edit(action, indices=indices, **args)
                    if count <= 5000:
                        self.assertTrue(result["ok"], result)
                        self.assertEqual(self.sent_indices(), indices)
                    else:
                        self.assertFalse(result["ok"], result)
                        self.bridge.assert_not_called()
                        self.assertEqual(result["error"]["code"], "BAD_PARAM")
                        self.assertEqual(result["error"]["details"],
                                         {"max_indices": 5000, "received_indices": 5001})
                        message = result["error"]["message"].lower().replace(",", "")
                        self.assertIn("5001", message)
                        self.assertIn("5000", message)
                        self.assertRegex(message, r"batch|chunk|split")

    def test_batch_guidance_preserves_action_semantics(self):
        for action, extras, required in (
            ("get", {}, (r"2000", r"max_verts")),
            ("set", {}, (r"position", r"pair|parallel", r"order")),
            ("move", {}, (r"non.overlapping", r"falloff\s*=\s*0")),
            ("move", {"falloff": 2.0}, (r"do not (split|batch)", r"overlap", r"change.*result")),
            ("conform", {}, (r"target", r"unchanged")),
        ):
            with self.subTest(action=action, extras=extras):
                args = self.prepare(action, 5001)
                result = self.edit(action, indices=list(range(1, 5002)), **args, **extras)
                self.assertFalse(result["ok"], result)
                self.bridge.assert_not_called()
                message = result["error"]["message"].lower()
                for pattern in required:
                    self.assertRegex(message, pattern)

    def test_nonpositive_values_filtered_before_counting(self):
        positives = list(range(1, 5001))
        indices = [0, -4] * 50 + positives
        for action in ("get", "move", "set", "conform"):
            with self.subTest(action=action):
                args = self.prepare(action, len(positives))
                result = self.edit(action, indices=indices, **args)
                self.assertTrue(result["ok"], result)
                self.assertEqual(self.sent_indices(), positives)

    def test_duplicate_entries_count_toward_limit(self):
        for count in (5000, 5001):
            with self.subTest(count=count):
                self.prepare("get", count)
                result = self.edit(indices=[7] * count)
                self.assertEqual(result["ok"], count == 5000, result)
                if count == 5000:
                    self.assertEqual(self.sent_indices(), [7] * count)
                else:
                    self.bridge.assert_not_called()

    def test_large_vertex_ids_and_existing_input_coercion(self):
        for indices, expected in (
            ([5001, 9000], [5001, 9000]),
            ("[5001, 9000]", [5001, 9000]),
            ("5001,9000", [5001, 9000]),
            (9000, [9000]),
        ):
            with self.subTest(indices=indices):
                self.prepare("get", len(expected))
                result = self.edit(indices=indices)
                self.assertTrue(result["ok"], result)
                self.assertEqual(self.sent_indices(), expected)

    def test_get_default_and_clamping(self):
        for requested, expected in ((None, 500), (-10, 1), (0, 1), (1, 1), (1999, 1999),
                                    (2000, 2000), (2001, 2000), (99999, 2000)):
            with self.subTest(max_verts=requested):
                self.prepare("get", 0)
                args = {} if requested is None else {"max_verts": requested}
                result = self.edit(**args)
                self.assertTrue(result["ok"], result)
                self.bridge.assert_called_once()
                script = self.bridge.call_args.args[0]
                match = re.search(r"if emitted < (\d+) do", script)
                self.assertIsNotNone(match, "Expected a bounded bridge response")
                self.assertEqual(int(match[1]), expected)

    def test_truncated_reads_explain_limit_and_paging(self):
        for returned in (500, 2000):
            with self.subTest(returned=returned):
                self.prepare("get", returned)
                vertices = "".join(f"V|{i}|[1,2,3]\n" for i in range(1, returned + 1))
                self.bridge.return_value = {"result": f"{vertices}META|10000|3000|{returned}|0\n"}
                result = self.edit(max_verts=returned)
                self.assertTrue(result["ok"], result)
                self.assertEqual(len(result["result"]["verts"]), returned)
                self.assertEqual(result["result"]["verts"][-1], {"i": returned, "pos": [1.0, 2.0, 3.0]})
                advice = result["result"]["truncated"].lower().replace(",", "")
                self.assertIn("2000", advice)
                self.assertRegex(advice, r"indices|index")
                self.assertRegex(advice, r"pag|batch|chunk")
        self.prepare("get", 0)
        self.assertNotIn("truncated", self.edit()["result"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("core", "full", "progressive"), default="core")
    options, test_args = parser.parse_known_args()
    PROFILE = options.profile
    unittest.main(argv=[sys.argv[0], *test_args])
