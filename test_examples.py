import math
import asyncio
import contextlib
import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import priceranger_mcp_tool_eval as sweep
import priceranger_mcp_eval as grader
import priceranger_mcp_quickstart as quickstart


class ContractTests(unittest.TestCase):
    def test_null_evidence_is_not_a_pass(self):
        for case in sweep.PLAN.values():
            status, issues = sweep._evidence_status({key: None for key in case['evidence']}, case['evidence'])
            self.assertEqual(status, 'THIN')
            self.assertTrue(issues)

    def test_asset_scope_is_not_guessed(self):
        with self.assertRaises(SystemExit):
            sweep._choose_asset({'allowed_to_you': []}, None)
        with self.assertRaises(SystemExit):
            sweep._choose_asset({'allowed_to_you': ['ETH']}, 'BTC')
        self.assertEqual(sweep._choose_asset({'allowed_to_you': ['ETH']}, None), 'ETH')

    def test_invalid_centers_do_not_produce_a_probe(self):
        for center in (None, False, 0, -1, math.nan, math.inf, '100'):
            self.assertIsNone(sweep._probe_price({'forecast_center': center}))
        self.assertAlmostEqual(sweep._probe_price({'forecast_center': 0.001}), 0.00099)

    def test_explicit_unavailability_and_collecting_are_not_success_claims(self):
        self.assertEqual(sweep._evidence_status({'available': False}, ('range_edge',)), ('UNAVAILABLE', []))
        payload = {'band_coverage': {'state': 'calibrating'}, 'band_baseline_skill_pct': None,
                   'center_beats_baseline': None}
        self.assertEqual(sweep._evidence_status(payload, tuple(payload)), ('COLLECTING', []))

    def test_bad_types_and_measured_coverage_fail(self):
        self.assertEqual(sweep._evidence_status({'allowed_to_you': 'ETH'}, ('allowed_to_you',))[0], 'THIN')
        payload = {'band_coverage': {'state': 'measured', 'band_coverage_pct': 101, 'coverage_target_pct': 90}}
        self.assertEqual(sweep._evidence_status(payload, tuple(payload))[0], 'THIN')

    def test_stale_evidence_is_labeled(self):
        payload = {'freshness': {'state': 'stale'}}
        self.assertEqual(sweep._evidence_status(payload, ('freshness',)), ('STALE', []))

    def test_grader_scope_filters_routing_rows(self):
        rows = [{'asset': 'BTC'}, {'symbol': 'ETH'}, 'SPY', {'asset': 'UNKNOWN'}]
        self.assertEqual(grader._scoped_routing_rows(rows, ['ETH']), [{'symbol': 'ETH'}])
        self.assertEqual(grader._scoped_routing_rows(rows, []), [])

    def test_grader_refuses_unknown_or_private_tools(self):
        grader._public_catalog(set(sweep.PLAN))
        with self.assertRaises(SystemExit):
            grader._public_catalog(set(sweep.PLAN) | {'trade_limit'})

    def test_grader_does_not_assert_historical_winners(self):
        self.assertNotIn('asset="BTC"', grader.QUESTION)
        self.assertNotIn('THE PREMIUM HOOK', grader.QUESTION)
        self.assertNotIn('the first signal to clear', grader.QUESTION)
        self.assertIn('INSUFFICIENT EVIDENCE', grader.QUESTION)

    def test_quickstart_uses_canonical_token_and_permitted_asset(self):
        calls = []
        class Client:
            def __init__(self, transport):
                self.transport = transport
            async def __aenter__(self):
                return self
            async def __aexit__(self, *arguments):
                return False
            async def call_tool(self, name, arguments):
                calls.append((name, arguments))
                data = {'whoami': {'endpoint_profile': 'analytics'},
                        'list_assets': {'allowed_to_you': ['ETH']},
                        'get_agent_brief': {'asset': 'ETH'}}[name]
                return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(data))])
        transports = []
        def transport(**arguments):
            transports.append(arguments)
            return arguments
        output = io.StringIO()
        with patch.dict(os.environ, {'PRICERANGER_MCP_TOKEN': 'canonical-test-value', 'PRICERANGER_TOKEN': 'legacy-test-value'}), \
                patch.object(quickstart, 'Client', Client), patch.object(quickstart, 'StreamableHttpTransport', transport), \
                contextlib.redirect_stdout(output):
            asyncio.run(quickstart.main())
        self.assertEqual(transports[0]['headers']['Authorization'], 'Bearer canonical-test-value')
        self.assertEqual(calls[-1], ('get_agent_brief', {'asset': 'ETH', 'compact': True}))
        self.assertNotIn('canonical-test-value', output.getvalue())

    def test_non_object_payloads_are_not_treated_as_evidence(self):
        value = sweep._payload(SimpleNamespace(structuredContent={'result': [1]}, content=[]))
        self.assertEqual(sweep._evidence_status(value, ('range_edge',))[0], 'THIN')
        value = sweep._payload(SimpleNamespace(content=[SimpleNamespace(text='[]')]))
        self.assertEqual(sweep._evidence_status(value, ('freshness',))[0], 'THIN')

    def test_range_edge_requires_accounting_and_reconciliation_state(self):
        payload = {'schema_version': 'range_edge.v2', 'freshness': {'state': 'fresh'},
                   'range_edge': {'published': True}}
        self.assertEqual(sweep._evidence_status(payload, tuple(payload))[0], 'THIN')
        payload['range_edge'].update(accounting='live_policy_stop_first', execution={'validated': False},
                                     stale=False, netcap_pct=-1.2, sample_windows=10)
        self.assertEqual(sweep._evidence_status(payload, tuple(payload)), ('OK', []))

    def test_sweep_skips_missing_probe_and_never_calls_unknown_tool(self):
        calls = []
        class Session:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *arguments):
                return False
            async def list_tools(self):
                return SimpleNamespace(tools=[SimpleNamespace(name=name) for name in list(sweep.PLAN) + ['trade_limit']])
            async def call_tool(self, name, arguments):
                calls.append((name, arguments))
                if name == 'whoami':
                    data = {'operator': 'synthetic', 'allowed_assets': ['ETH'], 'endpoint_profile': 'analytics', 'token_role': 'beta'}
                elif name == 'list_assets':
                    data = {'pool': ['BTC', 'ETH'], 'allowed_to_you': ['ETH'], 'access_note': 'Only ETH'}
                else:
                    data = {'available': False, 'asset': 'ETH'}
                return SimpleNamespace(structuredContent=data, content=[], isError=False)
        class Client:
            def __init__(self, *arguments, **keywords):
                pass
            def session(self, *arguments):
                return Session()
        with patch.object(sweep, 'MultiServerMCPClient', Client):
            result = asyncio.run(sweep.sweep(None))
        self.assertEqual(result['asset'], 'ETH')
        self.assertIn('trade_limit', result['leaked_write_tools'])
        self.assertFalse(any(name in {'trade_limit', 'get_touch_probability'} for name, arguments in calls))
        probe = next(row for row in result['rows'] if row['tool'] == 'get_touch_probability')
        self.assertEqual(probe['status'], 'SKIPPED')
        self.assertIn('no price was invented', probe['reason'])

    def test_report_handles_non_failure_states_and_skip_reasons(self):
        rows = [{'tool': state, 'status': state, 'ms': 0, 'bytes': 0, 'missing': [],
                 'error': None, 'value': None, 'reason': 'No input was guessed'}
                for state in ('SKIPPED', 'COLLECTING', 'STALE', 'UNAVAILABLE')]
        result = {'url': 'https://example.invalid/mcp', 'operator': 'synthetic', 'asset': 'ETH',
                  'advertised': 13, 'rows': rows, 'not_advertised': [], 'leaked_write_tools': {}}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(sweep.report(result), 0)
        self.assertIn('No input was guessed', output.getvalue())
        self.assertIn('contract checks passed', output.getvalue())


if __name__ == '__main__':
    unittest.main()
