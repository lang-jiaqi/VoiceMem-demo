from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from studio.core.utils.reply_modes.initialize import (
    FAST,
    MEDIUM,
    SLOW,
    QwenThinkingRouter,
    ThinkingDecision,
    parse_level,
)


class ThinkingRouterTests(unittest.TestCase):
    def test_missing_default_router_downloads_to_project_model_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "Qwen3-0.6B"
            router = QwenThinkingRouter.__new__(QwenThinkingRouter)
            router._local_model_dir = destination
            router._download_default = True
            router.model_name = str(destination)

            def download(**kwargs):
                self.assertEqual(kwargs["repo_id"], "Qwen/Qwen3-0.6B")
                self.assertEqual(kwargs["local_dir"], str(destination))
                self.assertTrue(kwargs["tqdm_class"])
                destination.mkdir(parents=True)
                (destination / "config.json").write_text("{}", encoding="utf-8")
                (destination / "tokenizer.json").write_text("{}", encoding="utf-8")
                (destination / "model.safetensors").write_bytes(b"test")

            with patch("huggingface_hub.snapshot_download", side_effect=download):
                selected = router._ensure_model_source()

        self.assertEqual(selected, str(destination))
        self.assertFalse(router._download_default)

    def test_configured_router_never_triggers_default_download(self):
        router = QwenThinkingRouter(model="/models/custom-router")
        with patch("huggingface_hub.snapshot_download") as download:
            self.assertEqual(router._ensure_model_source(), "/models/custom-router")
        download.assert_not_called()

    def test_parses_only_supported_labels(self):
        self.assertEqual(parse_level("fast").level, FAST)
        self.assertEqual(parse_level("The label is MEDIUM.").level, MEDIUM)
        self.assertEqual(parse_level("slow\n").level, SLOW)
        self.assertEqual(parse_level("即时").level, FAST)
        self.assertEqual(parse_level("记忆").level, MEDIUM)
        self.assertEqual(parse_level("深思").level, SLOW)

    def test_invalid_output_uses_explicit_fallback(self):
        self.assertEqual(parse_level("unknown", MEDIUM).level, MEDIUM)

    def test_levels_map_to_deepseek_effort(self):
        self.assertEqual(ThinkingDecision(FAST).reasoning_effort, "none")
        self.assertEqual(ThinkingDecision(MEDIUM).reasoning_effort, "none")
        self.assertEqual(ThinkingDecision(SLOW).reasoning_effort, "high")

    def test_levels_map_to_reply_modes(self):
        self.assertEqual(ThinkingDecision(FAST).reply_mode, "direct")
        self.assertEqual(ThinkingDecision(MEDIUM).reply_mode, "memory")
        self.assertEqual(ThinkingDecision(SLOW).reply_mode, "memory_cot")

    def test_only_depth_is_selected_by_the_model(self):
        router = QwenThinkingRouter(model='/unused')
        for index, (label, expected) in enumerate((('否', FAST), ('是', SLOW),
                                                 ('否。', FAST), ('不是深思', FAST),
                                                 ('记忆', FAST))):
            router._predict = lambda *_, answer=label: answer
            self.assertEqual(router.classify(f'合成问题{index}').level, expected)

    def test_memory_hint_cannot_change_depth_or_trigger_another_inference(self):
        from unittest.mock import Mock
        router = QwenThinkingRouter(model='/unused')
        router._predict = Mock(return_value='否')
        self.assertEqual(router.classify('我明天有什么安排？', True).level, FAST)
        self.assertEqual(router.classify('我明天有什么安排？', False).level, FAST)
        router._predict.assert_called_once()
        self.assertNotIn('预取', router._predict.call_args.args[2])

    def test_no_semantic_keyword_shortcuts_remain(self):
        from unittest.mock import Mock
        router = QwenThinkingRouter(model='/unused')
        router._predict = Mock(return_value='否')
        for text in ('你好', '我明天有什么安排？', '计算这个函数的积分', '嗯'):
            self.assertEqual(router.classify(text).level, FAST)
        self.assertEqual(router._predict.call_count, 4)

    def test_chinese_prompt_only_asks_for_reasoning_depth(self):
        router = QwenThinkingRouter(model='/unused')
        seen = {}
        def predict(system, examples, prompt):
            seen.update(system=system, examples=examples, prompt=prompt)
            return '否'
        router._predict = predict
        router.classify('介绍一下向量数据库')
        self.assertIn('中文语音助手', seen['system'])
        self.assertIn('不判断是否检索记忆', seen['system'])
        self.assertIn('当前用户：介绍一下向量数据库', seen['prompt'])
        self.assertTrue(all(label in {'是', '否'} for _, label in seen['examples']))

    def test_long_assistant_message_does_not_erase_user_context(self):
        router = QwenThinkingRouter(model='/unused')
        prompt = router._context_prompt('继续', [
            {'role': 'user', 'content': '请推导这个函数的积分。'},
            {'role': 'assistant', 'content': '说明' * 300},
        ], False)
        self.assertIn('请推导这个函数的积分。', prompt)
        self.assertLess(len(prompt), 400)

    def test_history_is_part_of_depth_cache(self):
        router = QwenThinkingRouter(model='/unused')
        prompts = []
        def predict(_system, _examples, prompt):
            prompts.append(prompt)
            return '是' if '推导' in prompt else '否'
        router._predict = predict
        simple = [{'role': 'user', 'content': '讲个故事。'}]
        deep = [{'role': 'user', 'content': '推导这个函数的积分。'}]
        self.assertEqual(router.classify('继续', history=simple).level, FAST)
        self.assertEqual(router.classify('继续', history=deep).level, SLOW)
        self.assertEqual(router.classify('继续', history=deep).level, SLOW)
        self.assertEqual(len(prompts), 2)

    def test_warmup_still_loads_the_inference_path(self):
        from unittest.mock import Mock
        router = QwenThinkingRouter(model='/unused')
        router._predict = Mock(return_value='否')
        self.assertEqual(router.warmup().level, FAST)
        router._predict.assert_called_once()


class MemoryRoutingIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, Mock
        from studio.core.utils.routing.component import Routing
        from voicemem.stream import empty_result
        self.agent = Routing()
        self.agent._THINKING_ROUTER_ON = True
        self.agent._replay_id = lambda *_: ''
        self.fresh = empty_result()
        self.agent.vm = SimpleNamespace(
            classify=Mock(return_value=SimpleNamespace(slots=[], entities=[])),
            search=Mock(return_value=self.fresh))
        self.router = SimpleNamespace(classify_async=AsyncMock(return_value=ThinkingDecision(FAST, '普通')))
        self.enterContext(patch('studio.core.utils.routing.component.thinking_router', return_value=self.router))

    def pending(self, *, route='deep', prepared=False, stranger=False):
        from types import SimpleNamespace
        from studio.core.utils.contracts.component import Pending
        from voicemem.stream import empty_result
        result = SimpleNamespace(search_mode='full') if prepared else empty_result()
        return Pending('我想接着上回那本书的话题聊。', 'existing-context' if prepared else '',
                       result, route=route, stranger=stranger, replay='existing-replay')

    async def test_normal_depth_does_not_discard_prefetched_memory(self):
        pending = self.pending(prepared=True)
        result = pending.result
        await self.agent.route_pending_thinking(pending)
        self.assertEqual(pending.reply_mode, 'memory')
        self.assertIs(pending.result, result)
        self.assertEqual(pending.memory_context, 'existing-context')
        self.assertEqual(pending.replay, 'existing-replay')
        self.agent.vm.search.assert_not_called()

    async def test_existing_gate_handles_history_without_studio_keyword_rules(self):
        from voicemem import gate
        for text in ('昨天聊到的那本书叫什么？', '我想接着上回那次旅行的话题聊。'):
            with self.subTest(text=text):
                pending = self.pending(prepared=True)
                pending.text = text
                pending.route = gate.route(text, semantic=False)
                self.assertEqual(pending.route, gate.DEEP)
                await self.agent.route_pending_thinking(pending)
                self.assertEqual(pending.reply_mode, 'memory')
                self.assertEqual(pending.memory_context, 'existing-context')
        self.agent.vm.search.assert_not_called()

    async def test_gate_memory_eligibility_completes_missing_retrieval(self):
        from voicemem import gate
        pending = self.pending()
        pending.text = '我明天有什么安排？'
        await self.agent.route_pending_thinking(pending)
        self.assertEqual((pending.reply_mode, pending.route), ('memory', gate.DEEP))
        self.agent.vm.classify.assert_called_once_with(pending.text)
        self.agent.vm.search.assert_called_once_with(pending.text, slots=[], entities=[], emotion='')
        self.assertIs(pending.result, self.fresh)

    async def test_gate_shallow_and_normal_depth_stay_direct(self):
        pending = self.pending(route='shallow', prepared=True)
        await self.agent.route_pending_thinking(pending)
        self.assertEqual(pending.reply_mode, 'direct')
        self.assertEqual(pending.memory_context, '')
        self.assertEqual(pending.replay, '')
        self.agent.vm.search.assert_not_called()

    async def test_deep_reasoning_upgrades_shallow_to_memory_cot(self):
        self.router.classify_async.return_value = ThinkingDecision(SLOW, '深思')
        pending = self.pending(route='shallow')
        await self.agent.route_pending_thinking(pending)
        self.assertEqual(pending.reply_mode, 'memory_cot')
        self.agent.vm.search.assert_called_once()

    async def test_classifier_failure_preserves_memory_and_privacy(self):
        self.router.classify_async.side_effect = RuntimeError('synthetic classifier failure')
        pending = self.pending(prepared=True)
        result = pending.result
        await self.agent.route_pending_thinking(pending)
        self.assertEqual(pending.reply_mode, 'memory')
        self.assertIs(pending.result, result)
        self.assertEqual(pending.memory_context, 'existing-context')
        stranger = self.pending(prepared=True, stranger=True)
        await self.agent.route_pending_thinking(stranger)
        self.assertEqual(stranger.memory_context, '')
        self.assertEqual(stranger.result.hits, [])
        self.agent.vm.search.assert_not_called()

    async def test_stranger_cannot_use_owner_memory_even_in_deep_mode(self):
        self.router.classify_async.return_value = ThinkingDecision(SLOW, '深思')
        pending = self.pending(prepared=True, stranger=True)
        await self.agent.route_pending_thinking(pending)
        self.assertEqual(pending.memory_context, '')
        self.assertEqual(pending.replay, '')
        self.agent.vm.search.assert_not_called()

    async def test_unavailable_search_does_not_silently_downgrade_memory_mode(self):
        self.agent.vm.search.side_effect = RuntimeError('synthetic retrieval failure')
        pending = self.pending()
        await self.agent.route_pending_thinking(pending)
        self.assertEqual(pending.reply_mode, 'memory')
        self.assertEqual(pending.memory_context, '')


if __name__ == "__main__":
    unittest.main()
