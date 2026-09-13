"""Synthetic reasoning-depth check; memory eligibility is tested separately with fake stores."""
import argparse
import json
import time

from studio.core.utils.reply_modes.component import QwenThinkingRouter


CASES = (
    ('我明天有什么安排？', (), 'memory'),
    ('你好，我明天有什么安排？', (), 'memory'),
    ('你还记得我上次说喜欢什么吗？', (), 'memory'),
    ('我平时喜欢喝什么咖啡？', (), 'memory'),
    ('按我的口味，推荐一种咖啡。', (), 'memory'),
    ('结合我上次说过的偏好，讲个故事。', (), 'memory'),
    ('我上回决定去哪里旅行？', (), 'memory'),
    ('你好', (), 'direct'),
    ('请介绍一下你自己', (), 'direct'),
    ('讲一个小动物的故事。', (), 'direct'),
    ('咖啡一般有哪些种类？', (), 'direct'),
    ('什么是数学证明？', (), 'direct'),
    ('不用深入思考，简单介绍一下。', (), 'direct'),
    ('解释“请深入思考”这句话。', (), 'direct'),
    ('请深入思考一下这个问题。', (), 'memory_cot'),
    ('嗯，你能深度思考一下吗？', (), 'memory_cot'),
    ('计算 x 平方的不定积分。', (), 'memory_cot'),
    ('结合我以前说过的预算，比较三个方案的风险和收益。', (), 'memory_cot'),
    ('帮我设计一套兼顾成本、可靠性和上线时间的迁移方案。', (), 'memory_cot'),
    ('周日呢', (('user', '我上次说过的周末计划是什么？'), ('assistant', '你想问哪一天？')), 'memory'),
    ('继续', (('user', '请深入思考这个方案的取舍。'), ('assistant', '我先分析了成本，还需要考虑可靠性。')), 'memory_cot'),
    ('为什么', (('user', '天空为什么是蓝色的？'), ('assistant', '因为大气散射。')), 'direct'),
    ('谢谢', (('user', '计算这个函数的积分。'), ('assistant', '推导已经完成。')), 'direct'),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--assert-quality', action='store_true')
    args = parser.parse_args()
    router = QwenThinkingRouter(device=args.device)
    router.warmup()
    calls = 0
    predict = router._predict
    def counted(*a, **kw):
        nonlocal calls
        calls += 1
        return predict(*a, **kw)
    router._predict = counted
    rows = []
    for text, history, expected in CASES:
        start, before = time.monotonic(), calls
        decision = router.classify(text, history=[{'role': role, 'content': content} for role, content in history])
        rows.append({'text': text, 'expected_depth': 'high' if expected == 'memory_cot' else 'none',
                     'actual_depth': decision.reasoning_effort,
                     'model_called': calls != before, 'raw': decision.raw,
                     'wait_ms': round((time.monotonic() - start) * 1000, 1)})
    correct = sum(row['expected_depth'] == row['actual_depth'] for row in rows)
    print(json.dumps({'scope': 'reasoning_depth_only', 'correct': correct,
                      'total': len(rows), 'model_calls': calls, 'cases': rows},
                     ensure_ascii=False, indent=2))
    if args.assert_quality and correct / len(rows) < .9:
        raise SystemExit('Depth smoke check failed; this is not a memory-recall or live-conversation benchmark.')


if __name__ == '__main__':
    main()
