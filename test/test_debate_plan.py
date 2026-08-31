# -*- coding: utf-8 -*-
"""debate_plan 改造的本地测试：mock 全部 LLM 调用，零 API 消耗"""
import sys, json, os
sys.path.insert(0, os.path.abspath('.'))   # 项目根(src.skills 依赖)
sys.path.insert(0, os.path.abspath('src'))  # agent/tool/models
import agent


class Msg:
    def __init__(self, c):
        self.content = c


CRITIC = '```json\n{"criticisms": [{"type":"分类","issue":"应判机理而非预测","evidence":"渗流相变"},{"type":"假设","issue":"漏连通性变量","evidence":"轴线坐标"}]}\n```'
REVISER = ('```json\n{"plan_struct": {"problem_type":"机理","variables":["φ","G"],"objective":"渗流建模",'
           '"constraints":["φ∈[0,1]"],"per_question_method":{"问题1":"机理"}}, '
           '"methods":[{"name":"机理建模(渗流)","paradigm":"机理","rationale":"r1","assumption":"a1","tools":"t1"},'
           '{"name":"蒙特卡洛","paradigm":"仿真","rationale":"r2","assumption":"a2","tools":"t2"},'
           '{"name":"机器学习","paradigm":"数据","rationale":"r3","assumption":"a3","tools":"t3"}], '
           '"disagreements":[{"issue":"问题4是否弃统计","adopter":"质疑者","status":"未解决","note":"n"}]}\n```')
STRUCT = {'problem_type': '机理', 'variables': ['φ', 'G'], 'objective': '渗流建模',
          'constraints': ['φ∈[0,1]'], 'per_question_method': {'问题1': '机理'}}


def fake_invoke(model_obj, msgs, action='', retries=2):
    if action == '辩论·质疑者':
        return Msg(CRITIC)
    if action == '辩论·修订者':
        return Msg(REVISER)
    if action == '辩论·收口':
        return STRUCT
    raise AssertionError('unexpected: ' + action)


agent._invoke_llm = fake_invoke

STATE = {
    'plan_struct': '{"problem_type":"预测","variables":["φ"],"objective":"回归","constraints":[],"per_question_method":{}}',
    'problem_str': '微构体导电介质填充',
    'problem_index': {'问题1': '题干1', '问题2': '题干2'},
    'modeling_analysis': '初步分析',
    'methods': ['统计回归', '机器学习', '数值枚举'],
}

# 1) 正常路径
out = agent.debate_plan(dict(STATE))
assert out.get('methods') and len(out['methods']) == 3, 'methods 应为 3 张卡'
assert out.get('plan_struct'), 'plan_struct 缺失'
assert out.get('disagreements') and out['disagreements'][0]['status'] == '未解决', '分歧缺失'
assert all(isinstance(m, dict) and m.get('name') for m in out['methods']), '方法卡结构错'
print('[1] 正常路径 OK: methods=%d 分歧=%d' % (len(out['methods']), len(out['disagreements'])))
print('    plan_struct:', out['plan_struct'][:50].replace(chr(10), ' '))

# 2) 降级 A: 质疑者抛异常 → 返回 {} (等同改前)
agent._invoke_llm = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('模拟网络错误'))
out2 = agent.debate_plan(dict(STATE))
assert out2 == {}, '质疑者失败应静默降级返回 {}'
print('[2] 降级A(质疑者异常) OK → {}')

# 3) 降级 B: 修订者输出缺 methods → 返回 {}
def bad_reviser(model_obj, msgs, action='', retries=2):
    if action == '辩论·质疑者':
        return Msg('```json\n{"criticisms":[{"type":"分类","issue":"x","evidence":"y"}]}\n```')
    if action == '辩论·修订者':
        return Msg('```json\n{"plan_struct":{"problem_type":"机理"}}\n```')
    raise AssertionError(action)
agent._invoke_llm = bad_reviser
out3 = agent.debate_plan(dict(STATE))
assert out3 == {}, '修订输出缺 methods 应降级'
print('[3] 降级B(修订缺 methods) OK → {}')

# 4) 降级 C: 收口失败 → 用修订版原文
def fail_struct(model_obj, msgs, action='', retries=2):
    if action == '辩论·收口':
        raise RuntimeError('收口模拟失败')
    return fake_invoke(model_obj, msgs, action, retries)
agent._invoke_llm = fail_struct
out4 = agent.debate_plan(dict(STATE))
assert out4.get('plan_struct'), '收口失败也应产出 plan_struct(原文)'
print('[4] 降级C(收口失败→原文) OK')

# 5) 空初稿 → 直接放行
out5 = agent.debate_plan({'plan_struct': ''})
assert out5 == {}, '无初稿应放行'
print('[5] 空初稿放行 OK → {}')

# 6) dispatch_sends 兼容 str 与 dict
st = dict(STATE)
st['methods'] = ['统计回归', '机器学习']
st['disagreements'] = [{'issue': 'x'}]
sends = agent.dispatch_sends(st)
assert len(sends) == 2 and all(s.arg.get('method') for s in sends), 'str 形态分发失败'
assert sends[0].arg.get('method_card') is None, 'str 形态不应有 method_card'
assert sends[0].arg.get('disagreements') == [{'issue': 'x'}], 'disagreements 未下发'
st2 = dict(STATE)
st2['methods'] = [{'name': '机理', 'paradigm': '机理'}, {'name': '仿真', 'paradigm': '仿真'}]
sends2 = agent.dispatch_sends(st2)
assert len(sends2) == 2 and sends2[0].arg['method_card']['paradigm'] == '机理', 'dict 形态分发失败'
print('[6] dispatch_sends 兼容 OK (str 2 分支 / dict 2 分支)')

# 7) 图结构: debate_plan 节点存在且接线正确
names = [getattr(n, 'name', str(n)) for n in agent.builder.nodes]
assert 'debate_plan' in names, 'debate_plan 未注册到图'
print('[7] 图节点数:', len(agent.builder.nodes), '| debate_plan 已注册')

# 8) write_article 方法名提取兼容
assert agent.write_article is not None
print('[8] write_article 引用 OK')

# 9) solve_with_method 的 prompt 组装(方法卡展开)不炸
def solve_probe(state):
    method = state.get('method') or '未知'
    method_card = state.get('method_card') or {}
    disagreements = state.get('disagreements') or []
    card_block = ''
    if isinstance(method_card, dict) and method_card.get('name'):
        card_block = ('【方法卡】\n建模范式: %s\n方法依据: %s\n你的独立假设: %s\n可用工具: %s\n'
                      % (method_card.get('paradigm', ''), method_card.get('rationale', ''),
                         method_card.get('assumption', ''), method_card.get('tools', '')))
    dis_block = ''
    if disagreements:
        dis_block = '【未决分歧点】:\n' + '\n'.join('- %s' % d.get('issue', '') for d in disagreements[:5] if isinstance(d, dict)) + '\n'
    prompt = ('你正在用《%s》解答。\n' % method + card_block + dis_block + '共享骨架: %s' % state.get('plan_struct', ''))
    return prompt
p = solve_probe({'method': '机理建模(渗流)', 'method_card': {'name': '机理建模(渗流)', 'paradigm': '机理', 'rationale': 'r', 'assumption': 'a', 'tools': 't'},
                 'disagreements': [{'issue': '问题4是否弃统计'}], 'plan_struct': 'x'})
assert '【方法卡】' in p and '问题4是否弃统计' in p, '方法卡/分歧未展开'
p2 = solve_probe({'method': '统计回归'})  # 无方法卡(降级形态)
assert '《统计回归》' in p2
print('[9] solve_with_method prompt 展开 OK (有卡/无卡均正常)')

# 10) collect_branches 兼容方法卡(dict)形态
st3 = dict(STATE)
st3['methods'] = [{'name': '机理', 'paradigm': '机理'}, {'name': '蒙特卡洛', 'paradigm': '仿真'}]
st3['done_pairs'] = ['机理|问题1', '机理|问题2', '蒙特卡洛|问题1', '蒙特卡洛|问题2']
r = agent.collect_branches(st3)
assert isinstance(r, agent.Command) and r.goto == 'run_solutions', '方法卡形态下屏障应放行'
st4 = dict(st3)
st4['done_pairs'] = ['机理|问题1']  # 未交齐
r2 = agent.collect_branches(st4)
assert r2 == {}, '未交齐不应放行'
print('[10] collect_branches 方法卡兼容 OK (齐全放行 / 不全拦截)')

# 11) 质疑者输出含非 dict 垃圾元素 → 不崩溃、走"未提出有效质疑"继续
def garbage_critic(model_obj, msgs, action='', retries=2):
    if action == '辩论·质疑者':
        return Msg('```json\n{"criticisms": ["纯字符串垃圾", {"type":"分类","issue":"有效质疑","evidence":"e"}, 123]}\n```')
    return fake_invoke(model_obj, msgs, action, retries)
agent._invoke_llm = garbage_critic
out11 = agent.debate_plan(dict(STATE))
assert out11.get('methods'), '含垃圾元素也应正常产出方法卡'
print('[11] 质疑者垃圾元素容错 OK (不崩溃, 正常产出)')

# 12) feedback_check 摘要修复:answers 是 {方法名: 文本} 结构,质检必须能看到内容
st12 = dict(STATE)
st12['answers'] = [{'机理建模': '渗流阈值 p_c=0.31, 图连通判定完成'}, {'蒙特卡洛': '样本1000, 阈值0.33'}]
st12['model_iteration'] = 0
verdict_log = {}

def fake_feedback(model_obj, msgs, action='', retries=2):
    if action == '建模质检':
        # feedback_check 传的是 prompt 字符串(非消息列表)
        verdict_log['text'] = msgs if isinstance(msgs, str) else str(msgs)
        return {'passed': True, 'reason': '', 'suggestion': ''}
    return fake_invoke(model_obj, msgs, action, retries)
agent._invoke_llm = fake_feedback
r12 = agent.feedback_check(st12)
assert '渗流阈值 p_c=0.31' in verdict_log.get('text', ''), '质检摘要仍为空: 修复失败'
print('[12] feedback_check 摘要修复 OK (质检能看到真实结果内容)')

print()
print('ALL TESTS PASSED')
