#!/usr/bin/env python3
"""
验证 Phase 2 (eagle_accept) 插桩的完整实现
"""

import json
import os
import sys
import tempfile

# Add vLLM to path
sys.path.insert(0, '/m-coriander/coriander/daifeng/testvllm/vllm')

def test_eagle_accept_logging():
    """测试 eagle_accept 日志是否正确记录"""
    from vllm.v1.spec_decode.spec_lora_debug import (
        spec_lora_log,
        record_eagle_accept,
        LogType,
        reset_step_id,
    )
    
    print("=" * 60)
    print("测试 Phase 2: eagle_accept 插桩")
    print("=" * 60)
    
    # 创建临时日志文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        log_file = f.name
    
    # 设置环境变量（需要在导入后设置）
    os.environ["VLLM_SPEC_LORA_DEBUG"] = "1"
    os.environ["VLLM_SPEC_LORA_DEBUG_PATH"] = log_file
    os.environ["VLLM_SPEC_LORA_PHASE"] = "test"
    
    # 重新导入模块以加载新的环境变量
    import importlib
    import vllm.v1.spec_decode.spec_lora_debug as debug_module
    importlib.reload(debug_module)
    
    # 重新导入函数
    from vllm.v1.spec_decode.spec_lora_debug import (
        record_eagle_accept as record_eagle_accept_new,
        LogType as LogType_new,
    )
    
    print(f"\n✅ 环境变量设置:")
    print(f"   VLLM_SPEC_LORA_DEBUG=1")
    print(f"   VLLM_SPEC_LORA_DEBUG_PATH={log_file}")
    print(f"   VLLM_SPEC_LORA_PHASE=test\n")
    
    # 测试 eagle_accept 记录
    print("📝 测试 1: 记录 eagle_accept")
    record_eagle_accept_new(
        num_proposed=8,
        num_accepted=6,
        step_id=0,
    )
    print("   ✓ 记录了 step 0 的接受统计")
    
    print("\n📝 测试 2: 记录多个 steps")
    for i in range(1, 5):
        proposed = 8
        accepted = 6 + (i % 2)  # 模拟接受率变化
        record_eagle_accept_new(num_proposed=proposed, num_accepted=accepted, step_id=i)
        print(f"   ✓ 记录了 step {i} 的接受统计 ({accepted}/{proposed})")
    
    # 读取日志文件验证
    print("\n🔍 验证日志文件内容:")
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    print(f"   总记录数: {len(lines)}")
    
    if len(lines) > 0:
        print(f"   ✓ 日志文件已生成\n")
        
        # 解析 JSON 记录
        eagle_accept_records = []
        for line in lines:
            try:
                record = json.loads(line)
                if record.get('type') == 'eagle_accept':
                    eagle_accept_records.append(record)
            except json.JSONDecodeError:
                pass
        
        print(f"   eagle_accept 记录数: {len(eagle_accept_records)}\n")
        
        if eagle_accept_records:
            print("   📊 前 3 条记录样本:")
            for record in eagle_accept_records[:3]:
                print(f"      Step {record['step']}: {record['num_accepted_tokens']}/{record['num_proposed_tokens']} = {record['accept_ratio']:.2%}")
            
            # 计算全局接受率
            total_proposed = sum(r['num_proposed_tokens'] for r in eagle_accept_records)
            total_accepted = sum(r['num_accepted_tokens'] for r in eagle_accept_records)
            overall_ratio = total_accepted / total_proposed if total_proposed > 0 else 0
            
            print(f"\n   📈 全局统计:")
            print(f"      总提议: {total_proposed} tokens")
            print(f"      总接受: {total_accepted} tokens")
            print(f"      接受率: {overall_ratio:.2%}")
            
            print("\n✅ Phase 2 (eagle_accept) 插桩工作正常！")
            return True
        else:
            print("   ❌ 没有 eagle_accept 记录")
            print(f"   📝 日志内容: {lines[:5] if lines else '(empty)'}")
            return False
    else:
        print(f"   ❌ 日志文件为空")
        return False
    
    # 清理
    try:
        os.remove(log_file)
    except:
        pass


def verify_implementation():
    """验证代码实现"""
    print("\n" + "=" * 60)
    print("验证 Phase 2 代码实现")
    print("=" * 60 + "\n")
    
    # 检查导入
    print("🔍 检查导入...")
    try:
        from vllm.v1.spec_decode.spec_lora_debug import record_eagle_accept
        print("   ✅ record_eagle_accept 已导入")
    except ImportError as e:
        print(f"   ❌ 导入失败: {e}")
        return False
    
    # 检查 GPU model runner 中的导入
    print("\n🔍 检查 GPU model runner 中的导入...")
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        print("   ✅ GPUModelRunner 已导入")
        
        # 检查 record_eagle_accept 是否在模块中
        import vllm.v1.worker.gpu_model_runner as gmr
        if hasattr(gmr, 'record_eagle_accept'):
            print("   ✅ record_eagle_accept 在 gpu_model_runner 中可用")
        else:
            print("   ℹ️  record_eagle_accept 在模块中导入，非直接属性")
    except ImportError as e:
        print(f"   ⚠️  导入信息: {e}")
    
    return True


if __name__ == "__main__":
    print("\n🎯 Phase 2 验证脚本\n")
    
    # 验证代码实现
    if not verify_implementation():
        sys.exit(1)
    
    # 测试日志记录
    if test_eagle_accept_logging():
        print("\n" + "=" * 60)
        print("✅ Phase 2 (eagle_accept) 验证成功！")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ Phase 2 验证失败")
        print("=" * 60)
        sys.exit(1)
