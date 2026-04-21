#!/usr/bin/env python3
"""
Phase 3 & 4 验证脚本：LoRA State 和 Runtime Step 插桩
"""

import os
import sys
import json
import tempfile
import importlib

# 添加 vLLM 到路径
sys.path.insert(0, '/workspace')

def test_phase3_lora_state():
    """测试 Phase 3: LoRA state 插桩"""
    print("\n" + "="*60)
    print("测试 Phase 3: lora_state 插桩")
    print("="*60)
    
    # 设置环境变量
    log_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False).name
    os.environ['VLLM_SPEC_LORA_DEBUG'] = '1'
    os.environ['VLLM_SPEC_LORA_DEBUG_PATH'] = log_file
    os.environ['VLLM_SPEC_LORA_PHASE'] = 'test'
    
    print(f"\n✅ 环境变量设置:")
    print(f"   VLLM_SPEC_LORA_DEBUG={os.environ['VLLM_SPEC_LORA_DEBUG']}")
    print(f"   VLLM_SPEC_LORA_DEBUG_PATH={log_file}")
    print(f"   VLLM_SPEC_LORA_PHASE=test")
    
    # 导入 spec_lora_debug 模块
    try:
        from vllm.v1.spec_decode import spec_lora_debug
        importlib.reload(spec_lora_debug)
        from vllm.v1.spec_decode.spec_lora_debug import record_lora_state
        print("\n   ✅ record_lora_state 已导入")
    except ImportError as e:
        print(f"\n   ❌ 导入失败: {e}")
        return False
    
    # 测试记录 LoRA state
    print(f"\n📝 测试 1: 记录基础 LoRA state")
    try:
        record_lora_state(
            active_lora_names=['lora_1'],
            num_active_loras=1,
            phase='base'
        )
        print(f"   ✓ 记录了基础 LoRA state")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return False
    
    print(f"\n📝 测试 2: 记录多个活跃 LoRA")
    try:
        for step in range(1, 5):
            record_lora_state(
                active_lora_names=[f'lora_{i}' for i in range(1, step+1)],
                num_active_loras=step,
                phase='lora_infra',
                step_id=step
            )
            print(f"   ✓ 记录了 step {step} 的 LoRA state ({step} 个活跃 adapter)")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return False
    
    # 验证日志文件
    print(f"\n🔍 验证日志文件内容:")
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        print(f"   总记录数: {len(lines)}")
        
        if len(lines) > 0:
            print(f"   ✓ 日志文件已生成\n")
            
            lora_state_records = []
            for line in lines:
                try:
                    record = json.loads(line)
                    if record.get('type') == 'lora_state':
                        lora_state_records.append(record)
                except json.JSONDecodeError:
                    pass
            
            print(f"   lora_state 记录数: {len(lora_state_records)}\n")
            
            if lora_state_records:
                print("   📊 记录样本:")
                for record in lora_state_records[:3]:
                    print(f"      Step {record['step']}: {record['num_active_loras']} 个 LoRA - {record['active_lora_names']}")
                
                print("\n✅ Phase 3 (lora_state) 插桩工作正常！")
                return True
            else:
                print("   ❌ 没有 lora_state 记录")
                return False
        else:
            print(f"   ❌ 日志文件为空")
            return False
    except Exception as e:
        print(f"   ❌ 验证失败: {e}")
        return False
    finally:
        try:
            os.remove(log_file)
        except:
            pass


def test_phase4_runtime_step():
    """测试 Phase 4: Runtime step 插桩"""
    print("\n" + "="*60)
    print("测试 Phase 4: runtime_step 插桩")
    print("="*60)
    
    # 设置环境变量
    log_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False).name
    os.environ['VLLM_SPEC_LORA_DEBUG'] = '1'
    os.environ['VLLM_SPEC_LORA_DEBUG_PATH'] = log_file
    os.environ['VLLM_SPEC_LORA_PHASE'] = 'test'
    
    print(f"\n✅ 环境变量设置:")
    print(f"   VLLM_SPEC_LORA_DEBUG={os.environ['VLLM_SPEC_LORA_DEBUG']}")
    print(f"   VLLM_SPEC_LORA_DEBUG_PATH={log_file}")
    print(f"   VLLM_SPEC_LORA_PHASE=test")
    
    # 导入 spec_lora_debug 模块
    try:
        from vllm.v1.spec_decode import spec_lora_debug
        importlib.reload(spec_lora_debug)
        from vllm.v1.spec_decode.spec_lora_debug import record_runtime_step
        print("\n   ✅ record_runtime_step 已导入")
    except ImportError as e:
        print(f"\n   ❌ 导入失败: {e}")
        return False
    
    # 测试记录 runtime step
    print(f"\n📝 测试 1: 记录单个 runtime step")
    try:
        record_runtime_step(
            dt=0.05,
            num_tokens=128,
            throughput=2560.0,
            phase='base'
        )
        print(f"   ✓ 记录了 runtime step")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return False
    
    print(f"\n📝 测试 2: 记录多个 runtime steps")
    try:
        for step in range(1, 5):
            dt = 0.05 + (step * 0.01)
            num_tokens = 128 * (step + 1)
            throughput = num_tokens / dt
            
            record_runtime_step(
                dt=dt,
                num_tokens=num_tokens,
                throughput=throughput,
                phase='lora_infra',
                step_id=step
            )
            print(f"   ✓ 记录了 step {step} 的 runtime metrics ({num_tokens} tokens, {throughput:.1f} tok/s)")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return False
    
    # 验证日志文件
    print(f"\n🔍 验证日志文件内容:")
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        print(f"   总记录数: {len(lines)}")
        
        if len(lines) > 0:
            print(f"   ✓ 日志文件已生成\n")
            
            runtime_step_records = []
            for line in lines:
                try:
                    record = json.loads(line)
                    if record.get('type') == 'runtime_step':
                        runtime_step_records.append(record)
                except json.JSONDecodeError:
                    pass
            
            print(f"   runtime_step 记录数: {len(runtime_step_records)}\n")
            
            if runtime_step_records:
                print("   📊 记录样本:")
                for record in runtime_step_records[:3]:
                    print(f"      Step {record['step']}: dt={record['dt']:.3f}s, {record['num_tokens']} tokens, {record['throughput']:.1f} tok/s")
                
                # 计算全局统计
                total_time = sum(r['dt'] for r in runtime_step_records)
                total_tokens = sum(r['num_tokens'] for r in runtime_step_records)
                overall_throughput = total_tokens / total_time if total_time > 0 else 0
                
                print(f"\n   📈 全局统计:")
                print(f"      总时间: {total_time:.3f}s")
                print(f"      总 tokens: {total_tokens}")
                print(f"      平均吞吐: {overall_throughput:.1f} tok/s")
                
                print("\n✅ Phase 4 (runtime_step) 插桩工作正常！")
                return True
            else:
                print("   ❌ 没有 runtime_step 记录")
                return False
        else:
            print(f"   ❌ 日志文件为空")
            return False
    except Exception as e:
        print(f"   ❌ 验证失败: {e}")
        return False
    finally:
        try:
            os.remove(log_file)
        except:
            pass


if __name__ == '__main__':
    print("\n🎯 Phase 3 & 4 验证脚本\n")
    
    result3 = test_phase3_lora_state()
    result4 = test_phase4_runtime_step()
    
    print("\n" + "="*60)
    if result3 and result4:
        print("✅ Phase 3 & 4 (lora_state + runtime_step) 验证成功！")
        print("="*60)
        sys.exit(0)
    else:
        print("❌ Phase 3 & 4 验证失败！")
        print("="*60)
        sys.exit(1)
