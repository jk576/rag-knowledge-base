"""测试启发式语义分块器"""

import sys
sys.path.insert(0, '/Users/jk/Projects/rag-knowledge-base')

from src.core.semantic_chunker import SemanticChunker
from src.core.chunker import TextChunker
from src.rag_api.config import get_settings

settings = get_settings()

def test_semantic_chunker():
    """测试语义分块器"""
    print("=== 测试启发式语义分块器 ===")
    print(f"配置: target={settings.CHUNK_SIZE}, max={settings.MAX_CHUNK_SIZE}, min={settings.MIN_CHUNK_SIZE}")
    print()
    
    chunker = SemanticChunker()
    
    # 测试场景
    test_cases = [
        # 1. 正常长度文本
        ("正常段落", """
这是一个正常的段落，长度适中。它应该被作为一个完整的 chunk 返回。

这是第二个段落，同样长度适中。每个段落应该独立成为一个 chunk。
        """.strip()),
        
        # 2. 超长文本（模拟表格数据）
        ("超长表格文本（无边界）", "A1	B1	C1	D1	E1	F1	G1	H1	I1	J1	K1	L1	M1	N1	O1	P1	Q1	R1	S1	T1	U1	V1	W1	X1	Y1	Z1\n" * 1000),
        
        # 3. 带标题的文档
        ("Markdown 文档", """
# 第一章 概述

这是第一章的内容。介绍了系统的基本概念和设计理念。

# 第二章 详细设计

## 2.1 系统架构

系统采用分层架构设计，包括表示层、业务层、数据层。

## 2.2 核心模块

核心模块包括用户管理、权限控制、数据处理等功能。

# 第三章 实现方案

本章详细介绍各模块的实现方案和技术选型。
        """.strip()),
        
        # 4. 超长单段落（>8000字符）
        ("超长单段落", "测试内容。" * 5000),  # 约 25000 字符
        
        # 5. 混合内容
        ("混合内容", """
# 项目背景

本项目旨在解决现有系统存在的问题，提升用户体验。

1. 问题一：性能瓶颈
   当前系统在高并发场景下性能下降明显。

2. 问题二：数据一致性
   分布式环境下数据同步存在延迟。

3. 问题三：用户体验
   界面设计不够友好，操作流程复杂。

表格数据示例：
列1	列2	列3	列4	列5
数据1	数据2	数据3	数据4	数据5

结论：通过本次优化，系统性能提升50%，用户满意度提升80%。
        """.strip()),
    ]
    
    results = []
    
    for name, text in test_cases:
        print(f"\n--- 测试: {name} ---")
        print(f"原文长度: {len(text)} 字符")
        
        chunks = chunker.chunk_text(text)
        
        print(f"分块数量: {len(chunks)}")
        print(f"长度分布: {[len(c) for c in chunks[:10]]}{'...' if len(chunks) > 10 else ''}")
        
        # 验证关键指标
        max_len = max(len(c) for c in chunks) if chunks else 0
        min_len = min(len(c) for c in chunks) if chunks else 0
        
        passed = True
        issues = []
        
        # 检查是否超过硬性上限
        if max_len > settings.MAX_CHUNK_SIZE:
            passed = False
            issues.append(f"❌ 超长 chunk: {max_len} > {settings.MAX_CHUNK_SIZE}")
        else:
            issues.append(f"✅ 长度符合上限: {max_len} <= {settings.MAX_CHUNK_SIZE}")
        
        # 检查是否有过小的 chunk（特殊情况除外）
        if min_len < 50 and len(chunks) > 1:
            issues.append(f"⚠️ 可能过小的 chunk: {min_len}")
        
        # 检查语义完整性（简单检查）
        for i, chunk in enumerate(chunks):
            # 不应该在词中间切分（除非必要）
            if chunk.endswith((" ", "\t")) and len(chunk) > 100:
                issues.append(f"⚠️ Chunk {i} 在空白处结尾")
        
        results.append({
            "name": name,
            "passed": passed,
            "original_len": len(text),
            "chunks": len(chunks),
            "max_len": max_len,
            "min_len": min_len,
            "issues": issues
        })
        
        for issue in issues:
            print(issue)
    
    return results


def test_text_chunker_integration():
    """测试 TextChunker 集成"""
    print("\n\n=== 测试 TextChunker 集成 ===")
    
    # 测试语义分块模式
    chunker_semantic = TextChunker(use_semantic=True)
    chunker_legacy = TextChunker(use_semantic=False)
    
    # 超长文本测试
    long_text = "这是一个测试段落，用于验证分块器的行为。" * 1000  # 约 35000 字符
    
    print(f"\n原文长度: {len(long_text)} 字符")
    
    # 语义分块
    semantic_chunks = chunker_semantic.chunk_text(long_text)
    print(f"\n语义分块:")
    print(f"  数量: {len(semantic_chunks)}")
    print(f"  最大长度: {max(len(c) for c in semantic_chunks)}")
    print(f"  是否全部符合上限: {all(len(c) <= settings.MAX_CHUNK_SIZE for c in semantic_chunks)}")
    
    # 传统分块
    legacy_chunks = chunker_legacy.chunk_text(long_text)
    print(f"\n传统分块:")
    print(f"  数量: {len(legacy_chunks)}")
    print(f"  最大长度: {max(len(c) for c in legacy_chunks)}")
    print(f"  是否全部符合上限: {all(len(c) <= settings.MAX_CHUNK_SIZE for c in legacy_chunks)}")


def test_boundary_cases():
    """测试边界情况"""
    print("\n\n=== 测试边界情况 ===")
    
    chunker = SemanticChunker()
    
    cases = [
        ("空文本", ""),
        ("仅空白", "   \n\n   "),
        ("极短文本", "测试"),
        ("恰好 max_size", "测" * settings.MAX_CHUNK_SIZE),
        ("略超 max_size", "测" * (settings.MAX_CHUNK_SIZE + 100)),
        ("2倍 max_size", "测" * (settings.MAX_CHUNK_SIZE * 2)),
        ("5倍 max_size", "测" * (settings.MAX_CHUNK_SIZE * 5)),
    ]
    
    for name, text in cases:
        print(f"\n--- {name} (len={len(text)}) ---")
        chunks = chunker.chunk_text(text)
        
        if not chunks:
            print("  结果: 空列表（正确处理空/无效文本）")
            continue
        
        max_len = max(len(c) for c in chunks)
        print(f"  chunks: {len(chunks)}, max_len: {max_len}")
        
        if max_len <= settings.MAX_CHUNK_SIZE:
            print(f"  ✅ 符合上限")
        else:
            print(f"  ❌ 超过上限! {max_len} > {settings.MAX_CHUNK_SIZE}")


if __name__ == "__main__":
    results = test_semantic_chunker()
    test_text_chunker_integration()
    test_boundary_cases()
    
    # 总结
    print("\n\n=== 测试总结 ===")
    passed_count = sum(1 for r in results if r["passed"])
    print(f"核心测试: {passed_count}/{len(results)} 通过")
    
    if passed_count == len(results):
        print("✅ 所有测试通过！分块器符合硬性上限要求。")
    else:
        print("❌ 部分测试失败，需要修复。")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['name']}: {r['issues']}")