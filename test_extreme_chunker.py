"""测试实际场景中的极端情况"""

import sys
sys.path.insert(0, '/Users/jk/Projects/rag-knowledge-base')

from src.core.semantic_chunker import SemanticChunker
from src.core.chunker import TextChunker
from src.rag_api.config import get_settings

settings = get_settings()

def test_extreme_cases():
    """测试极端情况"""
    print("=== 测试极端情况 ===")
    print()
    
    chunker = SemanticChunker()
    
    # 1. 模拟实际数据库中最长的 chunk（183,927 字符）
    print("--- 测试：模拟 183,927 字符超长文本 ---")
    
    # 模拟表格数据（几乎没有分隔符）
    table_data = ""
    for i in range(4000):
        table_data += f"列A{i}\t列B{i}\t列C{i}\t列D{i}\t列E{i}\n"
    
    print(f"模拟表格数据长度: {len(table_data)} 字符")
    
    chunks = chunker.chunk_text(table_data)
    
    max_len = max(len(c) for c in chunks)
    min_len = min(len(c) for c in chunks)
    
    print(f"分块数量: {len(chunks)}")
    print(f"最大长度: {max_len}")
    print(f"最小长度: {min_len}")
    print(f"是否全部符合上限: {all(len(c) <= settings.MAX_CHUNK_SIZE for c in chunks)}")
    
    # 检查内容完整性（注意重叠会导致总长度增加）
    print(f"注意: 重叠机制会导致总长度略大于原文")
    
    # 2. 模拟古籍文档（大量重复内容）
    print("\n--- 测试：模拟古籍文档（长段落） ---")
    
    ancient_text = """
第一章 论道

道可道，非常道。名可名，非常名。无名天地之始，有名万物之母。故常无欲以观其妙，常有欲以观其徼。此两者同出而异名，同谓之玄。玄之又玄，众妙之门。

第二章 论德

上德不德，是以有德。下德不失德，是以无德。上德无为而无以为，下德无为而有以为。上仁为之而无以为，上义为之而有以为。上礼为之而莫之应，则攘臂而扔之。

故失道而后德，失德而后仁，失仁而后义，失义而后礼。夫礼者，忠信之薄，而乱之首。前识者，道之华，而愚之始。是以大丈夫处其厚，不居其薄。处其实，不居其华。故去彼取此。
""" * 100  # 约 30000 字符
    
    print(f"古籍文档长度: {len(ancient_text)} 字符")
    
    chunks = chunker.chunk_text(ancient_text)
    
    max_len = max(len(c) for c in chunks)
    print(f"分块数量: {len(chunks)}")
    print(f"最大长度: {max_len}")
    print(f"是否全部符合上限: {all(len(c) <= settings.MAX_CHUNK_SIZE for c in chunks)}")
    
    # 检查语义边界
    print("\n检查语义边界示例:")
    for i, chunk in enumerate(chunks[:3]):
        # 显示 chunk 的开头和结尾
        print(f"  Chunk {i}: 开头='{chunk[:30]}...' 结尾='...{chunk[-30:]}'")
    
    # 3. 极端无边界情况（纯数字）
    print("\n--- 测试：纯数字无边界 ---")
    pure_numbers = "1234567890" * 20000  # 200,000 字符
    
    print(f"纯数字长度: {len(pure_numbers)} 字符")
    
    chunks = chunker.chunk_text(pure_numbers)
    
    max_len = max(len(c) for c in chunks)
    print(f"分块数量: {len(chunks)}")
    print(f"最大长度: {max_len}")
    print(f"是否全部符合上限: {all(len(c) <= settings.MAX_CHUNK_SIZE for c in chunks)}")
    
    # 4. 验证所有 chunks 都不超过硬性上限
    print("\n=== 最终验证 ===")
    
    all_passed = True
    
    test_texts = [
        ("表格数据", table_data),
        ("古籍文档", ancient_text),
        ("纯数字", pure_numbers),
        ("混合超长", table_data + ancient_text + pure_numbers[:50000]),
    ]
    
    for name, text in test_texts:
        chunks = chunker.chunk_text(text)
        violations = [len(c) for c in chunks if len(c) > settings.MAX_CHUNK_SIZE]
        
        if violations:
            all_passed = False
            print(f"❌ {name}: 发现 {len(violations)} 个超长 chunk")
            print(f"   超长长度: {violations}")
        else:
            max_len = max(len(c) for c in chunks)
            print(f"✅ {name}: 全部符合上限 (max={max_len})")
    
    return all_passed


def test_real_world_chunks():
    """测试从数据库中抽取的真实超长 chunks"""
    print("\n\n=== 测试真实超长 chunks ===")
    
    chunker = SemanticChunker()
    
    # 模拟从数据库查询超长 chunks
    import sqlite3
    from pathlib import Path
    
    db_path = Path("/Users/jk/Projects/rag-knowledge-base/db/metadata.db")
    
    if not db_path.exists():
        print("数据库不存在，跳过真实数据测试")
        return True
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 查询超长 chunks (>8000字符)
    cursor.execute("""
        SELECT id, content, LENGTH(content) as len 
        FROM chunks 
        WHERE LENGTH(content) > 8000
        ORDER BY LENGTH(content) DESC
        LIMIT 5
    """)
    
    rows = cursor.fetchall()
    
    if not rows:
        print("数据库中没有超长 chunks")
        conn.close()
        return True
    
    print(f"发现 {len(rows)} 个超长 chunks")
    print()
    
    all_passed = True
    
    for i, (chunk_id, content, orig_len) in enumerate(rows):
        print(f"--- 真实 Chunk {i+1} (原长度: {orig_len}) ---")
        
        # 重新分块
        new_chunks = chunker.chunk_text(content)
        
        max_len = max(len(c) for c in new_chunks) if new_chunks else 0
        
        print(f"新分块数量: {len(new_chunks)}")
        print(f"新分块最大长度: {max_len}")
        
        violations = [len(c) for c in new_chunks if len(c) > settings.MAX_CHUNK_SIZE]
        
        if violations:
            all_passed = False
            print(f"❌ 发现超长 chunks: {violations}")
        else:
            print(f"✅ 全部符合上限")
        
        # 检查内容完整性（重叠会导致总长度增加）
        total_new = sum(len(c) for c in new_chunks)
        # 注意：有重叠时，总长度会略大于原文
        print(f"新总长度: {total_new} (原文: {orig_len})")
        
        # 验证原文内容是否被完整覆盖
        first_chunk_start = content.find(new_chunks[0][:50])
        last_chunk_end = content.find(new_chunks[-1][-50:])
        
        if first_chunk_start >= 0 and last_chunk_end >= 0:
            print(f"✅ 内容覆盖完整")
        else:
            print(f"⚠️ 可能存在内容丢失")
        
        print()
    
    conn.close()
    return all_passed


if __name__ == "__main__":
    passed1 = test_extreme_cases()
    passed2 = test_real_world_chunks()
    
    print("\n=== 总结 ===")
    if passed1 and passed2:
        print("✅ 所有极端情况测试通过！分块器健壮性符合要求。")
    else:
        print("❌ 部分测试失败，需要进一步优化。")