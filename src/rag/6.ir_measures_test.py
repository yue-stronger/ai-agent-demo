# -*- coding: utf-8 -*-
import ir_measures


def create_sample_data():
    """创建示例数据：查询、相关文档标注、检索结果"""
    
    # 1. 查询集合
    queries = [
        {"qid": "q1", "text": "游泳防水功能"},
        {"qid": "q2", "text": "手表续航时间"},
        {"qid": "q3", "text": "睡眠监测准确性"}
    ]
    
    # 2. 相关性标注（qrels）- 人工标注哪些文档与查询相关
    # 格式：查询ID, 文档ID, 相关性分数（0=不相关，1=相关）
    qrels = [
        # q1: 游泳防水功能 - 相关文档
        ir_measures.Qrel("q1", "doc2", 1),  # 运动模式文档（包含游泳）
        ir_measures.Qrel("q1", "doc4", 1),  # 注意事项文档（游泳防水）
        ir_measures.Qrel("q1", "doc1", 0),  # 产品介绍（不直接相关）
        ir_measures.Qrel("q1", "doc3", 0),  # 续航能力（不相关）
        ir_measures.Qrel("q1", "doc5", 0),  # 常见问题（不相关）
        
        # q2: 手表续航时间 - 相关文档
        ir_measures.Qrel("q2", "doc3", 1),  # 续航能力文档
        ir_measures.Qrel("q2", "doc1", 0),  # 产品介绍（不直接相关）
        ir_measures.Qrel("q2", "doc2", 0),  # 运动模式（不相关）
        ir_measures.Qrel("q2", "doc4", 0),  # 注意事项（不相关）
        ir_measures.Qrel("q2", "doc5", 0),  # 常见问题（不相关）
        
        # q3: 睡眠监测准确性 - 相关文档
        ir_measures.Qrel("q3", "doc1", 1),  # 产品介绍（健康监测）
        ir_measures.Qrel("q3", "doc5", 1),  # 常见问题（睡眠数据）
        ir_measures.Qrel("q3", "doc2", 0),  # 运动模式（不相关）
        ir_measures.Qrel("q3", "doc3", 0),  # 续航能力（不相关）
        ir_measures.Qrel("q3", "doc4", 0),  # 注意事项（不相关）
    ]
    
    # 3. 模拟RAG系统的检索结果
    # 格式：查询ID, 文档ID, 相关性分数（检索系统给出的分数）
    run_results = [
        # q1: 游泳防水功能 - 检索结果（按分数排序）
        ir_measures.ScoredDoc("q1", "doc4", 0.95),  # 正确：游泳防水相关
        ir_measures.ScoredDoc("q1", "doc2", 0.88),  # 正确：运动模式包含游泳
        ir_measures.ScoredDoc("q1", "doc1", 0.75),  # 错误：产品介绍不直接相关
        ir_measures.ScoredDoc("q1", "doc3", 0.65),  # 错误：续航不相关
        ir_measures.ScoredDoc("q1", "doc5", 0.45),  # 错误：常见问题不相关
        
        # q2: 手表续航时间 - 检索结果
        ir_measures.ScoredDoc("q2", "doc3", 0.92),  # 正确：续航能力相关
        ir_measures.ScoredDoc("q2", "doc1", 0.78),  # 错误：产品介绍不直接相关
        ir_measures.ScoredDoc("q2", "doc4", 0.68),  # 错误：注意事项不相关
        ir_measures.ScoredDoc("q2", "doc2", 0.55),  # 错误：运动模式不相关
        ir_measures.ScoredDoc("q2", "doc5", 0.42),  # 错误：常见问题不相关
        
        # q3: 睡眠监测准确性 - 检索结果
        ir_measures.ScoredDoc("q3", "doc5", 0.89),  # 正确：睡眠数据问题相关
        ir_measures.ScoredDoc("q3", "doc1", 0.82),  # 正确：健康监测相关
        ir_measures.ScoredDoc("q3", "doc2", 0.71),  # 错误：运动模式不相关
        ir_measures.ScoredDoc("q3", "doc3", 0.63),  # 错误：续航不相关
        ir_measures.ScoredDoc("q3", "doc4", 0.48),  # 错误：注意事项不相关
    ]
    
    return queries, qrels, run_results

def evaluate_rag_performance():
    """使用ir_measures评估RAG系统性能"""
    
    print("=== RAG系统性能评估 ===\n")
    
    # 1. 准备测试数据
    queries, qrels, run_results = create_sample_data()
    
    print("1. 测试数据概览：")
    print(f"   查询数量: {len(set(q['qid'] for q in queries))}")
    print(f"   相关性标注: {len(qrels)} 条")
    print(f"   检索结果: {len(run_results)} 条")
    
    # 2. 定义评估指标
    measures = [
        ir_measures.P(cutoff=3),      # Precision@3: 前3个结果中相关文档的比例
        ir_measures.R(cutoff=3),      # Recall@3: 前3个结果覆盖了多少相关文档
    ]
    
    # 手动计算F1@3（因为ir_measures 0.4.1没有F1指标）
    def calculate_f1_at_3(p_at_3, r_at_3):
        if p_at_3 + r_at_3 == 0:
            return 0.0
        return 2 * (p_at_3 * r_at_3) / (p_at_3 + r_at_3)
    
    # 3. 执行评估
    print("\n2. 开始评估...")
    results = ir_measures.calc_aggregate(measures, qrels, run_results)
    
    # 4. 输出总体指标
    print("\n3. 总体性能指标：")
    p_at_3 = results[ir_measures.P(cutoff=3)]
    r_at_3 = results[ir_measures.R(cutoff=3)]
    f1_at_3 = calculate_f1_at_3(p_at_3, r_at_3)
    
    print(f"   P@3: {p_at_3:.4f}")
    print(f"   R@3: {r_at_3:.4f}")
    print(f"   F1@3: {f1_at_3:.4f}")
    
    # 5. 分查询详细分析
    print("\n4. 分查询详细分析：")
    
    for qid in ["q1", "q2", "q3"]:
        query_text = next(q['text'] for q in queries if q['qid'] == qid)
        print(f"\n   查询 {qid}: {query_text}")
        
        # 显示该查询的检索结果
        query_run = [r for r in run_results if r.query_id == qid][:3]  # 只看前3个
        print("   检索结果（前3个）:")
        for i, result in enumerate(query_run, 1):
            # 检查是否相关
            is_relevant = any(q.doc_id == result.doc_id and q.relevance > 0 
                            for q in qrels if q.query_id == qid)
            status = "✓相关" if is_relevant else "✗不相关"
            print(f"     {i}. {result.doc_id} (分数: {result.score:.2f}) - {status}")
        
        # 为单个查询计算指标
        query_qrels = [q for q in qrels if q.query_id == qid]
        query_results = [r for r in run_results if r.query_id == qid]
        single_query_results = ir_measures.calc_aggregate(measures, query_qrels, query_results)
        
        p_score = single_query_results[ir_measures.P(cutoff=3)]
        r_score = single_query_results[ir_measures.R(cutoff=3)]
        f1_score = calculate_f1_at_3(p_score, r_score)
        
        print(f"     P@3: {p_score:.4f}")
        print(f"     R@3: {r_score:.4f}")
        print(f"     F1@3: {f1_score:.4f}")

def analyze_results():
    """分析评估结果并给出改进建议"""
    
    print("\n=== 结果分析与改进建议 ===")
    
    # 模拟一些典型的RAG问题场景
    scenarios = [
        {
            "问题": "Precision@3 较低",
            "原因": "检索到的前3个文档中包含不相关内容",
            "改进建议": [
                "优化文本分块策略，确保语义完整性",
                "调整嵌入模型，选择更适合领域的模型",
                "增加查询扩展或重写机制"
            ]
        },
        {
            "问题": "Recall@3 较低", 
            "原因": "相关文档没有被检索到前3位",
            "改进建议": [
                "增加检索结果数量（如top-k从3增加到5）",
                "使用混合检索（向量+关键词）",
                "优化向量数据库的相似度计算方法"
            ]
        },
        {
            "问题": "F1@3 不平衡",
            "原因": "Precision和Recall差距较大",
            "改进建议": [
                "平衡检索策略，避免过度优化单一指标",
                "使用重排序模型进一步优化结果顺序",
                "根据业务场景调整P/R权重"
            ]
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['问题']}")
        print(f"   原因: {scenario['原因']}")
        print("   改进建议:")
        for suggestion in scenario['改进建议']:
            print(f"     • {suggestion}")

if __name__ == "__main__":
    # 执行RAG评估
    evaluate_rag_performance()
    
    # 分析结果
    analyze_results()
    
    print("\n=== 评估完成 ===")
    print("提示: 在实际应用中，建议:")
    print("1. 收集真实用户查询和相关性标注")
    print("2. 定期评估和监控RAG系统性能")
    print("3. 根据评估结果持续优化检索和生成策略")