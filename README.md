# ProSIA

项目：多模态 Lysine PTM（ESM2 + ProtT5 + Structure）训练与特征整理。

快速开始：
1. 创建并激活虚拟环境：
   python -m venv .venv
   .venv\Scripts\activate

2. 安装依赖：
   pip install -r requirements.txt

3. 目录约定（将数据放到这些相对路径）：
   - data/...
   - plm_embeddings/esm2_embedding/*.pt
   - plm_embeddings/prott5_embedding/*.pt
   - structure/embedding/*.pt

4. 验证/整理 embeddings：
   python extract_embeddings.py --root "E:/多位点/ProSIA" --proteins "E:/多位点/ProSIA/data/all_2811_proteins.csv"

5. 运行训练脚本示例（快速 smoke test）：
   python run_mt_esm_prott5_gated_structure.py --experiment-name myexp --device cpu --num-folds 1 --epochs 1
