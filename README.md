# Chain-of-Thought 对比实验

本仓库实现了一份 Zero-Shot 与 Zero-Shot-CoT 的对比实验脚本，以及自动生成《实验报告.pdf》的工具脚本。你可以直接运行脚本调用真实 API，或者基于 `data/sample_run.json` 中的示例数据重放实验流程。

## 项目结构
- `cot_experiment.py`：核心实验脚本，负责调用大模型 API 并保存每道题在两种提示方式下的回答。
- `generate_report.py`：读取 `results.json` 并生成符合作业模板的《实验报告.pdf》。
- `data/sample_run.json`：实验记录，可用于复现实验流程/调试报告生成。
- `outputs/`：脚本运行后自动产生的结果目录（包含 JSON 与 Markdown 对比表）。
- `实验报告.pdf`

## 环境准备
1. 建议使用 Python ≥ 3.10。系统若没有 `pip`，可以运行 `python3 get-pip.py --break-system-packages` 获取。
2. 安装依赖：
   ```bash
   python3 -m pip install --break-system-packages -r <(cat <<'REQ'
   openai>=1.51.0
   fpdf2>=2.7.8
REQ
   )
   ```
   或者在虚拟环境中通过 `pip install -e .` 安装。

## 运行实验
1. 设置 API Key（示例）：
   ```bash
   export OPENAI_API_KEY="sk-xxxx"
   ```
2. 执行实验脚本：
   ```bash
   python3 cot_experiment.py \
     --model gpt-4o-mini \
     --temperature 0.1 \
     --save-markdown
   ```
   - 默认问题列表定义在 `cot_experiment.py` → `DEFAULT_QUESTIONS`。
   - 若要自定义输入，可提供 `--questions-file my_questions.json`（格式参照 `QuestionSpec`）。
   - `--dry-run`：完全跳过 API，生成可重复的模拟数据。
   - `--replay-file`：从现有 JSON 结果中复现实验（示例命令：`python3 cot_experiment.py --replay-file data/sample_run.json --tag sample-run`）。

脚本会把每次运行保存到 `outputs/<run_id>/` 下，包含：
- `results.json`：完整 prompt、回答、自动判定。
- `results.md`：对比表，方便复制到报告。

## 生成实验报告
1. 选择想要写入报告的 `results.json`（如 `outputs/sample-run/results.json`）。
2. 运行：
   ```bash
   python3 generate_report.py \
     --results outputs/sample-run/results.json \
     --course "<课程名称>" \
     --name "<姓名>" \
     --student-id "<学号>" \
     --output 实验报告.pdf
   ```
   - 默认字体引用系统自带 `NotoSansCJK`，可通过 `--font` 指向其他 TTF/OTF。
   - 报告内容包含：基础信息、引言、实验方法、结果分析、讨论、结论、附录（完整源代码）。

## 作业提交建议
1. 根据实际 API 调用重新运行 `cot_experiment.py`，确保 `outputs/<run_id>/results.json` 记录真实回答。
2. 用最新结果生成报告：`python3 generate_report.py --results outputs/<run_id>/results.json ...`。
3. 打包提交所需文件：
   ```bash
   zip -r 姓名-学号-COT实验.zip cot_experiment.py 实验报告.pdf outputs/<run_id>
   ```
   （如需附上 `data/sample_run.json` 供复现，可一起打包。）

如需扩展题目、模型或评估策略，只需修改 `cot_experiment.py` 中的配置与逻辑即可。祝实验顺利！
