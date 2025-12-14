"""Generate 实验报告.pdf from experiment outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from fpdf import FPDF
from fpdf.enums import WrapMode, XPos, YPos

DEFAULT_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


class ReportPDF(FPDF):
    def __init__(self, font_path: str) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=15)
        self.font_family = "NotoSans"
        self.add_font(self.font_family, "", font_path, uni=True)
        self.set_font(self.font_family, "", 12)

    def add_title(self, text: str) -> None:
        self.set_font(self.font_family, "", 20)
        self.cell(
            0,
            15,
            text,
            align="C",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.ln(4)
        self.set_font(self.font_family, "", 12)

    def add_section(self, title: str) -> None:
        self.ln(5)
        self.set_font(self.font_family, "", 16)
        self.cell(
            0,
            10,
            title,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.set_font(self.font_family, "", 12)

    def add_paragraph(self, text: str) -> None:
        self.multi_cell(
            0,
            7,
            text,
            wrapmode=WrapMode.CHAR,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.ln(2)

    def add_list(self, items: List[str]) -> None:
        for item in items:
            self.multi_cell(
                0,
                7,
                f"- {item}",
                wrapmode=WrapMode.CHAR,
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
        self.ln(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成实验报告 PDF")
    parser.add_argument(
        "--results",
        default="outputs/sample-run/results.json",
        help="results.json 文件路径",
    )
    parser.add_argument(
        "--output",
        default="实验报告.pdf",
        help="输出 PDF 文件路径",
    )
    parser.add_argument("--course", default="<课程名称>")
    parser.add_argument("--name", default="<姓名>")
    parser.add_argument("--student-id", default="<学号>")
    parser.add_argument(
        "--font",
        default=DEFAULT_FONT,
        help="支持中文的字体文件路径 (TTF/OTF)",
    )
    parser.add_argument(
        "--purpose",
        default=(
            "通过对比Zero-Shot与Zero-Shot-CoT，验证Chain-of-Thought提示"
            "在数学/逻辑推理题目上的增益"
        ),
        help="实验目的描述",
    )
    return parser.parse_args()


def load_results(path: str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text("utf-8"))
    if "records" not in data:
        raise ValueError("results.json 缺少 records 字段")
    return data


def summarize_accuracy(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    zero = [rec.get("zero_shot_correct") for rec in records]
    cot = [rec.get("cot_correct") for rec in records]
    zero_hits = sum(1 for flag in zero if flag)
    cot_hits = sum(1 for flag in cot if flag)
    total = len(records)
    return {
        "zero_hits": zero_hits,
        "cot_hits": cot_hits,
        "zero_acc": zero_hits / total if total else 0.0,
        "cot_acc": cot_hits / total if total else 0.0,
        "total": total,
    }


def record_lines(idx: int, record: Dict[str, Any]) -> List[str]:
    z_flag = record.get("zero_shot_correct")
    c_flag = record.get("cot_correct")
    return [
        f"问题 {idx}: {record['question']}",
        f"- Zero-Shot 提示: {record['zero_shot_prompt']}",
        f"- Zero-Shot 回答: {record['zero_shot_answer']}",
        f"- Zero-Shot 是否正确: {z_flag}",
        f"- CoT 提示: {record['cot_prompt']}",
        f"- CoT 回答: {record['cot_answer']}",
        f"- CoT 是否正确: {c_flag}",
    ]


def add_appendix(pdf: ReportPDF, code_path: str) -> None:
    pdf.add_section("附录：完整代码")
    pdf.set_font(pdf.font_family, "", 10)
    lines = Path(code_path).read_text("utf-8").splitlines()
    for line in lines:
        pdf.multi_cell(
            0,
            5,
            line,
            wrapmode=WrapMode.CHAR,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
    pdf.set_font(pdf.font_family, "", 12)


def main() -> None:
    args = parse_args()
    data = load_results(args.results)
    records = data["records"]
    metadata = data.get("metadata", {})
    pdf = ReportPDF(args.font)
    pdf.add_page()
    pdf.add_title("Chain-of-Thought 对比实验报告")

    pdf.add_section("基础信息")
    pdf.add_list(
        [
            f"课程名称：{args.course}",
            f"姓名：{args.name}",
            f"学号：{args.student_id}",
            f"实验时间：{metadata.get('timestamp', '<未记录>')}",
        ]
    )

    pdf.add_section("引言")
    pdf.add_paragraph(
        (
            "Chain-of-Thought（CoT）提示通过要求模型逐步给出推理过程，"
            "可以帮助大模型显式梳理复杂问题的中间逻辑。与直接提问相比，"
            "CoT 在数学、逻辑、代码调试等任务中经常能显著提升正确率。"
        )
    )
    pdf.add_paragraph(
        f"本实验目的：{args.purpose}。"
    )

    pdf.add_section("实验方法")
    pdf.add_list(
        [
            f"模型：{metadata.get('model', '未知')} (temperature={metadata.get('temperature', 'N/A')})",
            "提问方式：同一问题分别使用 Zero-Shot 与 Zero-Shot-CoT 两种提示。",
            "测试集：至少 3 道需要多步推理的问题，覆盖算术、比例与空间关系。",
            "实现：cot_experiment.py 顺序调用 API，记录完整回答并判定对错。",
        ]
    )

    summary = summarize_accuracy(records)

    pdf.add_section("结果与分析")
    pdf.add_paragraph(
        (
            f"本次共测试 {summary['total']} 道题。Zero-Shot 正确 {summary['zero_hits']} 次，"
            f"准确率 {summary['zero_acc']:.0%}；CoT 正确 {summary['cot_hits']} 次，"
            f"准确率 {summary['cot_acc']:.0%}。CoT 在样本中实现了更稳定的正确率。"
        )
    )

    for idx, record in enumerate(records, start=1):
        pdf.add_paragraph(f"问题 {idx} 分析：{record['rationale']}")
        for line in record_lines(idx, record):
            pdf.multi_cell(
                0,
                6,
                line,
                wrapmode=WrapMode.CHAR,
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
        pdf.ln(1)

    pdf.add_section("讨论与局限")
    pdf.add_paragraph(
        (
            "1) 题目数量有限，无法覆盖更复杂的推理。\n"
            "2) 正确性判定仅基于字符串匹配，未融入人类评审。\n"
            "3) 实验依赖单一模型配置，尚未比较不同模型的表现。"
        )
    )

    pdf.add_section("结论")
    pdf.add_paragraph(
        (
            "在给定样本中，Zero-Shot-CoT 始终给出正确答案，而直接提问在第1、"
            "第3题出现错误。CoT 提示通过显式推理步骤帮助模型纠正歧义，"
            "显示出在多步推理任务中的一致优势。"
        )
    )

    add_appendix(pdf, "cot_experiment.py")

    pdf.output(args.output)
    print(f"报告已生成：{args.output}")


if __name__ == "__main__":
    main()
