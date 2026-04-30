"""
CSV 파일의 LLM 토큰 사용량 추정 도구

한글 CSV 파일을 각 LLM 모델별 토크나이저로 분석하여
예상 토큰 수와 비용을 추정합니다.

사용법:
    python estimate_tokens.py                          # 기본 CSV 파일 분석
    python estimate_tokens.py --file my_data.csv       # 특정 파일 분석
    python estimate_tokens.py --columns Question Answer # 특정 컬럼만 분석
    python estimate_tokens.py --sample 100             # 100행만 샘플링하여 빠른 추정
    python estimate_tokens.py --model gpt-4o           # 특정 모델 기준으로만 분석
"""

import argparse
import io
import os
import sys
import time

# Windows 콘솔 한글 호환을 위해 stdout을 UTF-8로 설정
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import tiktoken
except ImportError:
    print("[!] tiktoken 패키지가 설치되어 있지 않습니다. 설치 중...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken"])
    import tiktoken

import pandas as pd


# ==========================================
# 모델별 토크나이저 및 가격 정보 (2026.04 기준)
# ==========================================
MODEL_INFO = {
    "gpt-4o": {
        "encoding": "o200k_base",
        "input_price_per_1m": 2.50,     # USD per 1M input tokens
        "output_price_per_1m": 10.00,
        "description": "GPT-4o (최신)",
    },
    "gpt-4o-mini": {
        "encoding": "o200k_base",
        "input_price_per_1m": 0.15,
        "output_price_per_1m": 0.60,
        "description": "GPT-4o Mini (경제적)",
    },
    "gpt-4-turbo": {
        "encoding": "cl100k_base",
        "input_price_per_1m": 10.00,
        "output_price_per_1m": 30.00,
        "description": "GPT-4 Turbo",
    },
    "gpt-3.5-turbo": {
        "encoding": "cl100k_base",
        "input_price_per_1m": 0.50,
        "output_price_per_1m": 1.50,
        "description": "GPT-3.5 Turbo",
    },
    "claude-sonnet": {
        "encoding": "cl100k_base",  # 근사치 (Anthropic 자체 토크나이저와 유사)
        "input_price_per_1m": 3.00,
        "output_price_per_1m": 15.00,
        "description": "Claude 3.5 Sonnet (근사 추정)",
    },
    "gemini-2.0-flash": {
        "encoding": "cl100k_base",  # 근사치
        "input_price_per_1m": 0.10,
        "output_price_per_1m": 0.40,
        "description": "Gemini 2.0 Flash (근사 추정)",
    },
}


def count_tokens(text: str, encoding) -> int:
    """텍스트의 토큰 수를 계산합니다."""
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(encoding.encode(text))


def estimate_csv_tokens(
    filepath: str,
    columns: list = None,
    sample_size: int = 0,
    model_filter: str = None,
):
    """
    CSV 파일의 토큰 수를 추정합니다.

    Args:
        filepath: CSV 파일 경로
        columns: 분석할 컬럼 목록 (None이면 전체)
        sample_size: 샘플링할 행 수 (0이면 전체)
        model_filter: 특정 모델만 분석 (None이면 전체 모델)
    """
    # --- 파일 로드 ---
    print(f"\n{'='*60}")
    print(f"  CSV 토큰 사용량 추정기")
    print(f"{'='*60}")

    if not os.path.exists(filepath):
        print(f"\n[X] 파일을 찾을 수 없습니다: {filepath}")
        sys.exit(1)

    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"\n[파일] {filepath} ({file_size_mb:.2f} MB)")

    # 인코딩 자동 탐지 (한글 CSV 호환)
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"   인코딩: {enc}")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        print("[X] CSV 파일 인코딩을 감지할 수 없습니다.")
        sys.exit(1)

    total_rows = len(df)
    print(f"   전체 행 수: {total_rows:,}개")
    print(f"   컬럼 목록: {df.columns.tolist()}")

    # --- 컬럼 필터링 ---
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            print(f"\n[X] 존재하지 않는 컬럼: {missing}")
            print(f"    사용 가능한 컬럼: {df.columns.tolist()}")
            sys.exit(1)
        df_target = df[columns]
        print(f"   분석 대상 컬럼: {columns}")
    else:
        # 텍스트 컬럼만 자동 선택 (숫자, 날짜 등 제외 가능)
        text_cols = df.select_dtypes(include=["object"]).columns.tolist()
        df_target = df[text_cols] if text_cols else df
        print(f"   분석 대상 컬럼 (텍스트): {text_cols if text_cols else df.columns.tolist()}")

    # --- 샘플링 ---
    is_sampled = False
    if sample_size > 0 and sample_size < len(df_target):
        df_sample = df_target.sample(n=sample_size, random_state=42)
        is_sampled = True
        print(f"\n[샘플링] {sample_size:,}행 랜덤 추출 (전체 추정치를 계산합니다)")
    else:
        df_sample = df_target
        print(f"\n[전체] 데이터 분석 모드 ({len(df_sample):,}행)")

    # --- 전체 텍스트 조합 ---
    # 각 행을 하나의 텍스트로 합침 (LLM에 보낼 때의 시나리오)
    texts_per_row = []
    for _, row in df_sample.iterrows():
        row_text = " ".join(str(v) for v in row.values if pd.notna(v))
        texts_per_row.append(row_text)

    # --- 컬럼별 텍스트 길이 통계 ---
    print(f"\n{'-'*60}")
    print(f"  컬럼별 텍스트 길이 통계 (글자 수)")
    print(f"{'-'*60}")
    print(f"  {'컬럼':<20} {'평균':>8} {'중앙값':>8} {'최소':>8} {'최대':>8}")
    print(f"  {'-'*52}")

    for col in df_target.columns:
        col_data = df_sample[col].astype(str)
        lengths = col_data.str.len()
        print(
            f"  {col:<20} {lengths.mean():>8.0f} {lengths.median():>8.0f} "
            f"{lengths.min():>8.0f} {lengths.max():>8.0f}"
        )

    # --- 모델별 토큰 수 추정 ---
    models_to_analyze = MODEL_INFO.items()
    if model_filter:
        if model_filter not in MODEL_INFO:
            print(f"\n[X] 알 수 없는 모델: {model_filter}")
            print(f"    지원 모델: {list(MODEL_INFO.keys())}")
            sys.exit(1)
        models_to_analyze = [(model_filter, MODEL_INFO[model_filter])]

    results = {}

    for model_name, info in models_to_analyze:
        encoding = tiktoken.get_encoding(info["encoding"])
        start = time.time()

        # 행별 토큰 수 계산
        token_counts = [count_tokens(text, encoding) for text in texts_per_row]

        elapsed = time.time() - start

        sample_total = sum(token_counts)
        avg_per_row = sample_total / len(token_counts) if token_counts else 0

        # 샘플링인 경우 전체 추정
        if is_sampled:
            estimated_total = int(avg_per_row * total_rows)
        else:
            estimated_total = sample_total

        results[model_name] = {
            "encoding": info["encoding"],
            "description": info["description"],
            "sample_tokens": sample_total,
            "estimated_total": estimated_total,
            "avg_per_row": avg_per_row,
            "min_per_row": min(token_counts) if token_counts else 0,
            "max_per_row": max(token_counts) if token_counts else 0,
            "input_price_per_1m": info["input_price_per_1m"],
            "output_price_per_1m": info["output_price_per_1m"],
            "elapsed": elapsed,
        }

    # --- 결과 출력 ---
    print(f"\n{'='*60}")
    print(f"  모델별 토큰 수 추정 결과")
    if is_sampled:
        print(f"  (샘플 {sample_size:,}행 기반 -> 전체 {total_rows:,}행 추정)")
    print(f"{'='*60}")

    for model_name, r in results.items():
        print(f"\n  [*] {model_name} ({r['description']})")
        print(f"      토크나이저: {r['encoding']}")
        print(f"      행당 평균 토큰: {r['avg_per_row']:,.0f} tokens")
        print(f"      행당 최소/최대: {r['min_per_row']:,} / {r['max_per_row']:,} tokens")
        print(f"      {'추정 ' if is_sampled else ''}전체 토큰 수: {r['estimated_total']:,} tokens")
        print(f"      분석 소요 시간: {r['elapsed']:.2f}초")

    # --- 비용 추정 ---
    print(f"\n{'='*60}")
    print(f"  예상 비용 추정 (입력 토큰 기준)")
    print(f"{'='*60}")
    print(f"\n  {'모델':<20} {'토큰 수':>15} {'입력비용(USD)':>14} {'입력비용(KRW)':>14}")
    print(f"  {'-'*63}")

    krw_rate = 1380  # 대략적인 환율

    for model_name, r in results.items():
        total = r["estimated_total"]
        input_cost = (total / 1_000_000) * r["input_price_per_1m"]
        input_cost_krw = input_cost * krw_rate

        print(
            f"  {model_name:<20} {total:>13,}  ${input_cost:>11.4f}  W{input_cost_krw:>11,.0f}"
        )

    # --- 시나리오별 비용 (전체 데이터를 1회 처리한다고 가정) ---
    print(f"\n{'-'*60}")
    print(f"  시나리오별 비용 추정 (전체 {total_rows:,}행 1회 처리)")
    print(f"{'-'*60}")
    print(f"  * 출력 토큰은 입력의 50%로 가정 (질문-답변 요약 등)")
    print(f"  * 환율: 1 USD = {krw_rate:,} KRW\n")

    print(f"  {'모델':<20} {'입력비용':>10} {'출력비용':>10} {'합계(USD)':>10} {'합계(KRW)':>12}")
    print(f"  {'-'*63}")

    output_ratio = 0.5  # 출력 토큰 = 입력의 50% 가정

    for model_name, r in results.items():
        total = r["estimated_total"]
        input_cost = (total / 1_000_000) * r["input_price_per_1m"]
        output_tokens = int(total * output_ratio)
        output_cost = (output_tokens / 1_000_000) * r["output_price_per_1m"]
        total_cost = input_cost + output_cost
        total_cost_krw = total_cost * krw_rate

        print(
            f"  {model_name:<20} ${input_cost:>8.4f} ${output_cost:>8.4f} "
            f"${total_cost:>8.4f} W{total_cost_krw:>10,.0f}"
        )

    # --- 한글 토큰 효율 분석 ---
    print(f"\n{'='*60}")
    print(f"  한글 토큰 효율 분석")
    print(f"{'='*60}")

    # 전체 텍스트의 글자 수
    total_chars = sum(len(t) for t in texts_per_row)
    if is_sampled:
        total_chars_estimated = int((total_chars / sample_size) * total_rows)
    else:
        total_chars_estimated = total_chars

    for model_name, r in results.items():
        chars_per_token = total_chars / r["sample_tokens"] if r["sample_tokens"] > 0 else 0
        print(f"\n  [*] {model_name}")
        print(f"      글자/토큰 비율: {chars_per_token:.2f} 글자 = 1 토큰")
        print(f"      (참고: 영어는 약 4글자 = 1토큰, 한글은 약 1.5~2글자 = 1토큰)")

    print(f"\n{'='*60}")
    print(f"  [v] 분석 완료")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="한글 CSV 파일의 LLM 토큰 사용량 추정 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python estimate_tokens.py                              # 기본 파일 분석
  python estimate_tokens.py --file data.csv              # 특정 파일
  python estimate_tokens.py --columns Question Answer    # 특정 컬럼만
  python estimate_tokens.py --sample 500                 # 500행 샘플링
  python estimate_tokens.py --model gpt-4o-mini          # 특정 모델만
        """,
    )
    parser.add_argument(
        "--file", type=str, default="data/bk21_qna_dataset.csv",
        help="분석할 CSV 파일 경로 (기본: data/bk21_qna_dataset.csv)",
    )
    parser.add_argument(
        "--columns", nargs="+", type=str, default=None,
        help="분석할 컬럼 목록 (예: --columns Question Answer)",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="샘플링할 행 수 (0: 전체 분석, 빠른 추정 시 100~500 권장)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(MODEL_INFO.keys()),
        help="특정 모델만 분석",
    )

    args = parser.parse_args()

    estimate_csv_tokens(
        filepath=args.file,
        columns=args.columns,
        sample_size=args.sample,
        model_filter=args.model,
    )


if __name__ == "__main__":
    main()
