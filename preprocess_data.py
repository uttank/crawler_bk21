import pandas as pd
import os
import re

DATA_DIR = "data"
INPUT_FILE = f"{DATA_DIR}/bk21_qna_dataset.csv"
OUTPUT_FILE = f"{DATA_DIR}/bk21_qna_cleaned.csv"

# 줄 전체가 인사말/보일러플레이트일 때만 제거하는 패턴.
# 본문 중간의 같은 단어(예: "...관련하여 문의드립니다.")는 절대 건드리지 않는다.
GREETING_LINE_PATTERNS = [
    r'^\(?수정\)?\s*안녕(?:하세요|하십니까)[\.,\s]*BK21\s*사업팀입니다[\.\,\!\?\s]*$',
    r'^\(?수정\)?\s*안녕(?:하세요|하십니까)[\.\,\!\?\s]*$',
    r'^\(?수정\)?\s*BK21\s*사업팀입니다[\.\,\!\?\s]*$',
    r'^감사(?:합니다|드립니다)[\.\,\!\?\s]*$',
    r'^고맙습니다[\.\,\!\?\s]*$',
    r'^문의드립니다[\.\,\!\?\s]*$',
    r'^문의\s*부탁드립니다[\.\,\!\?\s]*$',
    r'^질문(?:드립니다|있습니다)[\.\,\!\?\s]*$',
    r'^수고(?:하십니다|하십니까|많으십니다)[\.\,\!\?\s]*$',
    r'^늘\s*수고가\s*많으십니다[\.\,\!\?\s]*$',
    r'^답변에?\s*미리\s*감사드립니다[\.\,\!\?\s]*$',
    r'^답변(?:을)?\s*기다리겠습니다[\.\,\!\?\s]*$',
    r'^답변\s*부탁드립니다[\.\,\!\?\s]*$',
    r'^답변에?\s*감사드립니다[\.\,\!\?\s]*$',
    r'^문의주신\s*사항에\s*대해\s*다음과\s*같이\s*안내\s*드립니다[\.\,\!\?\s]*$',
]
GREETING_RE = re.compile('|'.join(GREETING_LINE_PATTERNS), flags=re.IGNORECASE)

# 단락 시작에서 prefix로 떨어내야 할 보일러플레이트 (한 줄에 본문과 함께 붙어있을 때)
PARAGRAPH_PREFIX_PATTERNS = [
    # "(수정) 안녕하세요, BK21사업팀입니다." + "문의주신 사항에 대해 다음과 같이 안내 드립니다."
    r'(?:\(수정\)\s*)?안녕(?:하세요|하십니까)[\.\,\s]*BK21\s*사업팀입니다[\.\,\s]*',
    r'(?:\(수정\)\s*)?BK21\s*사업팀입니다[\.\,\s]*',
    r'(?:\(수정\)\s*)?안녕(?:하세요|하십니까)[\.\,\s]+(?=\S)',
    r'(?:\(수정\)\s*)?문의주신\s*사항에\s*대해\s*다음과\s*같이\s*안내\s*드립니다[\.\,\s]*',
]
PARAGRAPH_PREFIX_RES = [
    re.compile(r'(^|\n)[ \t]*' + pat, flags=re.IGNORECASE)
    for pat in PARAGRAPH_PREFIX_PATTERNS
]

# 작성자 + 타임스탬프 블록 (BKS관리자 외 일반 이름 답글까지 포함)
AUTHOR_TIMESTAMP_RE = re.compile(
    r'(?:^|\n)[ \t]*[A-Za-z가-힣][A-Za-z가-힣0-9 ]{1,14}\s*\r?\n\s*'
    r'\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}\s*\r?\n?',
)

# 끝에 붙는 맺음말
TRAILING_RE = re.compile(
    r'(?:\s*(?:감사합니다|감사드립니다|고맙습니다|이상입니다))[\.\,\!\?\s]*$',
    flags=re.IGNORECASE,
)


def remove_greetings(text):
    if not isinstance(text, str):
        return ""

    # 1. NBSP / zero-width 공백 정규화
    text = text.replace('\xa0', ' ').replace('​', '')

    # 2. 작성자+타임스탬프 블록 제거 (다단 댓글에도 적용)
    text = AUTHOR_TIMESTAMP_RE.sub('\n', text)

    # 3. 단락 시작 prefix(인사말 + BK21사업팀입니다 + 문의주신 사항...) 제거
    #    여러 번 반복해서 chained prefix를 한 번에 떨어냄
    for _ in range(3):
        before = text
        for rx in PARAGRAPH_PREFIX_RES:
            text = rx.sub(r'\1', text)
        if text == before:
            break

    # 4. 줄 단위 인사말 라인 제거
    out_lines = []
    for line in re.split(r'\r?\n', text):
        stripped = line.strip()
        if not stripped:
            out_lines.append('')
            continue
        if GREETING_RE.match(stripped):
            continue
        out_lines.append(line)
    text = '\n'.join(out_lines)

    # 5. 끝맺음말
    text = TRAILING_RE.sub('', text)

    # 6. 줄바꿈 정규화
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def preprocess_data():
    print(f"=== BK21 Q&A 데이터 전처리 시작 ===")
    
    if not os.path.exists(INPUT_FILE):
        print(f"[!] 원본 파일을 찾을 수 없습니다: {INPUT_FILE}")
        return

    # 1. 데이터 로드
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    initial_count = len(df)
    print(f"[-] 원본 데이터 로드: {initial_count:,}건")

    # 2. 미답변 제거
    # '답변이 등록되지 않았습니다' 텍스트가 포함된 행 제거
    df_cleaned = df[~df['Answer'].astype(str).str.contains('답변이 등록되지', na=False)].copy()
    
    # 3. 결측치 처리
    # Date 결측치는 '알 수 없음'으로 변경
    df_cleaned['Date'] = df_cleaned['Date'].fillna('알 수 없음')
    
    # Question, Answer 결측치 빈 문자열로 변경
    df_cleaned['Question'] = df_cleaned['Question'].fillna('')
    df_cleaned['Answer'] = df_cleaned['Answer'].fillna('')

    # 4. 답변 메타데이터(날짜) 분리 추출
    # "BKS관리자\n2026-04-27 16:22:55" 패턴에서 날짜 부분만 캡처하여 새 컬럼 생성
    df_cleaned['Answer_Date'] = df_cleaned['Answer'].str.extract(r'BKS[가-힣a-zA-Z]*\s*\r?\n\s*(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})')
    # 결측치(매칭 안된 경우)는 빈 문자열로 처리
    df_cleaned['Answer_Date'] = df_cleaned['Answer_Date'].fillna('')

    # 5. 인사말 및 특수문자 정제 (위에서 추출했으므로 텍스트에서는 지움)
    df_cleaned['Question'] = df_cleaned['Question'].apply(remove_greetings)
    df_cleaned['Answer'] = df_cleaned['Answer'].apply(remove_greetings)

    # 6. 빈 텍스트가 된 행 처리
    # 인사말만 있어서 빈칸이 된 질문/답변 필터링 (보수적으로 10자 이상만 남김)
    df_cleaned = df_cleaned[(df_cleaned['Question'].str.len() > 5) & (df_cleaned['Answer'].str.len() > 5)]

    # 6. 임베딩용 통합 텍스트 생성 (Question + Answer)
    # RAG 검색 시 질문과 답변 내용이 모두 포함되어야 검색 정확도가 올라갑니다.
    df_cleaned['Combined_Text'] = df_cleaned.apply(
        lambda row: f"[질문]\n{row['Question']}\n\n[답변]\n{row['Answer']}", axis=1
    )

    # 7. 저장
    df_cleaned.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    
    final_count = len(df_cleaned)
    removed_count = initial_count - final_count
    
    print(f"[-] 정제 완료: {final_count:,}건 남음 (제거됨: {removed_count:,}건)")
    print(f"[-] 결과 파일 저장됨: {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess_data()
