import pandas as pd
import os

INPUT_FILE = "bk21_qna_dataset.csv"
OUTPUT_FILE = "bk21_qna_cleaned.csv"

import re

def remove_greetings(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Answer에 항상 붙는 작성자 및 날짜 패턴 제거
    # 예: "BKS관리자\n2026-04-27 16:22:55"
    text = re.sub(r'BKS[가-힣a-zA-Z]*\s*\n\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}', '', text)
    
    # 2. 일반적인 인사말/맺음말 제거
    greetings = [
        r'안녕[하십니까세요]+[\.\,\!\?\s]*',
        r'수고[하십니까많으십니다]+[\.\,\!\?\s]*',
        r'감사[합니다]+[\.\,\!\?\s]*',
        r'고맙[습니다]+[\.\,\!\?\s]*',
        r'문의[드립니다]+[\.\,\!\?\s]*',
        r'질문[드립니다]+[\.\,\!\?\s]*',
        r'BK21[ ]*사업팀입니다[\.\,\!\?\s]*',
        r'늘 수고가 많으십니다[\.\,\!\?\s]*',
    ]
    
    for pattern in greetings:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
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
    df_cleaned['Answer_Date'] = df_cleaned['Answer'].str.extract(r'BKS[가-힣a-zA-Z]*\s*\n(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})')
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
