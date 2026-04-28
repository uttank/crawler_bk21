import os
import time
import random
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse

# ==========================================
# 환경 설정
# ==========================================
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}
# 고정된 통합 파일명 (업데이트 모드에서 활용)
CSV_FILENAME = "bk21_qna_dataset.csv"

# ==========================================
# 네트워크 예외 처리 헬퍼 함수
# ==========================================
def safe_requests_get(url, params=None, headers=None, retries=3, timeout=10):
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"[!] 네트워크 에러 발생 (최종): {e}")
                raise
            print(f"[*] 네트워크 불안정... {attempt + 1}/{retries}회 재시도 중 ({e})")
            time.sleep(2)

# ==========================================
# Phase 1: 목록 수집
# ==========================================
def get_recent_post_ids(limit=None, months=None, update_mode=False, existing_ids=set()):
    post_ids = []
    page_index = 1
    base_url = "https://bk21four.nrf.re.kr/qnaBbs/selectBoardList.do"
    
    cutoff_date = None
    if months is not None and months > 0:
        cutoff_date = datetime.now() - relativedelta(months=months)
        print(f"[*] 데이터 수집 기준일: {cutoff_date.strftime('%Y-%m-%d')} 이후 게시글")
    else:
        print("[*] 데이터 수집 기준일: 전체 기간 (제한 없음)")
        
    while True:
        print(f"[-] 목록 페이지 {page_index} 탐색 중...")
        params = {
            "bbsId": "BBSMSTR_000000000022",
            "pageIndex": page_index
        }
        
        response = safe_requests_get(base_url, params=params, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")
        
        rows = soup.select("table.ctt_table02 tr")
        if len(rows) <= 1: 
            print("[!] 게시글 목록을 찾을 수 없습니다. (마지막 페이지 도달)")
            break
            
        should_stop = False
        
        for row in rows:
            tds = row.find_all("td")
            if not tds or len(tds) < 6:
                continue 
                
            num_text = tds[0].text.strip()
            if num_text == "공지":
                continue 
            
            date_text = tds[5].text.strip()
                
            try:
                post_date = datetime.strptime(date_text, "%Y-%m-%d")
                if cutoff_date and post_date < cutoff_date:
                    print(f"[*] {date_text} 게시글 발견. 지정된 개월({months}개월) 범위를 벗어나 탐색을 중지합니다.")
                    should_stop = True
                    break
            except ValueError:
                pass
                
            title_a = row.select_one("a")
            if title_a and title_a.has_attr("onclick"):
                onclick_text = title_a["onclick"]
                if "fn_egov_inqire_notice" in onclick_text:
                    try:
                        nttId = onclick_text.split("'")[1]
                        
                        # 업데이트 모드: 기존 CSV에 있는 nttId를 만나면 새로 올라온 글 수집이 끝난 것으로 간주
                        if update_mode and str(nttId) in existing_ids:
                            print(f"[*] 기존 데이터베이스에 존재하는 게시물(nttId: {nttId})을 발견하여 조기 종료합니다. (새 글 탐색 완료)")
                            should_stop = True
                            break
                            
                        post_ids.append(nttId)
                        if limit and limit > 0 and len(post_ids) >= limit:
                            print(f"[*] 지정된 수집 제한({limit}개)에 도달하여 조기 종료합니다.")
                            should_stop = True
                            break
                    except IndexError:
                        pass
                
        if should_stop:
            break
            
        page_index += 1
        time.sleep(random.uniform(0.5, 1.5))
        
    return post_ids

# ==========================================
# Phase 2 & 3: 원문 수집 및 직접 파싱 (토큰 소모 X)
# ==========================================
def fetch_and_extract_data_rule_based(nttId):
    url = f"https://bk21four.nrf.re.kr/qnaBbs/selectBoardArticle.do?nttId={nttId}&bbsId=BBSMSTR_000000000022"
    
    try:
        response = safe_requests_get(url, headers=HEADERS)
    except Exception as e:
        print(f"[!] 웹페이지 접근 실패 (nttId: {nttId}): {e}")
        return {}
        
    soup = BeautifulSoup(response.text, "html.parser")
    
    # --- 날짜/작성자 등 메타정보 추출 ---
    date_text = ""
    # 보통 상세페이지 상단에 날짜가 있음
    info_dls = soup.select("div.bbs_info dl dt")
    for dt in info_dls:
        if "작성일" in dt.text:
            dd = dt.find_next_sibling("dd")
            if dd:
                date_text = dd.text.strip()
                
    # --- 질문(Question) 추출 ---
    title_h5 = soup.select_one("h5.bbs_title")
    title_text = title_h5.get_text(strip=True) if title_h5 else ""
    
    q_div = soup.select_one("div.bbs_text")
    q_text = q_div.get_text(separator="\n", strip=True) if q_div else ""
    
    full_question = f"[{title_text}]\n{q_text}".strip()
    
    # --- 답변(Answer) 추출 ---
    answer_text = ""
    answer_table = soup.select_one("table.listTable")
    if answer_table:
        tds = answer_table.find_all("td")
        for td in tds:
            text = td.get_text(separator="\n", strip=True)
            if "댓글이 없습니다" not in text and "로그인하신후" not in text:
                answer_text += text + "\n"
    
    answer_text = answer_text.strip()
    if not answer_text:
        answer_text = "답변이 등록되지 않았습니다."
        
    return {
        "nttId": nttId,
        "Date": date_text,
        "Question": full_question,
        "Answer": answer_text
    }

# ==========================================
# 메인 실행 파이프라인
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="BK21 Q&A Fast Crawler (Rule-based)")
    parser.add_argument("--limit", type=int, default=0, help="수집할 최대 게시글 수 제한 (0: 제한 없음)")
    parser.add_argument("--months", type=int, default=0, help="과거 N개월 데이터만 수집 (0: 전체 기간)")
    parser.add_argument("--update", action="store_true", help="기존 데이터셋(CSV)을 읽어와서 새 글만 추가로 이어쓰기")
    args = parser.parse_args()

    existing_ids = set()
    
    if args.update:
        if os.path.exists(CSV_FILENAME):
            df_existing = pd.read_csv(CSV_FILENAME)
            if "nttId" in df_existing.columns:
                existing_ids = set(df_existing["nttId"].astype(str))
                print(f"[*] 기존 데이터베이스({CSV_FILENAME}) 로드 완료. (총 {len(existing_ids)}건 발견)")
            else:
                print(f"[!] 기존 파일에 nttId 컬럼이 없어 업데이트 모드를 사용할 수 없습니다. 새로 수집합니다.")
        else:
            print(f"[*] 기존 파일({CSV_FILENAME})이 존재하지 않아 새로 전체 수집을 진행합니다.")
    else:
        # 업데이트 모드가 아니면 매번 새로 시작할 때 기존 파일을 덮어쓰기 위해 삭제
        if os.path.exists(CSV_FILENAME):
            os.remove(CSV_FILENAME)

    print("\n=== [1단계] 게시글 ID 목록 수집 시작 ===")
    post_ids = get_recent_post_ids(limit=args.limit, months=args.months, update_mode=args.update, existing_ids=existing_ids)
    
    print(f"[-] 총 {len(post_ids)}개의 새로 수집할 게시글 확인됨.")
    
    if not post_ids:
        print("[✓] 최신 상태입니다. 새로 수집할 게시글이 없습니다. 종료합니다.")
        return

    results_count = 0
    print("\n=== [2&3단계] 각 게시글 직접 파싱 추출 (API X, 실시간 저장) ===")
    start_time = time.time()
    
    for idx, nttId in enumerate(post_ids, 1):
        print(f"[-] 진행 중: {idx}/{len(post_ids)} (nttId: {nttId})")
        
        extracted_data = fetch_and_extract_data_rule_based(nttId)
        if extracted_data.get("Question"):
            results_count += 1
            
            df_row = pd.DataFrame([extracted_data])
            header_flag = not os.path.exists(CSV_FILENAME)
            df_row.to_csv(CSV_FILENAME, mode='a', index=False, header=header_flag, encoding="utf-8-sig")
            
        time.sleep(random.uniform(0.5, 1.5))

    elapsed = time.time() - start_time
    print(f"\n=== [4단계] 데이터 추출 완료 (소요시간: {elapsed:.2f}초) ===")
    if results_count > 0:
        print(f"[v] 총 {results_count}건의 새 데이터가 성공적으로 저장/추가되었습니다: {CSV_FILENAME}")
    else:
        print("[!] 유효하게 추출된 데이터가 없습니다.")

if __name__ == "__main__":
    main()
