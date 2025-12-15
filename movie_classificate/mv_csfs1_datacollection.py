import csv
import time
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import TimeoutException, NoSuchElementException

def setup_driver():
    """Chrome 드라이버 설정"""
    chrome_options = Options() # 브라우저 옵션 설정
    # chrome_options.add_argument('--headless')  # 브라우저 창을 숨기려면 주석 해제
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    
    # 추가: Chrome 내부 오류 메시지 숨기기
    chrome_options.add_argument('--disable-logging')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-background-networking')
    chrome_options.add_argument('--disable-sync')
    chrome_options.add_argument('--disable-background-timer-throttling')
    chrome_options.add_argument('--disable-backgrounding-occluded-windows')
    chrome_options.add_argument('--disable-renderer-backgrounding')
    chrome_options.add_argument('--disable-features=TranslateUI')
    chrome_options.add_argument('--disable-ipc-flooding-protection')
    chrome_options.add_argument('--log-level=3')
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    try:
        driver = webdriver.Chrome(options=chrome_options) # ChromeDriver 경로가 시스템 PATH에 있을 경우
        return driver ##3. 드라이버 객체 반환 
    except Exception as e:
        print(f"❌ Chrome 드라이버 설정 실패: {e}")
        print("Chrome 드라이버가 설치되어 있는지 확인해주세요.")
        return None

def set_sort_order_to_release_date(driver):
    """정렬 순서를 개봉일순으로 변경"""
    try:
        print("🔧 정렬 순서를 개봉일순으로 변경 중...")
        
        # 정렬 드롭다운 찾기
        sort_dropdown = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "sOrderBy"))
        ) # html select 태그 찾기
        
        # Select 객체 생성
        select = Select(sort_dropdown)
        
        # 현재 선택된 옵션 확인
        current_option = select.first_selected_option.text
        print(f"  현재 정렬 옵션: {current_option}")
        
        # 개봉일순(value="4")으로 변경
        select.select_by_value("4")
        
        # 변경 후 확인
        new_option = select.first_selected_option.text
        print(f"  변경된 정렬 옵션: {new_option}")
        
        # 페이지 새로고침 대기 (정렬 변경 후 자동으로 새로고침됨)
        time.sleep(3)
        
        # 페이지 로드 완료 대기
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "tbl_comm"))
        )
        
        print("✅ 개봉일순으로 정렬 변경 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 정렬 순서 변경 실패: {e}")
        return False

def navigate_to_page(driver, target_page, current_page=1):
    """특정 페이지로 이동하는 함수"""
    try:
        if target_page == current_page:
            return True
            
        print(f"📄 {current_page}페이지에서 {target_page}페이지로 이동 중...")
        
        # 현재 페이지 그룹 (1-10, 11-20, 21-30, ...)
        current_group_start = ((current_page - 1) // 10) * 10 + 1
        current_group_end = current_group_start + 9
        
        # 타겟 페이지 그룹
        target_group_start = ((target_page - 1) // 10) * 10 + 1
        target_group_end = target_group_start + 9
        
        print(f"  현재 페이지 그룹: {current_group_start}-{current_group_end}")
        print(f"  타겟 페이지 그룹: {target_group_start}-{target_group_end}")
        
        # 같은 그룹 내에서 이동
        if current_group_start == target_group_start:
            print(f"  같은 그룹 내 이동: {target_page}번 페이지 버튼 클릭")
            try:
                page_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, f"//a[@onclick=\"goPage('{target_page}')\"]"))
                )
                driver.execute_script("arguments[0].click();", page_button)
                time.sleep(3)
                return True
            except TimeoutException:
                print(f"  ❌ {target_page}페이지 버튼을 찾을 수 없습니다.")
                return False
        
        # 다른 그룹으로 이동해야 하는 경우
        else:
            # 타겟 그룹까지 "다음" 버튼으로 이동
            steps_needed = (target_group_start - current_group_start) // 10
            print(f"  다른 그룹으로 이동: {steps_needed}번의 '다음' 버튼 클릭 필요")
            
            for step in range(steps_needed):
                try:
                    # "다음" 버튼 찾기 및 클릭
                    next_button = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "a.btn.next"))
                    )
                    # ***복잡한 조건이 필요할 때 css / 단순한 태그를 찾을 때는 BY.TAG_NAME 사용 ***
                    
                    # 다음 그룹의 첫 페이지 번호 계산
                    next_group_start = current_group_start + 10
                    print(f"    {step + 1}단계: '다음' 버튼 클릭 (→ {next_group_start}페이지 그룹)")
                    
                    driver.execute_script("arguments[0].click();", next_button)
                    time.sleep(3)

                    # 페이지 로드 완료 대기
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "tbl_comm"))
                    )

                    current_group_start = next_group_start
                    
                except TimeoutException:
                    print(f"    ❌ {step + 1}단계에서 '다음' 버튼을 찾을 수 없습니다.")
                    return False
            
            # 타겟 그룹에 도달했으면 특정 페이지로 이동
            if target_page % 10 != 1:  # 그룹의 첫 페이지가 아닌 경우
                try:
                    page_button = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, f"//a[@onclick=\"goPage('{target_page}')\"]"))
                    )
                    print(f"  최종: {target_page}번 페이지 버튼 클릭")
                    driver.execute_script("arguments[0].click();", page_button)
                    time.sleep(3)
                except TimeoutException:
                    print(f"  ❌ {target_page}페이지 버튼을 찾을 수 없습니다.")
                    return False
            
            return True
            
    except Exception as e:
        print(f"❌ 페이지 이동 실패: {e}")
        return False

def clean_synopsis_text(synopsis):
    """줄거리 텍스트를 CSV에 적합하게 정리"""
    if not synopsis:
        return "줄거리 정보 없음"
    
    # 줄바꿈을 공백으로 변환
    cleaned = synopsis.replace('\n', ' ').replace('\r', ' ')
    
    # 연속된 공백을 단일 공백으로 변환
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # 앞뒤 공백 제거
    cleaned = cleaned.strip()
    
    return cleaned

# extract_movie_data_with_synopsis(driver, page, current_page) # 함수 호출 위치에 추가
def extract_movie_data_with_synopsis(driver, page_num=1, current_page=1): ##5 드라이버, 페이지 번호, 현재 페이지 번호를 인자로 받는 함수
    """영화 제목, 장르, 제작연도, 줄거리를 모두 추출하는 함수"""
    base_url = "https://www.kobis.or.kr/kobis/business/mast/mvie/searchMovieList.do"
    
    try:
        # 첫 페이지인 경우 사이트에 접속하고 정렬 순서 변경
        if page_num == 1:
            print(f"🌐 KOBIS 사이트에 접속 중...")
            driver.get(base_url) ##6. 사이트 접속
            
            # 페이지 로드 대기, 조건부대기 15초까지기다린다. 15초가 지남녀 오류발생
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "tbl_comm"))
            ) ##7. tbl_comm요소가 페이지에 나타날때까지 대기
            # 웹페이지가 완전히 로드 되기를 기다리기 위해
            
            ##8. 정렬 순서를 개봉일순으로 변경
            if not set_sort_order_to_release_date(driver):
                print("⚠️ 정렬 순서 변경에 실패했지만 크롤링을 계속합니다.")
                
            current_page = 1
        else:
            # 다른 페이지로 이동
            if not navigate_to_page(driver, page_num, current_page):
                print(f"❌ {page_num}페이지로 이동할 수 없습니다.")
                return [], current_page
        
        # 페이지 로드 완료 대기
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "tbl_comm"))
        )
        
        print(f"🎬 {page_num}페이지 로딩 완료")

        ##9. 영화 행 찾기 <tr> 요소들
        movie_rows = driver.find_elements(By.CSS_SELECTOR, "table.tbl_comm tbody tr")
        # print(movie_rows)
        print(f"📊 {page_num}페이지에서 {len(movie_rows)}개의 영화 행을 발견했습니다.")
        
        movie_data = []
        
        for i, row in enumerate(movie_rows, 1): ##10. 각 영화 행에 대해 반복
            try:
                print(f"  🎭 {i}번째 영화 처리 중...")
                
                # 기본 정보 추출
                cells = row.find_elements(By.TAG_NAME, "td")
                ##11. cells은 <td> 요소들의 리스트
                if len(cells) < 7: # 최소한의 셀 개수 확인
                    continue
                
                ##11. 한글 제목 추출
                # print(cells)
                korean_title_cell = cells[0]
                korean_title_link = korean_title_cell.find_element(By.CSS_SELECTOR, "a[onclick*='mstView']")
                korean_title = korean_title_link.get_attribute("title") or korean_title_link.text.strip()

                ##12. 영어 제목 추출
                english_title = ""
                if len(cells) > 1:
                    english_title_cell = cells[1]
                    try:
                        english_title_link = english_title_cell.find_element(By.CSS_SELECTOR, "a[onclick*='mstView']")
                        english_title = english_title_link.get_attribute("title") or english_title_link.text.strip()
                    except NoSuchElementException:
                        pass

                ##13. 제작연도 추출
                production_year = ""
                if len(cells) > 3:
                    year_span = cells[3].find_element(By.TAG_NAME, "span")
                    production_year = year_span.get_attribute("title") or year_span.text.strip()
                
                ##14. 장르 추출
                genre = ""
                if len(cells) > 6:
                    genre_span = cells[6].find_element(By.TAG_NAME, "span")
                    genre = genre_span.get_attribute("title") or genre_span.text.strip()

                ##15. 우선순위에 따른 제목 선택
                if english_title and korean_title:
                    selected_title = english_title
                    display_info = f"{english_title} (영어)"
                elif korean_title:
                    selected_title = korean_title
                    display_info = f"{korean_title} (한글)"
                elif english_title:
                    selected_title = english_title
                    display_info = f"{english_title} (영어)"
                else:
                    continue
                
                print(f"    📝 제목: {display_info}")
                print(f"    🎬 장르: {genre}")
                print(f"    📅 제작연도: {production_year}")
                
                # 영화 제목 클릭하여 상세 정보 가져오기
                print(f"    🔍 '{selected_title}' 줄거리 수집 중...")
                
                ##16. 스크롤해서 요소가 보이도록 함
                driver.execute_script("arguments[0].scrollIntoView(true);", korean_title_link)
                time.sleep(1)
                
                ##17. JavaScript 클릭
                driver.execute_script("arguments[0].click();", korean_title_link)
                
                # 상세 정보 다이얼로그 대기
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".ui-dialog .layer"))
                    )
                    time.sleep(2)  # 추가 로딩 대기
                    
                    # 줄거리 추출
                    synopsis = ""
                    try:
                        synopsis_element = driver.find_element(By.CSS_SELECTOR, ".info.info2 .desc_info") ##18. 줄거리 요소찾기
                        synopsis = synopsis_element.text.strip() ##19. 줄거리 원본 텍스트 추출
                        synopsis = clean_synopsis_text(synopsis)  ##20. CSV 저장에 적합하도록 텍스트 정리
                        print(f"    📖 줄거리: {synopsis[:50]}..." if len(synopsis) > 50 else f"    📖 줄거리: {synopsis}")
                    except NoSuchElementException:
                        synopsis = "줄거리 정보 없음"
                        print(f"    📖 줄거리: 정보 없음")

                    ##21. 닫기 버튼 클릭
                    try:
                        close_button = WebDriverWait(driver, 5).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, "a.close[onclick*='dtlRmAll']"))
                        )
                        driver.execute_script("arguments[0].click();", close_button)
                        time.sleep(1)  # 닫기 후 대기
                    except TimeoutException:
                        print("    ⚠️ 닫기 버튼을 찾을 수 없습니다.")
                        # ESC 키로 닫기 시도
                        driver.execute_script("if(typeof dtlRmAll === 'function') dtlRmAll();")
                        time.sleep(1)
                
                except TimeoutException:
                    print(f"    ❌ '{selected_title}' 상세 정보 로딩 실패")
                    synopsis = "줄거리 로딩 실패"
                
                ##22. 데이터 저장 (번호는 나중에 추가됨)
                movie_info = {
                    '영화제목': selected_title,
                    '장르': genre if genre else "장르 정보 없음",
                    '제작연도': production_year if production_year else "제작연도 정보 없음",
                    '줄거리': synopsis
                }
                movie_data.append(movie_info)
                ##23. 처리 완료 메시지
                print(f"    ✅ '{selected_title}' 처리 완료")
                
                # 각 영화 처리 간 대기
                time.sleep(2)
                
            except Exception as e:
                print(f"    ❌ {i}번째 영화 처리 중 오류: {e}")
                continue
        
        print(f"📊 {page_num}페이지에서 총 {len(movie_data)}개의 영화 정보를 수집했습니다.")
        return movie_data, page_num ##24. 수집된 영화 데이터와 현재 페이지 번호 반환
        
    except Exception as e:
        print(f"❌ {page_num}페이지 처리 중 오류: {e}")
        return [], current_page

def save_to_csv(movies_data, filename='kobis_movies_with_synopsis.csv'):
    """영화 데이터를 CSV 파일로 저장 (번호 포함, 올바른 쌍따옴표 처리)"""
    if not movies_data:
        print("저장할 데이터가 없습니다.")
        return False
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            # 번호 칼럼을 맨 앞에 추가
            fieldnames = ['번호', '영화제목', '장르', '제작연도', '줄거리']
            
            # quoting=csv.QUOTE_ALL로 모든 필드를 쌍따옴표로 감싸기
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            
            writer.writeheader()
            
            # 각 영화에 번호를 부여하여 저장
            for index, movie in enumerate(movies_data, 1):
                row_data = {
                    '번호': str(index),  # 문자열로 변환
                    '영화제목': movie.get('영화제목', ''),
                    '장르': movie.get('장르', ''),
                    '제작연도': movie.get('제작연도', ''),
                    '줄거리': movie.get('줄거리', '')
                }
                writer.writerow(row_data)
        
        print(f"✅ 총 {len(movies_data)}개의 영화 정보가 {filename}에 저장되었습니다.")
        print(f"📋 번호는 1번부터 {len(movies_data)}번까지 자동으로 부여되었습니다.")
        print(f"🔤 모든 필드가 쌍따옴표로 감싸져 CSV 형식이 올바르게 처리됩니다.")
        return True
        
    except Exception as e:
        print(f"❌ CSV 파일 저장 실패: {e}")
        return False

def crawl_movies_with_selenium(max_pages=3): 
    """Selenium을 사용한 영화 정보 크롤링"""
    print(f"🚀 KOBIS 영화 정보 크롤링 시작 (개봉일순, 최대 {max_pages}페이지)")
    print("=" * 80)
    
    driver = setup_driver() ##2.드라이버 설정 함수 호출
    if not driver:
        return []
    
    try:
        all_movie_data = [] # 모든 영화 데이터를 저장할 리스트
        total_movie_count = 0  # 전체 영화 개수 카운터
        current_page = 1  # 현재 페이지 추적
        
        for page in range(1, max_pages + 1):
            print(f"\n📄 {page}페이지 처리 중...")
            
            movie_data, current_page = extract_movie_data_with_synopsis(driver, page, current_page) ##4. 영화 데이터 추출 함수 호출
            all_movie_data.extend(movie_data) ##25. 전체 데이터에 추가
            # print("영화전체데이터 : ",all_movie_data)
            total_movie_count += len(movie_data) ##26. 전체 영화 개수 업데이트
            # print("영화 전체 갯수 : ",total_movie_count)
            
            print(f"📈 현재까지 총 {total_movie_count}개의 영화 정보가 수집되었습니다.")

            ##27. 페이지 간 대기
            if page < max_pages:
                print(f"⏳ 다음 페이지 처리 준비...")
                time.sleep(2)
        
        return all_movie_data
        
    except Exception as e:
        print(f"❌ 크롤링 중 오류 발생: {e}")
        return []
    
    finally:
        try:
            driver.quit()
            print("🔒 브라우저를 종료했습니다.")
        except:
            pass

def main():
    print("KOBIS 영화 정보 크롤링 (개봉일순, 줄거리 포함)")
    print("-" * 60)
    
    # 사용자 입력
    try:
        max_pages = int(input("수집할 페이지 수를 입력하세요 (기본값: 3): ") or "3")
        if max_pages > 20: # max_pages 입력 받고 20 이상이면 경고 메시지 출력
            confirm = input(f"{max_pages}페이지는 시간이 매우 오래 걸릴 수 있습니다. 계속하시겠습니까? (y/n): ")
            if confirm.lower() != 'y': # y가 아니면 기본값 3페이지로 설정
                max_pages = 3
                print("기본값 3페이지로 설정됩니다.")
    except ValueError:
        max_pages = 3
        print("기본값 3페이지로 설정됩니다.")
    
    # 영화 데이터 수집
    all_movie_data = crawl_movies_with_selenium(max_pages) ##28. 크롤링 함수의 반환값 저장
    
    # 결과 저장 및 출력
    if all_movie_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'kobis_movies_pages_{timestamp}.csv'
        
        save_to_csv(all_movie_data, filename) ##29. 저장 함수 호출
        
        # 결과 요약
        print("\n" + "=" * 80)
        print("📋 크롤링 완료 요약 (개봉일순, 페이지 네비게이션 지원):")
        print("=" * 80)
        print(f"✅ 총 {len(all_movie_data)}개의 영화 정보를 성공적으로 수집했습니다!")
        print(f"📁 파일명: {filename}")
        print(f"🔢 번호: 1번부터 {len(all_movie_data)}번까지 자동 부여")
        
        # 연도별 통계
        year_count = {}
        for movie in all_movie_data:
            year = movie['제작연도']
            if year in year_count:
                year_count[year] += 1
            else:
                year_count[year] = 1
        
        print(f"\n📊 제작연도별 영화 수:")
        sorted_years = sorted(year_count.items(), key=lambda x: x[0], reverse=True)
        for year, count in sorted_years[:10]:
            print(f"   {year}: {count}편")
        
        # 샘플 데이터 출력 (번호 포함)
        print(f"\n📋 수집된 영화 정보 샘플:")
        for i, movie in enumerate(all_movie_data[:3], 1):
            print(f"{i}. {movie['영화제목']}")
            print(f"   장르: {movie['장르']}")
            print(f"   제작연도: {movie['제작연도']}")
            print(f"   줄거리: {movie['줄거리'][:100]}..." if len(movie['줄거리']) > 100 else f"   줄거리: {movie['줄거리']}")
            print()
            
    else:
        print("❌ 수집된 데이터가 없습니다.")
        print("Chrome 드라이버가 설치되어 있는지 확인해주세요.")

if __name__ == "__main__":
    main() ##1. main() 함수 호출
