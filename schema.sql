CREATE TABLE IF NOT EXISTS 'joint-table' 
(
    # Label
    `di_type` int(11) NOT NULL COMMENT '사고여부 (10: 사고, 20: 비사고)',     
    `di_case` text NOT NULL COMMENT '사고 형태',
    `di_case_code` varchar(10) NOT NULL COMMENT '유해상황 코드',
    `di_case_detail` varchar(50) NOT NULL COMMENT '유해상황 상세',


    # X
    `sch_code` int(11) NOT NULL AUTO_INCREMENT COMMENT '점검내역 고유번호',

    # ?
    `c_code` int(11) NOT NULL COMMENT '업체 고유번호 (작성자)',
    `e_code` int(11) NOT NULL COMMENT '직원 고유번호 (작성자)',
    `sch_date` datetime NOT NULL COMMENT '연월일',

    # 1 (0, 2, 0)
    `sch_time_start` text NOT NULL COMMENT '점검시간 (시작시간)',
    `sch_time_end` text NOT NULL COMMENT '점검시간 (종료시간)',

    # 2 (0, 12, 12)
    `sch_01` int(11) NOT NULL COMMENT '1번 선택지',
    `sch_01_note` text NOT NULL COMMENT '1번 비고',
    `sch_02` int(11) NOT NULL COMMENT '2번 선택지',
    `sch_02_note` text NOT NULL COMMENT '2번 비고',
    `sch_03` int(11) NOT NULL COMMENT '3번 선택지',
    `sch_03_note` text NOT NULL COMMENT '3번 비고',
    `sch_04` int(11) NOT NULL COMMENT '4번 선택지',
    `sch_04_note` text NOT NULL COMMENT '4번 비고',
    `sch_05` int(11) NOT NULL COMMENT '5번 선택지',
    `sch_05_note` text NOT NULL COMMENT '5번 비고',
    `sch_06` int(11) NOT NULL COMMENT '6번 선택지',
    `sch_06_note` text NOT NULL COMMENT '6번 비고',
    `sch_07` int(11) NOT NULL COMMENT '7번 선택지',
    `sch_07_note` text NOT NULL COMMENT '7번 비고',
    `sch_08` int(11) NOT NULL COMMENT '8번 선택지',
    `sch_08_note` text NOT NULL COMMENT '8번 비고',
    `sch_09` int(11) NOT NULL COMMENT '9번 선택지',
    `sch_09_note` text NOT NULL COMMENT '9번 비고',
    `sch_10` int(11) NOT NULL COMMENT '10번 선택지',
    `sch_10_note` text NOT NULL COMMENT '10번 비고',
    `sch_11` int(11) NOT NULL COMMENT '11번 선택지',
    `sch_11_note` text NOT NULL COMMENT '11번 비고',
    `sch_12` int(11) NOT NULL COMMENT '12번 선택지',
    `sch_12_note` text NOT NULL COMMENT '12번 비고',
    
    # X
    `sch_insertdate` datetime NOT NULL COMMENT '관리대장 등록일',


    # X
    `ci_code` int(11) NOT NULL AUTO_INCREMENT COMMENT '화학물질 관리대장 고유번호',
    
    # 3 (0, 2, 0)
    `c_code` int(11) NOT NULL COMMENT '업체고유번호',
    `e_code` int(11) NOT NULL COMMENT '직원 고유번호',
    
    # ?
    `ci_date` text NOT NULL COMMENT '연월일',
    `ci_time` text NOT NULL COMMENT '시간',
    
    # ?
    `ci_type` int(11) NOT NULL COMMENT '"구분 (입고 - 제조: 10, 수입:11, 구입:12, 출고 - 사용:20, 판매:21)  제조,수입 선택시 관리자 내 관리대장에 ""제조,수입"" 항목에 체크되도록, 수입, 구입 또는 판매인 경우에 하단 구입(판매)명세가 존재함."',
    `ci_count` int(11) NOT NULL COMMENT '수량 (양수는 입고, 음수는 출고)',
    
    # ?
    `ci_name` text NOT NULL COMMENT '구입명세 - 상호',
    `ci_number` text NOT NULL COMMENT '사업자등록번호',
    
    # X
    `ci_addr` text NOT NULL COMMENT '주소',
    `ci_phone` text NOT NULL COMMENT '전화번호',
    
    # ?
    `ci_use` text NOT NULL COMMENT '용도 (출고인 경우만 사용 함)',
    `ci_license` text NOT NULL COMMENT '구매자 영업 허가 구분 (출고인 경우만 사용 함)',
    
        `ci_in_out` int(11) NOT NULL COMMENT '대장 타입 (10:입고, 20:출고)',
    # ?
    `ci_insertdate` datetime NOT NULL COMMENT '물질 등록일',


    # 4 (0, 1, 0)
    `di_mtrl_eng` text NOT NULL COMMENT '영문 물질명',
        `di_mtrl_kor` text NOT NULL COMMENT '국문 물질명',
        `di_cas_no` varchar(50) NOT NULL COMMENT 'CAS 번호',
        `di_ke_no` varchar(20) NOT NULL COMMENT 'KE 번호',
    
    # 5 (0, 1, 1)
    `di_mtrl_clsf` varchar(200) NOT NULL COMMENT '물질 분류',
        `di_cnt_hcs` text NOT NULL COMMENT '함량정보 유독물질',
        `di_cnt_rs` text NOT NULL COMMENT '함량정보 금지물질',
        `di_cnt_ps` text NOT NULL COMMENT '함량정보 제한물질',
        `di_cnt_srpa` text NOT NULL COMMENT '함량정보 사고대비물질',
    
    # 6? (0, 0, 1)
    `di_hazd_sttm_url` varchar(200) NOT NULL COMMENT '유해위험문구 주소',

    # 7 (0, 1, 0)
    'action' text NOT NULL COMMENT '작업자 행동'


    # (0, 17, 14)

)

"""
작업자행동목록
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
밸브 점검
밸브 점검 중 오조작
반응기 내부 세척작업
교반기임펠러 교체
접이식임펠러 교체
전자식 밸브 설치
안전밸브 후단 플래어시스템 연결
설비 위험등급 설정관리
보조탱크 주입밸브 조작
보조탱크 주입밸브 오조작
감지기 설치
경보기 설치
개인보호장구 착용
개인보호장구 미착용
탱크로리 폐기물을 구별하여 처리
탱크로리 폐기물을 구별하지 않고 처리
다른 폐기물과 혼합을 피해서 폐기처리
다른 폐기물과 혼합하여 폐기처리
온습도 관리
온습도 미관리
볼배 보관 안전마개 및 밸브잠금 점검
볼배 보관 안전마개 및 밸브잠금 오조작
휴대용감지기를 활용하여 농도를 측정
개방된 용기에 폐기물을 소량씩 투입
개방된 용기에 폐기물을 대량 투입
드럼으로 드레인작업을 진행
"""