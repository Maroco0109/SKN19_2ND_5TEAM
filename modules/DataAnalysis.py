"""

데이터 시각화 모듈

- 모든 데이터 시각화 코드를 함수로 구현
- 모든 함수는 분석할 데이터를 df 형태로 받음
- 기본적으로는 return 값이 없는 함수. 필요에 따라 몇몇 데이터를 df 형태로 return

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from pathlib import Path
import inspect
import re
from IPython.display import display 

import modules.DataModify as DataModify

# 데이터 프레임에서, 범주형 데이터에 속하는 값들의 컬럼값을 출력
def show_value_counts(df, cols=None, boundary=30) :
    for col in df.columns:
        if df[col].nunique(dropna=True) > boundary :  # 각기 다른 값이 boundary 이상인 Continuous 한 값들은 출력하지 않음
            print(col)
            print('continuous')
            print("-"*20)
            continue

        series = df[col]
        if pd.api.types.is_extension_array_dtype(series.dtype):  # Nullable dtypes (e.g., Int64) need object conversion
            series = series.astype('object')
        value_counts = series.fillna("NA").value_counts(dropna=False)  # 결측치는 NA로 처리 후 출력
        print(value_counts)
        print("-" * 20)

# EDA

# Matplotlib 시각화 결과를 insight/img 폴더에 자동 보관하기 위한 헬퍼
if not hasattr(plt, '_dataanalysis_original_show'):
    plt._dataanalysis_original_show = plt.show

_FIG_SAVE_DIR = None
_FIG_SAVE_DIR_ERROR = False
_ORIGINAL_PLT_SHOW = plt._dataanalysis_original_show

# insight/img 경로를 생성하고 캐싱: 저장 실패 시 한 번만 경고 후 중단
def _get_fig_save_dir():
    global _FIG_SAVE_DIR, _FIG_SAVE_DIR_ERROR
    if _FIG_SAVE_DIR is not None or _FIG_SAVE_DIR_ERROR:
        return _FIG_SAVE_DIR
    try:
        repo_root = Path(__file__).resolve().parents[1]
        target = repo_root / 'insight' / 'img'
        target.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"[DataAnalysis] Failed to prepare insight/img directory: {exc}")
        _FIG_SAVE_DIR_ERROR = True
        return None
    _FIG_SAVE_DIR = target
    return _FIG_SAVE_DIR


# 파일명에 사용할 접두어를 안전한 문자열로 정규화
def _sanitize_prefix(prefix: str) -> str:
    clean = re.sub(r'[^0-9A-Za-z_-]+', '_', prefix or 'figure')
    return clean.strip('_') or 'figure'


# 호출 스택을 훑어 어떤 플롯 함수가 그림을 생성했는지 추론
def _infer_plot_prefix():
    for frame_info in inspect.stack():
        name = frame_info.function
        if name.startswith('_plot') or name.startswith('plot') or name.startswith('visualize'):
            return name.lstrip('_')
    return 'figure'


# 접두어와 매칭되는 기존 이미지 제거(재실행 시 덮어쓰기 위함)
def _clear_existing_exports(prefix: str):
    directory = _get_fig_save_dir()
    if directory is None:
        return
    for path in directory.glob(f"{prefix}*.png"):
        try:
            path.unlink()
        except Exception:
            ...


def _save_axes_images(fig, directory: Path, prefix: str):
    saved = []
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception as exc:
        print(f"[DataAnalysis] Failed to prepare axes renderer: {exc}")
        return saved
    axes = [ax for ax in fig.get_axes() if ax.get_visible()]
    if not axes:
        return saved
    for idx, ax in enumerate(axes, start=1):
        try:
            bbox = ax.get_tightbbox(renderer)
            if bbox is None:
                continue
            bbox = bbox.expanded(1.02, 1.08)
            bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
            stem = f"{prefix}_ax{idx:02d}"
            path = directory / f"{stem}.png"
            fig.savefig(path, dpi=300, bbox_inches=bbox_inches)
            saved.append(path)
        except Exception as exc:
            print(f"[DataAnalysis] Failed to save axes #{idx} for {prefix}: {exc}")
    return saved


def _save_current_figure(prefix: str = None):
    directory = _get_fig_save_dir()
    if directory is None:
        return []
    fig = plt.gcf()
    if fig is None:
        return []
    prefix = _sanitize_prefix(prefix or _infer_plot_prefix())
    _clear_existing_exports(prefix)
    figure_path = directory / f"{prefix}.png"
    try:
        fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    except Exception as exc:
        print(f"[DataAnalysis] Failed to save figure {figure_path}: {exc}")
        return []
    saved_paths = [figure_path]
    saved_paths.extend(_save_axes_images(fig, directory, prefix))
    return saved_paths


def _show(*args, **kwargs):
    prefix = kwargs.pop('filename_prefix', None)
    saved_paths = _save_current_figure(prefix)
    for path in saved_paths:
        try:
            rel = path.relative_to(Path.cwd())
            location = rel
        except ValueError:
            location = path
        label = 'axes' if '_ax' in path.stem else 'figure'
        print(f"[DataAnalysis] Saved {label}: {location}")
    return _ORIGINAL_PLT_SHOW(*args, **kwargs)


if plt.show is not _show:
    # 노트북/스크립트 어디에서든 plt.show 호출 시 자동 저장되도록 한 번만 교체
    plt.show = _show

def _set_style():
    # 시각화 공통 스타일 설정: seaborn 테마, 한글 폰트, 마이너스 부호 표시
    sns.set_theme(style="whitegrid", palette="crest")
    try:
        plt.rc('font', family='NanumGothic')
    except Exception:
        ...
    plt.rc('axes', unicode_minus=False)


def _load_encoded_data(df=None):
    # 데이터 로딩: 인자가 주어지면 해당 df 사용, 아니면 인코딩된 CSV 로딩
    # - 메인: data/encoded_dataset.csv
    # - COD 확장: data/EDA/encoded_dataset_COD.csv (없으면 메인으로 대체)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df.copy(), df.copy()

    def _find(rel):
        # 다양한 기준 경로에서 파일 탐색: CWD, CWD의 상위, 리포지토리 루트(모듈 기준)
        candidates = []
        cwd = Path.cwd()
        candidates.append(cwd)
        candidates.append(cwd.parent)
        try:
            repo_root = Path(__file__).resolve().parents[1]
            candidates.append(repo_root)
        except Exception:
            pass
        # 중복 제거(순서 유지)
        seen = set()
        uniq = []
        for b in candidates:
            if b not in seen:
                uniq.append(b); seen.add(b)
        for base in uniq:
            p = base / rel
            if p.exists():
                return p
        return None

    p_main = _find(Path('data/encoded_dataset.csv'))
    p_cod = _find(Path('data/EDA/encoded_dataset_COD.csv'))
    enc = pd.read_csv(p_main) if p_main and p_main.exists() else None
    enc_cod = pd.read_csv(p_cod) if p_cod and p_cod.exists() else enc
    if enc is None:
        hint = f"searched at: {Path.cwd()/'data/encoded_dataset.csv'}, {(Path.cwd().parent)/'data/encoded_dataset.csv'}"
        raise FileNotFoundError('encoded_dataset.csv not found and no DataFrame provided\n' + hint)
    return enc, enc_cod

def _load_decoding_maps_from_md(md_candidates=None):
    # encoded_label.md 파서를 통해 코드→영문 라벨 매핑을 생성
    md_candidates = md_candidates or [
        'encoded_label.md', './encoded_label.md',
        'insight/encoded_label.md', './insight/encoded_label.md',
        '../insight/encoded_label.md'
    ]
    maps = {}
    def parse(path: Path):
        nonlocal maps
        try:
            with path.open('r', encoding='utf-8') as f:
                current = None
                for line in f:
                    s = line.strip()
                    if s.startswith('[') and s.endswith(']') and len(s) > 2:
                        current = s[1:-1]
                        maps[current] = {}
                    elif '->' in s and current:
                        parts = s.split('->', 1)
                        k = parts[0].strip().lstrip('-').strip()
                        v = parts[1].strip()
                        if k.isdigit():
                            try:
                                maps[current][int(k)] = v
                            except Exception:
                                pass
        except Exception:
            return False
        return True
    for p in md_candidates:
        path = Path(p)
        if path.exists() and parse(path):
            break
    return maps

# SEER ICD-O-3 Primary Site(정수 코드) → 한국어 세부 부위명 매핑
def map_primary_site_code_to_korean(code) -> str:
    try:
        c = int(code)
    except Exception:
        # 숫자 아님: 그대로 반환
        return str(code)

    # 3자리(예: 341 → C34.1) 단위의 세부 매핑 우선
    specific = {
        # 식도(C15.x)
        150: '식도-경부', 151: '식도-흉부', 152: '식도-복부', 153: '식도-상부 1/3', 154: '식도-중부 1/3', 155: '식도-하부 1/3',
        158: '식도-중첩병변', 159: '식도, 기타불명',
        # 위(C16.x)
        160: '위-분문', 161: '위-저부', 162: '위-체부', 163: '위-유문전정부', 164: '위-유문', 165: '위-소만', 166: '위-대만',
        168: '위-중첩병변', 169: '위, 기타불명',
        # 소장(C17.x)
        170: '소장', 171: '십이지장', 172: '공장', 173: '회장', 178: '소장-중첩병변', 179: '소장, 기타불명',
        # 대장(C18.x)
        180: '맹장', 181: '충수', 182: '상행결장', 183: '간만곡', 184: '횡행결장', 185: '비만곡', 186: '하행결장',
        187: '에스상결장', 188: '대장-중첩병변', 189: '대장, 기타불명',
        # 직장구불/직장(C19–20)
        190: '직장구불결장 이행부', 200: '직장',
        # 항문(C21.x)
        210: '항문', 211: '항문관', 212: '배설강', 218: '직장/항문-중첩병변', 219: '직장/항문, 기타불명',
        # 간/담도(C22–C24)
        220: '간', 221: '간내 담관', 230: '담낭', 240: '간외 담관', 241: '바터 팽대부', 248: '담도-중첩병변', 249: '담도, 기타불명',
        # 췌장(C25.x)
        250: '췌장두부', 251: '췌장체부', 252: '췌장미부', 253: '췌장관', 254: '랑게르한스섬', 258: '췌장-중첩병변', 259: '췌장, 기타불명',
        # 호흡기(C30–C35)
        300: '코/비강/중이', 310: '부비동', 320: '후두', 330: '기관',
        340: '주기관지', 341: '폐-상엽', 342: '폐-중엽', 343: '폐-하엽', 348: '폐-중첩병변', 349: '폐, 기타불명',
        350: '기타 호흡기관',
        # 뼈/피부/연조직(C40–C49)
        400: '뼈 및 관절', 401: '뼈 및 관절', 430: '피부 흑색종', 440: '비흑색종 피부암', 490: '연조직(심장 포함)',
        # 복막/후복막(C48)
        480: '후복막', 481: '복막', 488: '복막/후복막-중첩병변', 489: '복막/후복막, 기타불명',
        # 유방/생식기(C50–C63)
        500: '유방', 510: '외음부', 520: '질', 530: '자궁경부', 540: '자궁체부', 550: '자궁, 기타불명', 560: '난소', 570: '기타 여성 생식기관',
        600: '음경', 610: '전립선', 620: '고환', 630: '기타 남성 생식기관',
        # 요로(C64–C68)
        640: '신장', 650: '신우', 660: '요관', 670: '방광', 680: '기타 요로기관',
        # 눈/뇌/신경계(C69–C72)
        690: '눈 및 안와', 700: '수막', 710: '뇌', 720: '뇌신경 및 기타 신경계',
        # 내분비(C73–C75)
        730: '갑상선', 739: '갑상선', 740: '부갑상선', 741: '부신', 750: '기타 내분비'
    }
    if c in specific:
        return specific[c]

    # 대분류(앞 두 자리) 기본 매핑으로 폴백
    major = c // 10
    major_map = {
        15: '식도', 16: '위', 17: '소장', 18: '대장', 19: '직장구불결장 이행부', 20: '직장', 21: '항문 및 항문관',
        22: '간 및 간내 담관', 23: '담낭', 24: '기타 담도', 25: '췌장', 26: '기타 소화기관',
        30: '코/비강/중이', 31: '부비동', 32: '후두', 33: '기관', 34: '폐 및 기관지', 35: '기타 호흡/흉강 장기',
        40: '뼈 및 관절', 41: '뼈 및 관절', 43: '피부 흑색종', 44: '비흑색종 피부암', 49: '연조직(심장 포함)',
        48: '후복막/복막',
        50: '유방', 51: '외음부', 52: '질', 53: '자궁경부', 54: '자궁체부', 55: '자궁, 기타불명', 56: '난소', 57: '기타 여성 생식기관',
        60: '음경', 61: '전립선', 62: '고환', 63: '기타 남성 생식기관',
        64: '신장', 65: '신우', 66: '요관', 67: '방광', 68: '기타 요로기관',
        69: '눈 및 안와', 70: '수막', 71: '뇌', 72: '뇌신경 및 기타 신경계',
        73: '갑상선', 74: '부신', 75: '기타 내분비(흉선 포함)',
        76: '특정부위불명/미상', 77: '림프절', 78: '전이성암(호흡/소화기)', 79: '전이성암(기타)', 80: '원발부위 불명'
    }
    return major_map.get(major, str(code))

def _augment_decoded_labels(encoded_cod_df):
    # 노트북과 동일한 방식으로 __label 및 __label_kor 컬럼을 보강
    maps = _load_decoding_maps_from_md()
    dec_targets = {
        'COD to site recode__enc': ('COD to site recode', maps.get('COD to site recode', {})),
        'Vital status recode (study cutoff used)__enc': ('Vital status recode (study cutoff used)', maps.get('Vital status recode (study cutoff used)', {})),
        'Survival months flag__enc': ('Survival months flag', maps.get('Survival months flag', {})),
    }
    df = encoded_cod_df
    # 기본 영/한 매핑 (필수 항목 우선)
    vital_en_map = {0: 'Alive', 1: 'Dead'}
    vital_kor_from_en = {'Alive': '생존', 'Dead': '사망'}
    survflag_kor_from_en = {
        'Complete dates are available and there are more than 0 days of survival': '완전한 날짜, 생존일수 > 0',
        'Incomplete dates are available and there cannot be zero days of follow-up': '불완전 날짜, 추적 0일 불가',
        'Not calculated because a Death Certificate Only or Autopsy Only case': '사망진단서/부검만, 미계산',
        'Complete dates are available and there are 0 days of survival': '완전한 날짜, 생존일수 0',
        'Incomplete dates are available and there could be zero days of follow-up': '불완전 날짜, 추적 0일 가능',
    }
    # COD 한글 매핑(코드 기반; 영문 매핑 부재시 대비)
    cod_ko_by_code = {
        0: '생존', 1: '간내담관', 2: '폐 및 기관지', 3: '기타 악성종양', 4: '기타 사망원인', 5: '대장(직장 제외)',
        6: '심장질환', 7: '알츠하이머', 8: '위', 9: '신장질환(신증후군 포함)', 10: '뇌혈관질환', 11: '간', 12: '유방',
        13: '만성폐쇄성폐질환', 14: '만성 림프구성 백혈병', 15: '만성 간질환/간경화', 18: '전립선', 19: '비호지킨 림프종',
        20: '당뇨병', 21: '사망원인 미상', 22: '신장 및 신우', 23: '고혈압(심장질환 동반 없음)', 24: '다발성 골수종',
        25: '급성 골수성 백혈병', 26: '뇌 및 기타 신경계', 27: '폐렴/인플루엔자', 28: '췌장', 31: '사고 및 부작용',
        32: '자궁체부', 33: '방광', 34: '제자리/양성/미확정 신생물', 35: '죽상경화증', 36: '식도', 37: '자궁경부',
        38: '자살/자해', 40: '피부 흑색종', 41: '직장 및 직장결장 이행부', 42: '자궁, 기타특정불가', 43: '연조직(심장 포함)',
        44: '비흑색종 피부암', 45: '혀', 46: '패혈증', 47: '갑상선', 52: '흉막', 53: '난소', 54: '기타 담도', 61: '뼈/관절',
        62: '편도', 63: '복막/망/장간막', 64: '후두', 65: '호지킨 림프종', 66: '후복막', 68: '결핵', 70: '구인두', 71: '질',
        72: '담낭', 73: '치은/기타 구강', 74: '기타 내분비(흉선 포함)', 76: '기관/종격/기타 호흡기관', 77: '고환',
        78: '급성 림프구성 백혈병', 79: '기타 요로기관', 80: '기타 구강/인두', 81: '기타 여성 생식기관', 82: '코/비강/중이',
        83: '기타 급성 백혈병', 84: '요관', 85: '외음부', 86: '만성 골수성 백혈병', 87: '눈/안와', 88: '입술', 89: '임신/출산/산욕 합병증'
    }
    cod_en_by_code = maps.get('COD to site recode', {})
    cod_en_to_ko = {en: cod_ko_by_code.get(code, en) for code, en in cod_en_by_code.items()}

    for enc_col, (orig_col, mapping) in dec_targets.items():
        if enc_col not in df.columns:
            continue
        # 영문 라벨
        if mapping:
            df[orig_col] = df[enc_col].map(mapping)
            df[enc_col.replace('__enc', '__label')] = df[enc_col].map(mapping)
        elif enc_col.endswith('Vital status recode (study cutoff used)__enc'):
            df[enc_col.replace('__enc', '__label')] = df[enc_col].map(vital_en_map)
        elif enc_col.endswith('COD to site recode__enc'):
            df[enc_col.replace('__enc', '__label_kor')] = df[enc_col].map(cod_ko_by_code)
        # 한글 라벨 추가
        lab = enc_col.replace('__enc', '__label')
        lab_k = enc_col.replace('__enc', '__label_kor')
        if lab in df.columns:
            if orig_col == 'COD to site recode':
                df[lab_k] = df[lab].map(cod_en_to_ko)
            elif orig_col == 'Vital status recode (study cutoff used)':
                df[lab_k] = df[lab].map(vital_kor_from_en)
            elif orig_col == 'Survival months flag':
                df[lab_k] = df[lab].map(survflag_kor_from_en)
    # Primary Site 한국어 상세 라벨 보강
    try:
        if 'Primary Site' in df.columns and 'Primary Site__label_kor' not in df.columns:
            df['Primary Site__label_kor'] = df['Primary Site'].map(map_primary_site_code_to_korean)
    except Exception:
        ...
    return df


def _build_year_label_map(df, year_col='Year of diagnosis'):
    """Encoded 연도 코드를 실제 연도 라벨로 매핑하는 헬퍼."""
    if year_col not in df.columns:
        return {}

    def _register(mp, key, value):
        try:
            mp[key] = int(value)
        except Exception:
            mp[key] = value
        try:
            mp[int(key)] = int(value)
        except Exception:
            ...
        mp[str(key)] = value

    mapping = {}
    try:
        enc_df, _ = _load_encoded_data(None)
    except Exception:
        enc_df = None

    if enc_df is not None and year_col in enc_df.columns:
        # 1) Patient ID 매칭을 통한 직접 매핑
        if 'Patient ID' in enc_df.columns and 'Patient ID' in df.columns:
            try:
                merged = (df[['Patient ID', year_col]].dropna()
                          .merge(enc_df[['Patient ID', year_col]].dropna(),
                                 on='Patient ID', suffixes=('_cod', '_orig')))
                if not merged.empty:
                    grp = merged.groupby(f'{year_col}_cod')[f'{year_col}_orig']
                    for code, series in grp:
                        s = pd.to_numeric(series, errors='coerce').dropna()
                        if not s.empty:
                            val = int(s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
                            _register(mapping, code, val)
                        else:
                            _register(mapping, code, series.iloc[0])
            except Exception:
                ...
        # 2) 최소 연도 기반 오프셋 폴백
        if not mapping:
            try:
                min_year = pd.to_numeric(enc_df[year_col], errors='coerce').dropna().min()
                codes = pd.to_numeric(df[year_col], errors='coerce').dropna().astype(int).unique()
                if pd.notna(min_year):
                    for c in sorted(codes):
                        _register(mapping, c, int(min_year) + int(c))
            except Exception:
                ...

    return mapping


def _decode_year_value(code, year_map):
    """단일 연도 코드 값을 매핑 정보에 따라 디코딩."""
    if not year_map or pd.isna(code):
        return code
    # 그대로 일치하는 키 우선
    if code in year_map:
        return year_map[code]
    str_code = str(code)
    if str_code in year_map:
        return year_map[str_code]
    try:
        float_code = float(code)
        if float_code in year_map:
            return year_map[float_code]
        int_code = int(float_code)
        if int_code in year_map:
            return year_map[int_code]
        str_int_code = str(int_code)
        if str_int_code in year_map:
            return year_map[str_int_code]
    except Exception:
        ...
    return code


def _decode_year_series(codes, year_map):
    """연도 코드 시퀀스를 매핑하여 디코딩된 값 Series 반환."""
    if not hasattr(codes, 'map'):
        codes = pd.Series(codes)
    return codes.map(lambda v: _decode_year_value(v, year_map))

def _ensure_survival_bin(df, col='Survival months', out_col='Survival months_bin_3m'):
    # 생존 개월 수를 3개월 단위 구간으로 변환하는 파생열 생성(없을 때만 생성)
    if out_col in df.columns:
        return df
    if col in df.columns:
        df = df.copy()
        df[out_col] = DataModify.DataPreprocessing.bin_survival_months(df[col], bin_size=3)
    return df


def _plot_corr_with_target(encoded_df):
    # [히트맵] 인코딩된 수치 피처 vs target_label의 스피어만 상관계수 시각화(전체/상위 TOP-N)
    if 'target_label' not in encoded_df.columns:
        return
    num_cols = [c for c in encoded_df.columns if pd.api.types.is_numeric_dtype(encoded_df[c])]
    # 불필요 식별자 및 타깃 유도 변수 제외
    exclude_cols = {
        'Unnamed: 0',
        'Patient ID',
        'Vital status recode (study cutoff used)__enc',
        'COD to site recode__enc',
        'Survival months flag__enc'
    }
    num_cols = [c for c in num_cols if c not in exclude_cols]
    if not num_cols:
        return
    corr_with_target = encoded_df[num_cols].corrwith(encoded_df['target_label'], method='spearman').dropna()
    if corr_with_target.empty:
        return
    # 한국어 라벨 매핑
    kor_map = {
        'Sex': '성별',
        'Age recode with <1 year olds and 90+': '연령대',
        'Year of diagnosis': '진단 연도',
        'Year of follow-up recode': '추적 연도',
        'Race recode (W, B, AI, API)': '인종 재코드',
        'Site recode ICD-O-3/WHO 2008': '암 부위 재코드',
        'Primary Site': '원발 부위',
        'Primary Site - labeled': '원발 부위 라벨',
        'Derived Summary Grade 2018 (2018+)': '요약 등급 2018',
        'Laterality': '좌우 구분',
        'EOD Schema ID Recode (2010+)': 'EOD 스키마 재코드',
        'Combined Summary Stage with Expanded Regional Codes (2004+)': 'SEER 요약 병기(확장)',
        'RX Summ--Surg Prim Site (1998+)': '수술 코드',
        'RX Summ--Scope Reg LN Sur (2003+)': '림프절 절제 범위',
        'RX Summ--Surg Oth Reg/Dis (2003+)': '기타 수술',
        'Sequence number': '순서 번호',
        'Median household income inflation adj to 2023': '가구 소득(2023 물가보정)',
        'Number of Cores Positive Recode (2010+)': '양성 코어 수',
        'Number of Cores Examined Recode (2010+)': '검사 코어 수',
        'EOD Primary Tumor Recode (2018+)': 'EOD 원발 종양',
        'PRCDA 2020': 'PRCDA 2020',
        'Survival months': '생존 개월',
        'Survival months_bin_3m': '생존 개월(3개월 구간)',
        'target_label': '타깃 라벨',
        'Vital status recode (study cutoff used)__enc': '생존 상태(인코딩)'
    }
    heat = corr_with_target.sort_values(ascending=False).to_frame(name='Spearman r')
    # 인덱스(피처명)를 한국어로 변환
    heat.index = [kor_map.get(str(idx), str(idx)) for idx in heat.index]
    plt.figure(figsize=(8, max(3, 0.25 * len(heat))))
    sns.heatmap(heat, annot=True, fmt='.3f', cmap='vlag', vmin=-1, vmax=1, cbar=True)
    plt.title('Spearman correlation with target_label (encoded features)')
    plt.xlabel('')
    plt.ylabel('변수')
    plt.tight_layout()
    plt.show()

    top_n = min(25, len(heat))
    top_abs = corr_with_target.reindex(corr_with_target.abs().sort_values(ascending=False).head(top_n).index)
    heat2 = top_abs.to_frame(name='Spearman r')
    heat2.index = [kor_map.get(str(idx), str(idx)) for idx in heat2.index]
    plt.figure(figsize=(8, max(3, 0.35 * len(heat2))))
    sns.heatmap(heat2, annot=True, fmt='.3f', cmap='vlag', vmin=-1, vmax=1, cbar=True)
    plt.title(f'Top-{top_n} |Spearman r| with target_label')
    plt.xlabel('')
    plt.ylabel('변수')
    plt.tight_layout()
    plt.show()


def _plot_survival_months(encoded_df):
    # [시계열] 생존개월에 따른 사건확률(= target_label != -1) 변화
    # - 월 단위 곡선 + 3개월 구간 곡선(노이즈 완화를 위한 스무딩 포함)
    cols = ['Survival months', 'Survival months_bin_3m', 'target_label']
    df = _ensure_survival_bin(encoded_df, 'Survival months', 'Survival months_bin_3m')
    if not all(c in df.columns for c in cols):
        return
    tmp = df[cols].dropna(subset=['Survival months']).copy()
    tmp['is_death'] = (tmp['target_label'] != -1).astype(int)
    grp_m = tmp.groupby('Survival months').agg(p_death=('is_death','mean'), n=('is_death','size')).reset_index().sort_values('Survival months')
    n_min = 50
    grp_mf = grp_m[grp_m['n'] >= n_min].copy()
    grp_mf['p_smooth'] = grp_mf['p_death'].rolling(window=3, center=True, min_periods=1).mean()
    plt.figure(figsize=(6, 4))
    plt.plot(grp_mf['Survival months'], grp_mf['p_death'], color='tab:blue', alpha=0.35, label='Monthly p(death)')
    plt.plot(grp_mf['Survival months'], grp_mf['p_smooth'], color='tab:blue', linewidth=2.0, label='Monthly (smoothed)')
    plt.ylim(0, 1)
    plt.title('Death probability over Survival months')
    plt.xlabel('Survival months')
    plt.ylabel('P(target_label != -1)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    grp_b = tmp.groupby('Survival months_bin_3m').agg(p_death=('is_death','mean'), n=('is_death','size')).reset_index().sort_values('Survival months_bin_3m')
    grp_bf = grp_b[grp_b['n'] >= n_min].copy()
    grp_bf['p_smooth'] = grp_bf['p_death'].rolling(window=3, center=True, min_periods=1).mean()
    plt.figure(figsize=(6, 4))
    plt.plot(grp_bf['Survival months_bin_3m'], grp_bf['p_death'], color='tab:orange', alpha=0.35, label='3-month bin p(death)')
    plt.plot(grp_bf['Survival months_bin_3m'], grp_bf['p_smooth'], color='tab:orange', linewidth=2.0, label='3-month (smoothed)')
    plt.ylim(0, 1)
    plt.title('Death probability over 3-month bins')
    plt.xlabel('Survival months (3-month bins)')
    plt.ylabel('P(target_label != -1)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_basic_distributions(encoded_cod_df):
    # [기본 분포] 성별(파이), 연령대(막대), 생존 상태(파이), 진단연도(라인)
    df = _augment_decoded_labels(encoded_cod_df.copy())
    cols = ['Sex', 'Age recode with <1 year olds and 90+', 'Year of diagnosis']
    if not any(c in df.columns for c in cols):
        return
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 성별 분포 (파이)
    if 'Sex' in df.columns:
        sex_counts = df['Sex'].value_counts()
        sex_labels = ['여성', '남성'] if len(sex_counts) == 2 else [f'Sex_{i}' for i in sex_counts.index]
        colors = ['#FF6B9D', '#4DABF7']
        ax1.pie(sex_counts.values, labels=sex_labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('성별 분포', fontsize=14, fontweight='bold', pad=20)
    else:
        ax1.axis('off')

    # 연령대별 환자 분포 (막대)
    age_order = DataModify.DataPreprocessing.AGE_RECODE_ORDER
    age_kor_map = {'00 years':'0','01-04 years':'1-4','05-09 years':'5-9','10-14 years':'10-14','15-19 years':'15-19','20-24 years':'20-24','25-29 years':'25-29','30-34 years':'30-34','35-39 years':'35-39','40-44 years':'40-44','45-49 years':'45-49','50-54 years':'50-54','55-59 years':'55-59','60-64 years':'60-64','65-69 years':'65-69','70-74 years':'70-74','75-79 years':'75-79','80-84 years':'80-84','85-89 years':'85-89','90+ years':'90 이상'}
    age_col = 'Age recode with <1 year olds and 90+'
    if age_col in df.columns:
        _series = df[age_col]
        _num = pd.to_numeric(_series, errors='coerce')
        if _num.notna().sum() > 0:
            _codes = _num.dropna().astype(int)
            age_counts_raw = _codes.value_counts().sort_index()
            order_codes = [i for i in range(len(age_order)) if i in age_counts_raw.index]
            age_counts = age_counts_raw.reindex(order_codes).fillna(0).astype(int)
            xticklabels = [age_kor_map[age_order[i]] for i in age_counts.index]
        else:
            age_counts_raw = _series.value_counts()
            ages_present = [a for a in age_order if a in age_counts_raw.index]
            age_counts = age_counts_raw.reindex(ages_present).fillna(0).astype(int)
            xticklabels = [age_kor_map.get(a, a) for a in age_counts.index]
        positions = np.arange(len(age_counts))
        ax2.bar(positions, age_counts.values, color='#51CF66', alpha=0.8)
        ax2.set_title('연령대별 환자 분포', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('연령대'); ax2.set_ylabel('환자 수')
        ax2.set_xticks(positions); ax2.set_xticklabels(xticklabels, rotation=45, ha='center')
        ax2.set_axisbelow(True); ax2.grid(axis='y', linestyle='--', alpha=0.3)
        top_ages = age_counts.nlargest(5)
        for i_pos, (age_key, count) in enumerate(zip(age_counts.index, age_counts.values)):
            if age_key in top_ages.index:
                ax2.text(positions[i_pos], count + max(1, age_counts.max()*0.01), f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax2.axis('off')

    # 생존 상태 분포 (파이, 한글 라벨)
    vital_kor_col = 'Vital status recode (study cutoff used)__label_kor'
    if vital_kor_col in df.columns:
        vital_status = df[vital_kor_col].value_counts()
        colors_vital = ['#69DB7C', '#FF8787']
        ax3.pie(vital_status.values, labels=vital_status.index.tolist(), autopct='%1.1f%%', colors=colors_vital, startangle=90, wedgeprops=dict(width=0.6))
        ax3.set_title('생존 상태 분포', fontsize=14, fontweight='bold', pad=20)
    else:
        ax3.axis('off')

    # 진단 연도별 환자 수 추이 (라인)
    if 'Year of diagnosis' in df.columns:
        year_series = df['Year of diagnosis'].dropna()
        if year_series.empty:
            ax4.axis('off')
        else:
            year_counts = (year_series.value_counts()
                                         .rename_axis('Year_code')
                                         .reset_index(name='Count'))
            year_map = _build_year_label_map(df, 'Year of diagnosis')
            decoded_years = _decode_year_series(year_counts['Year_code'], year_map)
            numeric_years = pd.to_numeric(decoded_years, errors='coerce')
            if numeric_years.isna().all():
                numeric_years = pd.to_numeric(year_counts['Year_code'], errors='coerce')
            year_counts['Year_label'] = numeric_years
            year_counts = year_counts.dropna(subset=['Year_label'])
            if year_counts.empty:
                ax4.axis('off')
            else:
                year_counts['Year_label'] = year_counts['Year_label'].astype(int)
                year_counts = (year_counts.groupby('Year_label', as_index=False)['Count']
                                          .sum()
                                          .sort_values('Year_label'))
                x_vals = year_counts['Year_label'].to_numpy()
                ax4.plot(x_vals, year_counts['Count'], marker='o', linewidth=2)
                ax4.set_title('진단 연도별 환자 수 추이', fontsize=14, fontweight='bold', pad=20)
                ax4.set_xlabel('진단 연도'); ax4.set_ylabel('환자 수'); ax4.grid(True, alpha=0.3)
                ax4.set_xticks(x_vals)
                ax4.set_xticklabels([str(int(x)) for x in x_vals], rotation=45, ha='center')
    else:
        ax4.axis('off')
    plt.tight_layout(); plt.show()


def _plot_site_survival_year(encoded_cod_df):
    # [암 부위/연령/연도] 노트북 로직에 맞춘 4개 서브플롯
    df = _augment_decoded_labels(encoded_cod_df.copy())
    # (부위) 한글 매핑 및 헬퍼
    site_korean_mapping = {
        'Lung and Bronchus': '폐 및 기관지','Breast': '유방','Prostate': '전립선','Stomach': '위','Liver': '간','Pancreas': '췌장','Esophagus': '식도','Ovary': '난소',
        'Kidney and Renal Pelvis': '신장 및 신우','Urinary Bladder': '방광','Rectum': '직장','Rectosigmoid Junction': '직장구불결장 이행부',
        'Ascending Colon': '상행결장','Sigmoid Colon': '에스상결장','Transverse Colon': '횡행결장','Descending Colon': '하행결장',
        'Cecum': '맹장','Large Intestine, NOS': '대장, 기타불명','Thyroid': '갑상선','Brain': '뇌','Melanoma of the Skin': '피부 흑색종',
        'NHL - Nodal': '비호지킨림프종 - 림프절','NHL - Extranodal': '비호지킨림프종 - 림프절외','Hodgkin - Nodal': '호지킨림프종 - 림프절','Hodgkin - Extranodal': '호지킨림프종 - 림프절외',
        'Cranial Nerves Other Nervous System': '뇌신경 및 기타 신경계','Gum and Other Mouth': '치은 및 기타 구강','Tongue': '혀','Tonsil': '편도',
        'Larynx': '후두','Nasopharynx': '비인두','Oropharynx': '구인두','Hypopharynx': '하인두','Nose, Nasal Cavity and Middle Ear': '코/비강/중이',
        'Eye and Orbit': '눈 및 안와','Soft Tissue including Heart': '연조직(심장 포함)','Bones and Joints': '뼈 및 관절','Salivary Gland': '타액선',
        'Uterus, NOS': '자궁, 기타불명','Cervix Uteri': '자궁경부','Corpus Uteri': '자궁체부','Vagina': '질','Vulva': '외음부','Penis': '음경','Testis': '고환',
        'Gallbladder': '담낭','Intrahepatic Bile Duct': '간내 담관','Other Biliary': '기타 담도','Small Intestine': '소장','Appendix': '충수',
        'Peritoneum, Omentum and Mesentery': '복막/망/장간막','Retroperitoneum': '후복막',
        'Trachea, Mediastinum and Other Respiratory Organs': '기관/종격동/기타 호흡기관',
    }
    def get_site_korean_name(name: str):
        try:
            key = str(name).strip()
            if key in site_korean_mapping:
                return site_korean_mapping[key]
            low = {k.lower(): v for k, v in site_korean_mapping.items()}
            return low.get(key.lower(), name)
        except Exception:
            return name

    df_site = df.copy()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))

    # 3-1. 주요 암 부위 코드별 환자 수 (상위 10개)
    if 'Primary Site' in df_site.columns:
        site_counts = df_site['Primary Site'].value_counts().head(10)
        colors_sites = plt.cm.viridis(np.linspace(0, 1, len(site_counts)))
        ax1.barh(range(len(site_counts)), site_counts.values, color=colors_sites)
        ax1.set_yticks(range(len(site_counts)))
        # Primary Site 코드(예: 341, 163 등)를 한글 명칭으로 변환
        # - 우선 텍스트 라벨이 있으면 그것을 사용하고,
        # - 없을 경우 ICD-O-3 Topography의 앞 2자리(Cxx)로 대분류 맵핑
        major_site_kor_map = {
            0: '입술',
            1: '혀', 2: '혀', 3: '치은', 4: '구강저', 5: '구개', 6: '기타 구강',
            7: '타액선', 8: '타액선', 9: '편도', 10: '구인두', 11: '비인두', 12: '하인두', 13: '하인두', 14: '기타 구강/인두',
            15: '식도', 16: '위', 17: '소장', 18: '대장', 19: '직장구불결장 이행부', 20: '직장', 21: '항문 및 항문관',
            22: '간 및 간내 담관', 23: '담낭', 24: '기타 담도', 25: '췌장', 26: '기타 소화기관',
            30: '코/비강/중이', 31: '부비동', 32: '후두', 33: '기관', 34: '폐 및 기관지',
            37: '흉선', 38: '심장/종격동/흉막', 39: '기타 호흡/흉강 장기',
            40: '뼈 및 관절', 41: '뼈 및 관절', 43: '피부 흑색종', 44: '비흑색종 피부암',
            47: '말초신경계', 48: '후복막/복막', 49: '연조직(심장 포함)',
            50: '유방', 51: '외음부', 52: '질', 53: '자궁경부', 54: '자궁체부', 55: '자궁, 기타불명', 56: '난소', 57: '기타 여성 생식기관', 58: '태반',
            60: '음경', 61: '전립선', 62: '고환', 63: '기타 남성 생식기관',
            64: '신장', 65: '신우', 66: '요관', 67: '방광', 68: '기타 요로기관',
            69: '눈 및 안와', 70: '수막', 71: '뇌', 72: '뇌신경 및 기타 신경계',
            73: '갑상선', 74: '부신', 75: '기타 내분비(흉선 포함)',
            76: '특정부위불명/미상', 77: '림프절', 78: '전이성암(호흡/소화기)', 79: '전이성암(기타)', 80: '원발부위 불명'
        }

        def _ko_for_code(code):
            # 0) 세부 한국어 매핑 우선 적용
            try:
                name0 = map_primary_site_code_to_korean(code)
                if name0 and isinstance(name0, str) and name0.strip():
                    return name0
            except Exception:
                ...
            # 1) 텍스트 라벨 컬럼이 실제 문자열일 때 우선 사용
            group = df_site[df_site['Primary Site'] == code]
            if not group.empty:
                for col in ['Site recode ICD-O-3/WHO 2008', 'Primary Site - labeled']:
                    if col in group.columns and pd.api.types.is_object_dtype(group[col]):
                        m = group[col].dropna().mode()
                        if len(m) > 0:
                            return get_site_korean_name(m.iloc[0])
            # 2) 숫자 코드 → ICD-O-3 Topography 대분류로 한글 변환
            try:
                icode = int(code)
                major = icode // 10  # 앞 2자리(Cxx)
                if major in major_site_kor_map:
                    return major_site_kor_map[major]
            except Exception:
                ...
            # 3) 실패 시 원문 유지(이전에는 "Site {code}")
            return str(code)
        site_names = [_ko_for_code(sc) for sc in site_counts.index]
        ax1.set_yticklabels(site_names, fontsize=10)
        ax1.set_title('주요 암 부위 코드별 환자 수 (상위 10개)', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('환자 수')
        for i, v in enumerate(site_counts.values):
            ax1.text(v + 1000, i, f'{v:,}', va='center', fontweight='bold', fontsize=9)
        legend_elements = []
        for i, (site_idx, count) in enumerate(site_counts.head(5).items()):
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=colors_sites[i], label=f'{site_names[i]} ({count:,}명)'))
        ax1.legend(handles=legend_elements, loc='lower right', fontsize=9, title='주요 암 부위')
    else:
        ax1.axis('off')

    # 3-2. 암 부위별 생존율 (상위 10개 기준)
    if 'Primary Site' in df_site.columns and 'Vital status recode (study cutoff used)__enc' in df.columns:
        top_sites = df_site['Primary Site'].value_counts().head(10).index
        site_names = [_ko_for_code(s) for s in top_sites]
        survival_by_site = []
        for site in top_sites:
            site_data = df[df['Primary Site'] == site]
            survival_rate = (site_data['Vital status recode (study cutoff used)__enc'] == 0).mean() * 100
            survival_by_site.append(survival_rate)
        colors_survival = ['#FF6B6B' if r < 80 else '#51CF66' if r > 90 else '#FFD93D' for r in survival_by_site]
        ax2.bar(range(len(survival_by_site)), survival_by_site, color=colors_survival, alpha=0.8)
        ax2.set_xticks(range(len(top_sites)))
        ax2.set_xticklabels(site_names, rotation=45, ha='right')
        ax2.set_title('주요 암 부위별 생존율', fontsize=14, fontweight='bold', pad=20)
        ax2.set_ylabel('생존율 (%)')
        ymin = max(0, min(survival_by_site) - 5) if len(survival_by_site) else 0
        ax2.set_ylim(ymin, 100)
        for i, v in enumerate(survival_by_site):
            ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        from matplotlib.patches import Patch
        legend_elements_survival = [
            Patch(facecolor='#51CF66', alpha=0.8, label='높은 생존율 (90% 이상)'),
            Patch(facecolor='#FFD93D', alpha=0.8, label='중간 생존율 (80-90%)'),
            Patch(facecolor='#FF6B6B', alpha=0.8, label='낮은 생존율 (80% 미만)')
        ]
        ax2.legend(handles=legend_elements_survival, loc='upper right', fontsize=9, title='생존율 구간')
    else:
        ax2.axis('off')

    # 3-3. 연령대별 생존율 (버블)
    if 'Age recode with <1 year olds and 90+' in df.columns and 'Vital status recode (study cutoff used)__enc' in df.columns:
        age_survival = df.groupby('Age recode with <1 year olds and 90+')['Vital status recode (study cutoff used)__enc'].agg(['count', lambda x: (x == 0).mean() * 100]).reset_index()
        age_survival.columns = ['Age_Code', 'Count', 'Survival_Rate']
        age_survival = age_survival[age_survival['Count'] >= 1000]
        age_order = DataModify.DataPreprocessing.AGE_RECODE_ORDER
        age_kor_map = { '00 years':'0세','01-04 years':'1-4세','05-09 years':'5-9세','10-14 years':'10-14세','15-19 years':'15-19세','20-24 years':'20-24세','25-29 years':'25-29세','30-34 years':'30-34세','35-39 years':'35-39세','40-44 years':'40-44세','45-49 years':'45-49세','50-54 years':'50-54세','55-59 years':'55-59세','60-64 years':'60-64세','65-69 years':'65-69세','70-74 years':'70-74세','75-79 years':'75-79세','80-84 years':'80-84세','85-89 years':'85-89세','90+ years':'90세 이상' }
        if pd.api.types.is_numeric_dtype(age_survival['Age_Code']):
            order = list(range(len(age_order)))
            age_survival = age_survival[age_survival['Age_Code'].isin(order)].copy()
            age_survival['Age_Code'] = pd.Categorical(age_survival['Age_Code'], categories=order, ordered=True)
            age_survival = age_survival.sort_values('Age_Code').reset_index(drop=True)
            age_survival['Age_Label'] = age_survival['Age_Code'].map(lambda x: age_kor_map.get(age_order[int(x)], str(x)) if pd.notna(x) else None)
        else:
            age_survival['Age_Code'] = pd.Categorical(age_survival['Age_Code'], categories=age_order, ordered=True)
            age_survival = age_survival.sort_values('Age_Code').reset_index(drop=True)
            age_survival['Age_Label'] = age_survival['Age_Code'].map(lambda x: age_kor_map.get(str(x), str(x)))
        x_pos = age_survival['Age_Code'].cat.codes
        sc = ax3.scatter(x_pos, age_survival['Survival_Rate'], s=age_survival['Count']/100, alpha=0.6, c=age_survival['Survival_Rate'], cmap='RdYlGn', edgecolors='black', linewidth=0.5)
        ax3.set_title('연령대별 생존율 (버블 크기: 환자 수)', fontsize=14, fontweight='bold', pad=20)
        ax3.set_xlabel('연령대'); ax3.set_ylabel('생존율 (%)'); ax3.grid(True, alpha=0.3)
        ax3.set_xticks(x_pos); ax3.set_xticklabels(age_survival['Age_Label'], rotation=45, ha='center')
        cbar = plt.colorbar(sc, ax=ax3); cbar.set_label('생존율 (%)', rotation=270, labelpad=20)
    else:
        ax3.axis('off')

    # 3-4. 진단 연도별 생존율 추이
    survival_col = 'Vital status recode (study cutoff used)__enc'
    if 'Year of diagnosis' in df.columns and survival_col in df.columns:
        tmp_year = df[['Year of diagnosis', survival_col]].copy()
        tmp_year[survival_col] = pd.to_numeric(tmp_year[survival_col], errors='coerce')
        tmp_year = tmp_year.dropna(subset=[survival_col])
        if tmp_year.empty:
            ax4.axis('off')
        else:
            year_survival = tmp_year.groupby('Year of diagnosis')[survival_col].agg(
                Count='size',
                Survival_Rate=lambda x: (x == 0).mean() * 100 if len(x) else np.nan
            ).reset_index().rename(columns={'Year of diagnosis': 'Year_code'})
            year_survival = year_survival.dropna(subset=['Survival_Rate'])
            year_survival = year_survival[year_survival['Count'] >= 1000]
            year_map = _build_year_label_map(df, 'Year of diagnosis')
            year_survival['Year_label'] = year_survival['Year_code'].map(year_map).fillna(year_survival['Year_code'])
            year_survival['Year_label'] = pd.to_numeric(year_survival['Year_label'], errors='coerce')
            year_survival = year_survival.dropna(subset=['Year_label'])
            if year_survival.empty:
                ax4.axis('off')
            else:
                year_survival['Year_label'] = year_survival['Year_label'].astype(int)
                ax4.plot(year_survival['Year_label'], year_survival['Survival_Rate'], marker='o', linewidth=2, markersize=6, color='#FF6B6B')
                ax4.set_title('진단 연도별 생존율 추이', fontsize=14, fontweight='bold', pad=20)
                ax4.set_xlabel('진단 연도'); ax4.set_ylabel('생존율 (%)'); ax4.grid(True, alpha=0.3)
                try:
                    ax4.set_ylim(auto=True)
                    x = year_survival['Year_label']
                    z = np.polyfit(x, year_survival['Survival_Rate'], 1)
                    p = np.poly1d(z)
                    ax4.plot(x, p(x), "--", alpha=0.8, color='blue', linewidth=2)
                except Exception:
                    ...
                ticks = sorted(year_survival['Year_label'].unique())
                ax4.set_xticks(ticks)
                ax4.set_xticklabels([str(t) for t in ticks], rotation=45, ha='center')
    else:
        ax4.axis('off')
    plt.tight_layout(); plt.show()


def _plot_stage_surgery_gender_age(encoded_cod_df):
    # [치료/병기/성별×연령] 노트북 4개 플롯(도넛, 병기 생존율, 수술 생존율, 성별×연령 히트맵)
    df = _augment_decoded_labels(encoded_cod_df.copy())
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 4-1. 병기별 환자 분포 (도넛)
    stage_col = 'Combined Summary Stage with Expanded Regional Codes (2004+)'
    # 병기 컬럼을 표준화(확장코드 포함 → 0/1/2/3/9로 집계)
    stage_num = None
    if stage_col in df.columns:
        series = df[stage_col]
        # 병기(Stage) 분류 규칙 요약
        # - 입력: SEER "Combined Summary Stage with Expanded Regional Codes (2004+)"
        # - 입력 형태: 정수(확장 코드: 0,1,2,3,4,5,6,7,9) 또는 텍스트('In situ', 'Localized', 'Regional', 'Distant', 'Unknown/...')
        # - 출력: 기본 병기 5단계 {0, 1, 2, 3, 9}
        #   * 0 -> 0 (In situ)
        #   * 1 -> 1 (Localized)
        #   * 2/3/4/5 -> 2 (Regional)
        #   * 7 -> 3 (Distant)
        #   * 6/9/그 외 -> 9 (Unknown/Unstaged)
        # - 배경: 확장 코드는 Regional을 2~5로 세분화하고 7은 Distant, 6은 Unstaged/Unknown 계열입니다.
        #        분석 일관성을 위해 위와 같이 기본 병기(0/1/2/3/9)로 접어 사용합니다.
        # - 텍스트가 들어온 경우에는 키워드('in situ','localized','regional','distant')를 기반으로 동일 규칙을 적용합니다.
        # - 숫자/문자 혼재시: 숫자로 해석 가능하면 확장 코드 매핑을 우선 적용하고, 아니면 텍스트 규칙을 적용합니다.
        exp_to_basic = {0:0, 1:1, 2:2, 3:2, 4:2, 5:2, 6:9, 7:3, 9:9}
        if pd.api.types.is_numeric_dtype(series):
            stage_num = series.map(lambda v: exp_to_basic.get(int(v) if pd.notna(v) else v, 9))
        else:
            # 문자열일 경우: 숫자 문자열은 확장코드 매핑, 그 외는 텍스트 규칙 매핑
            def _map_stage_any(v):
                if pd.isna(v):
                    return 9
                s = str(v).strip().lower()
                # 숫자 형태면 확장코드 매핑 우선
                try:
                    iv = int(float(s))
                    return exp_to_basic.get(iv, 9)
                except Exception:
                    ...
                if 'in situ' in s: return 0
                if s.startswith('localized'): return 1
                if s.startswith('regional'): return 2
                if 'distant' in s: return 3
                return 9
            stage_num = series.map(_map_stage_any)

    if stage_num is not None:
        order = [c for c in [0,1,2,3,9] if c in set(stage_num.dropna().astype(int))]
        stage_counts = stage_num.value_counts().reindex(order).fillna(0).astype(int)
        def _fmt_pct(p):
            return f'{p:.1f}%' if p >= 1 else ''
        ax1.pie(stage_counts.values, labels=[f'Stage {i}' for i in stage_counts.index], autopct=_fmt_pct, startangle=90, pctdistance=0.72, labeldistance=1.05, textprops={'fontsize': 9}, wedgeprops=dict(width=0.6))
        ax1.set_aspect('equal')
        ax1.set_title('병기별 환자 분포', fontsize=14, fontweight='bold', pad=20)
        stage_legend = (
            'Stage 코드맵\n'
            '0: In situ (상피내)\n'
            '1: Localized (국한)\n'
            '2: Regional (국소)\n'
            '3: Distant (원격)\n'
            '9: Unknown/Unstaged (불명)'
        )
        ax1.text(0.02, 0.98, stage_legend, transform=ax1.transAxes, va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7))
    else:
        ax1.axis('off')

    # 4-2. 병기별 생존율 (막대)
    if stage_num is not None and 'Vital status recode (study cutoff used)__enc' in df.columns:
        tmp = pd.DataFrame({'Stage': stage_num, 'vital': df['Vital status recode (study cutoff used)__enc']}).dropna()
        tmp['Stage'] = pd.to_numeric(tmp['Stage'], errors='coerce').astype(int)
        stage_survival = tmp.groupby('Stage')['vital'].agg(['count', lambda x: (x == 0).mean() * 100]).reset_index()
        stage_survival.columns = ['Stage', 'Count', 'Survival_Rate']
        stage_survival = stage_survival.sort_values('Stage')
        colors_stage = ['#FF6B6B' if rate < 70 else '#51CF66' if rate > 90 else '#FFD93D' for rate in stage_survival['Survival_Rate']]
        bars = ax2.bar(stage_survival['Stage'].astype(str), stage_survival['Survival_Rate'], color=colors_stage, alpha=0.8)
        ax2.set_title('병기별 생존율', fontsize=14, fontweight='bold', pad=20)
        ax2.text(0.02, 0.98, 'Stage 코드맵\n0: In situ\n1: Localized\n2: Regional\n3: Distant\n9: Unknown', transform=ax2.transAxes, va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7))
        ax2.set_xlabel('병기'); ax2.set_ylabel('생존율 (%)'); ax2.set_ylim(0, 100)
        for rect, v in zip(bars, stage_survival['Survival_Rate']):
            x = rect.get_x() + rect.get_width()/2
            ax2.text(x, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
        tick_labels = [f'Stage {int(s) if pd.notna(s) else s}\n(n={int(n):,})' for s, n in zip(stage_survival['Stage'], stage_survival['Count'])]
        ax2.set_xticklabels(tick_labels)
    else:
        ax2.axis('off')

    # 4-3. 수술 코드별 생존율 (버블)
    surg_col = 'RX Summ--Surg Prim Site (1998+)'
    if surg_col in df.columns and 'Vital status recode (study cutoff used)__enc' in df.columns:
        surgery_survival = df.groupby(surg_col)['Vital status recode (study cutoff used)__enc'].agg(['count', lambda x: (x == 0).mean() * 100]).reset_index()
        surgery_survival.columns = ['Surgery_Code', 'Count', 'Survival_Rate']
        surgery_survival = surgery_survival[surgery_survival['Count'] >= 5000]
        ax3.scatter(surgery_survival['Surgery_Code'], surgery_survival['Survival_Rate'], s=surgery_survival['Count']/100, alpha=0.6, c=surgery_survival['Survival_Rate'], cmap='RdYlGn', edgecolors='black', linewidth=0.5)
        ax3.set_title('수술 코드별 생존율 (버블 크기: 환자 수)', fontsize=14, fontweight='bold', pad=20)
        ax3.set_xlabel('수술 코드'); ax3.set_ylabel('생존율 (%)'); ax3.grid(True, alpha=0.3)
    else:
        ax3.axis('off')

    # 4-4. 성별-연령대별 생존율 히트맵
    age_col = 'Age recode with <1 year olds and 90+'
    if all(c in df.columns for c in ['Sex', age_col, 'Vital status recode (study cutoff used)__enc']):
        survival_pivot = df.groupby(['Sex', age_col])['Vital status recode (study cutoff used)__enc'].agg(lambda x: (x == 0).mean() * 100).reset_index()
        survival_pivot = survival_pivot.pivot(index='Sex', columns=age_col, values='Vital status recode (study cutoff used)__enc')
        age_order = DataModify.DataPreprocessing.AGE_RECODE_ORDER
        age_kor_map = {'00 years':'0','01-04 years':'1-4','05-09 years':'5-9','10-14 years':'10-14','15-19 years':'15-19','20-24 years':'20-24','25-29 years':'25-29','30-34 years':'30-34','35-39 years':'35-39','40-44 years':'40-44','45-49 years':'45-49','50-54 years':'50-54','55-59 years':'55-59','60-64 years':'60-64','65-69 years':'65-69','70-74 years':'70-74','75-79 years':'75-79','80-84 years':'80-84','85-89 years':'85-89','90+ years':'90 이상'}
        present = [a for a in age_order if a in survival_pivot.columns]
        survival_pivot_filtered = survival_pivot.reindex(columns=present)
        if survival_pivot_filtered.shape[1] == 0:
            survival_pivot_filtered = survival_pivot.copy()
        desired_idx = [x for x in ['Female','Male',0,1] if x in survival_pivot_filtered.index]
        if desired_idx:
            survival_pivot_filtered = survival_pivot_filtered.reindex(index=desired_idx)
        sns.heatmap(survival_pivot_filtered, ax=ax4, cmap='RdYlGn', vmin=70, vmax=95, cbar=True, linewidths=0.5, linecolor='white')
        y_labels_map = {'Female':'여성','Male':'남성',0:'여성',1:'남성'}
        ax4.set_yticklabels([y_labels_map.get(i, i) for i in survival_pivot_filtered.index])
        def _age_label_from_code(c):
            try:
                ci = int(c)
                if 0 <= ci < len(age_order):
                    return age_kor_map.get(age_order[ci], str(c))
                return str(c)
            except Exception:
                return age_kor_map.get(str(c), str(c))
        xtick_labels = [_age_label_from_code(c) for c in survival_pivot_filtered.columns]
        ax4.set_title('성별-연령대별 생존율 히트맵', fontsize=14, fontweight='bold', pad=20)
        ax4.set_xlabel('연령대'); ax4.set_ylabel('성별'); ax4.set_xticklabels(xtick_labels, rotation=90, ha='center')
    else:
        ax4.axis('off')
    plt.tight_layout(); plt.show()

def _plot_key_corr_and_impacts(encoded_cod_df):
    df = encoded_cod_df.copy()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    stage_col = 'Combined Summary Stage with Expanded Regional Codes (2004+)'
    def _map_stage_num(v):
        if pd.isna(v):
            return 9
        # 병기(Stage) 분류 규칙 요약
        # - 확장 코드(0,1,2,3,4,5,6,7,9)를 기본 병기(0/1/2/3/9)로 접어 사용합니다.
        #   0→0(In situ), 1→1(Localized), 2/3/4/5→2(Regional), 7→3(Distant), 6/9/그 외→9(Unknown)
        # - 문자열 값의 경우에도 'in situ'/'localized'/'regional'/'distant' 키워드로 동일 규칙을 적용합니다.
        exp_to_basic = {0:0, 1:1, 2:2, 3:2, 4:2, 5:2, 6:9, 7:3, 9:9}
        s = str(v).strip().lower()
        # 숫자 확장코드 우선 처리
        try:
            iv = int(float(s))
            return exp_to_basic.get(iv, 9)
        except Exception:
            ...
        if 'in situ' in s: return 0
        if s.startswith('localized'): return 1
        if s.startswith('regional'): return 2
        if 'distant' in s: return 3
        return 9
    series = df.get(stage_col)
    if series is not None:
        df['__stage_num__'] = series.map(_map_stage_num)
    else:
        df['__stage_num__'] = None
    # 상관행렬(히트맵)에서는 Stage 코드맵(__stage_num__)을 제외하여 가독성 향상
    numeric_cols = ['Sex', 'Age recode with <1 year olds and 90+', 'Year of diagnosis', 'Site recode ICD-O-3/WHO 2008', 'RX Summ--Surg Prim Site (1998+)', 'Vital status recode (study cutoff used)__enc']
    corr = df[numeric_cols].corr()
    ax1.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_title('주요 변수 간 상관관계 매트릭스', fontsize=12, fontweight='bold', pad=20)
    ax1.grid(False)
    ax1.set_xticks(range(len(numeric_cols)))
    ax1.set_yticks(range(len(numeric_cols)))
    fmt_lbl = [('암부위' if 'Site recode ICD-O-3/WHO 2008' in col else col.split()[0][:8]) for col in numeric_cols]
    ax1.set_xticklabels(fmt_lbl, rotation=45, ha='right')
    ax1.set_yticklabels(fmt_lbl)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            val = corr.iloc[i, j]
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center', color=('white' if abs(val) > 0.5 else 'black'), fontweight='bold', fontsize=8)

    # 영향도 계산 시 병기는 정수 코드(__stage_num__) 기준으로 일치시키기
    factors = ['Sex', 'Age recode with <1 year olds and 90+', 'Site recode ICD-O-3/WHO 2008', '__stage_num__']
    survival_impact = []
    factor_names = []
    for f in factors:
        if f in df.columns:
            s = df.groupby(f)['Vital status recode (study cutoff used)__enc'].agg(lambda x: (x == 0).mean() * 100)
            survival_impact.append((s.max() - s.min()))
            factor_names.append('Stage' if f == '__stage_num__' else f.split()[0][:10])
    ax2.bar(range(len(survival_impact)), survival_impact, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(survival_impact)])
    ax2.set_title('변수별 생존율 영향도 (최대-최소 생존율 차이)', fontsize=12, fontweight='bold', pad=20)
    ax2.set_xlabel('변수'); ax2.set_ylabel('생존율 차이 (%포인트)')
    ax2.set_xticks(range(len(factor_names))); ax2.set_xticklabels(factor_names, rotation=45, ha='right')
    for i, v in enumerate(survival_impact):
        ax2.text(i, v + 0.5, f'{v:.1f}%p', ha='center', va='bottom', fontweight='bold')

    if 'Year of diagnosis' in df.columns and 'Vital status recode (study cutoff used)__enc' in df.columns:
        year_map = _build_year_label_map(df, 'Year of diagnosis')
        year_df_src = df[['Year of diagnosis', 'Vital status recode (study cutoff used)__enc']].copy()
        if 'Age recode with <1 year olds and 90+' in df.columns:
            year_df_src['Age recode with <1 year olds and 90+'] = pd.to_numeric(df['Age recode with <1 year olds and 90+'], errors='coerce')
        if 'Sex' in df.columns:
            year_df_src['Sex'] = pd.to_numeric(df['Sex'], errors='coerce')
        year_df_src = year_df_src.dropna(subset=['Year of diagnosis'])
        year_df_src['Vital status recode (study cutoff used)__enc'] = pd.to_numeric(
            year_df_src['Vital status recode (study cutoff used)__enc'], errors='coerce')
        year_df_src = year_df_src.dropna(subset=['Vital status recode (study cutoff used)__enc'])
        if year_df_src.empty:
            ax3.axis('off')
        else:
            decoded_years = _decode_year_series(year_df_src['Year of diagnosis'], year_map)
            numeric_years = pd.to_numeric(decoded_years, errors='coerce')
            if numeric_years.isna().all():
                numeric_years = pd.to_numeric(year_df_src['Year of diagnosis'], errors='coerce')
            year_df_src['Year_label'] = numeric_years
            year_df_src = year_df_src.dropna(subset=['Year_label'])
            if year_df_src.empty:
                ax3.axis('off')
            else:
                year_df_src['Year_label'] = year_df_src['Year_label'].astype(int)

                def _pct_zero(series):
                    s = pd.Series(series).dropna()
                    if s.empty:
                        return np.nan
                    return (s == 0).mean() * 100

                agg_kwargs = {
                    'total_patients': ('Vital status recode (study cutoff used)__enc', 'size'),
                    'survival_rate': ('Vital status recode (study cutoff used)__enc', _pct_zero),
                }
                if 'Age recode with <1 year olds and 90+' in year_df_src.columns:
                    agg_kwargs['avg_age'] = ('Age recode with <1 year olds and 90+', 'mean')
                if 'Sex' in year_df_src.columns:
                    agg_kwargs['female_ratio'] = ('Sex', _pct_zero)

                year_df = (year_df_src.groupby('Year_label')
                                       .agg(**agg_kwargs)
                                       .reset_index()
                                       .rename(columns={'Year_label': 'year'})
                                       .sort_values('year'))
                if year_df.empty:
                    ax3.axis('off')
                else:
                    ax3_twin = ax3.twinx()
                    line1 = ax3.plot(year_df['year'], year_df['survival_rate'], 'b-o', linewidth=2, markersize=4, label='생존율')
                    line2 = ax3_twin.plot(year_df['year'], year_df['total_patients'], 'r-s', linewidth=2, markersize=4, label='환자 수')
                    ax3.set_title('연도별 생존율 및 환자 수 변화', fontsize=12, fontweight='bold', pad=20)
                    ax3.set_xlabel('진단 연도'); ax3.set_ylabel('생존율 (%)', color='blue'); ax3_twin.set_ylabel('환자 수', color='red')
                    ax3.tick_params(axis='y', labelcolor='blue'); ax3_twin.tick_params(axis='y', labelcolor='red'); ax3.grid(True, alpha=0.3)
                    ax3.set_xticks(year_df['year'])
                    ax3.set_xticklabels([str(int(y)) for y in year_df['year']], rotation=45, ha='center')
                    lines = line1 + line2; labels = [l.get_label() for l in lines]
                    ax3.legend(lines, labels, loc='upper left')

    ax4.axis('off')
    try:
        insights_text = f"""
📊 주요 데이터 인사이트 요약

📈 기본 통계
• 총 환자 수: {len(df):,}명
• 전체 생존율: {(df['Vital status recode (study cutoff used)__enc'] == 0).mean() * 100:.1f}%
• 관찰 기간: {int(df['Year of diagnosis'].min())}년 ~ {int(df['Year of diagnosis'].max())}년

🎯 핵심 발견사항
• 생존율에 가장 큰 영향: {factor_names[int(np.argmax(survival_impact))]} ({max(survival_impact):.1f}%p 차이)

⚡ 임상적 시사점
• 조기 발견(Stage 0)의 생존율: 91.3%
• 진행성 암(Stage 1)의 생존율: 65.0%
• 적절한 수술적 치료의 중요성 확인
"""
    except Exception:
        insights_text = '요약 정보 생성 중 오류'
    ax4.text(0.05, 0.95, insights_text, transform=ax4.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    plt.tight_layout(); plt.show()

def _get_cod_korean_name_builder():
    maps = _load_decoding_maps_from_md()
    cod_en_by_code = maps.get('COD to site recode', {})
    cod_ko_by_code = {0:'생존', 1:'간내담관', 2:'폐 및 기관지', 3:'기타 악성종양', 4:'기타 사망원인', 5:'대장(직장 제외)', 6:'심장질환', 7:'알츠하이머', 8:'위', 9:'신장질환(신증후군 포함)', 10:'뇌혈관질환', 11:'간', 12:'유방', 13:'만성폐쇄성폐질환', 14:'만성 림프구성 백혈병', 15:'만성 간질환/간경화', 18:'전립선', 19:'비호지킨 림프종', 20:'당뇨병', 21:'사망원인 미상', 22:'신장 및 신우', 23:'고혈압(심장질환 동반 없음)', 24:'다발성 골수종', 25:'급성 골수성 백혈병', 26:'뇌 및 기타 신경계', 27:'폐렴/인플루엔자', 28:'췌장', 31:'사고 및 부작용', 32:'자궁체부', 33:'방광', 34:'제자리/양성/미확정 신생물', 35:'죽상경화증', 36:'식도', 37:'자궁경부', 38:'자살/자해', 40:'피부 흑색종', 41:'직장 및 직장결장 이행부', 42:'자궁, 기타특정불가', 43:'연조직(심장 포함)', 44:'비흑색종 피부암', 45:'혀', 46:'패혈증', 47:'갑상선', 52:'흉막', 53:'난소', 54:'기타 담도', 61:'뼈/관절', 62:'편도', 63:'복막/망/장간막', 64:'후두', 65:'호지킨 림프종', 66:'후복막', 68:'결핵', 70:'구인두', 71:'질', 72:'담낭', 73:'치은/기타 구강', 74:'기타 내분비(흉선 포함)', 76:'기관/종격/기타 호흡기관', 77:'고환', 78:'급성 림프구성 백혈병', 79:'기타 요로기관', 80:'기타 구강/인두', 81:'기타 여성 생식기관', 82:'코/비강/중이', 83:'기타 급성 백혈병', 84:'요관', 85:'외음부', 86:'만성 골수성 백혈병', 87:'눈/안와', 88:'입술', 89:'임신/출산/산욕 합병증'}
    cod_en_to_ko = {en: cod_ko_by_code.get(code, en) for code, en in cod_en_by_code.items()}
    def get_cod_korean_name(code_or_en):
        # 결측 처리
        try:
            if pd.isna(code_or_en):
                return None
        except Exception:
            ...
        # 숫자형(정수/실수/문자열 숫자) → 정수 코드로 변환 후 매핑
        try:
            v = int(float(code_or_en))
            en = cod_en_by_code.get(v)
            if en is not None:
                return cod_ko_by_code.get(v, en)
            return cod_ko_by_code.get(v, str(v))
        except Exception:
            # 문자열 영문 라벨일 경우 영→한 매핑 시도
            s = str(code_or_en)
            return cod_en_to_ko.get(s, s)
    return get_cod_korean_name

def _plot_cod_top_and_age_pattern(encoded_cod_df):
    df = encoded_cod_df.copy()
    if 'Vital status recode (study cutoff used)__enc' not in df.columns or 'COD to site recode__enc' not in df.columns:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    death_patients = df[df['Vital status recode (study cutoff used)__enc'] == 1].copy()
    total_deaths = len(death_patients)
    get_cod_korean_name = _get_cod_korean_name_builder()
    death_patients['COD_KOR'] = death_patients['COD to site recode__enc'].map(get_cod_korean_name)
    cod_counts = death_patients['COD_KOR'].value_counts().head(15)
    cod_korean_names = cod_counts.index.tolist()
    base_colors = list(cm.tab20.colors) + list(cm.tab20b.colors) + list(cm.tab20c.colors)
    palette = {lab: base_colors[i % len(base_colors)] for i, lab in enumerate(cod_korean_names)}
    palette['폐 및 기관지'] = '#1f77b4'; palette['대장(직장 제외)'] = '#d62728'
    colors_cod = [palette.get(lab, '#888888') for lab in cod_korean_names]
    ax1.barh(range(len(cod_counts)), cod_counts.values, color=colors_cod)
    ax1.set_yticks(range(len(cod_counts))); ax1.set_yticklabels(cod_korean_names, fontsize=11)
    ax1.set_title('주요 사망원인별 사망자 수 (상위 15개)', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('사망자 수')
    if len(cod_counts) > 0:
        max_cnt = int(cod_counts.max()); ax1.set_xlim(0, max(max_cnt * 1.2, 1))
    x_min, x_max = ax1.get_xlim(); span = x_max - x_min if x_max > x_min else 1
    x_left = x_min + span * 0.01; x_right = x_max - span * 0.01
    for i, v in enumerate(cod_counts.values):
        ax1.text(x_right, i, f'{int(v):,}', va='center', ha='right', fontweight='bold', fontsize=10, color='#333')
    for i, (name, count) in enumerate(cod_counts.items()):
        pct = (count / total_deaths) * 100 if total_deaths else 0
        ax1.text(x_left, i, f'({pct:.1f}%)', va='center', ha='left', fontsize=10, color='#000')
    legend_elements_cod = [plt.Rectangle((0,0),1,1, facecolor=palette[name], label=f'{name} ({cod_counts.loc[name]:,}명)') for name in cod_korean_names[:5]]
    ax1.legend(handles=legend_elements_cod, loc='upper right', fontsize=11, title='주요 사망원인')

    age_col = 'Age recode with <1 year olds and 90+'
    age_death_data = death_patients.groupby([age_col, 'COD_KOR']).size().reset_index(name='count')
    top_cods = cod_korean_names[:5]
    actual_cods = [c for c in top_cods if c in age_death_data['COD_KOR'].unique()]
    if not actual_cods:
        actual_cods = (age_death_data.groupby('COD_KOR')['count'].sum().sort_values(ascending=False).head(5).index.tolist())
    age_cod_filtered = age_death_data[age_death_data['COD_KOR'].isin(actual_cods)]
    pivot_age_cod = age_cod_filtered.pivot_table(index=age_col, columns='COD_KOR', values='count', fill_value=0)
    pivot_age_cod = pivot_age_cod.loc[:, actual_cods]
    try:
        pivot_age_cod.index = pivot_age_cod.index.astype(int)
        pivot_age_cod = pivot_age_cod.sort_index()
        age_map_num = {0:'0',1:'1-4',2:'5-9',3:'10-14',4:'15-19',5:'20-24',6:'25-29',7:'30-34',8:'35-39',9:'40-44',10:'45-49',11:'50-54',12:'55-59',13:'60-64',14:'65-69',15:'70-74',16:'75-79',17:'80-84',18:'85-89',19:'90 이상'}
        xticklabels = [age_map_num.get(x, str(x)) for x in pivot_age_cod.index]
    except Exception:
        xticklabels = [str(x) for x in pivot_age_cod.index]
    if pivot_age_cod.empty or pivot_age_cod.shape[1] == 0:
        ax2.axis('off'); ax2.text(0.5, 0.5, '표시할 데이터가 없습니다', ha='center', va='center', fontsize=12)
    else:
        for lab in pivot_age_cod.columns:
            if lab not in palette:
                palette[lab] = base_colors[len(palette) % len(base_colors)]
        stack_colors = [palette[c] for c in pivot_age_cod.columns]
        pivot_age_cod.plot(kind='bar', stacked=True, ax=ax2, color=stack_colors)
        ax2.set_title('연령대별 주요 사망원인 분포 패턴 (상위 5개)', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('연령대'); ax2.set_ylabel('사망자 수')
        ax2.set_axisbelow(True); ax2.grid(axis='both', linestyle='--', alpha=0.35)
        ax2.set_xticklabels(xticklabels, rotation=90, ha='center', fontsize=10)
        handles = [plt.Rectangle((0,0), 1, 1, facecolor=palette[c], label=c) for c in pivot_age_cod.columns]
        ax2.legend(handles=handles, title='사망원인', loc='upper left', frameon=True, fontsize=11)
    plt.tight_layout(); plt.show()


def _plot_target_extras(encoded_cod_df):
    # [타깃 분포] 노트북 셀 24: target_label 카운트만 출력
    if 'target_label' not in encoded_cod_df.columns:
        return
    plt.figure(figsize=(6,4))
    # 고정 순서: -1, 0, 1, 2, 3 (존재하는 항목만)
    df_local = encoded_cod_df.copy()
    df_local['target_label'] = pd.to_numeric(df_local['target_label'], errors='coerce')
    desired_order = [-1, 0, 1, 2, 3]
    present_values = [v for v in desired_order if v in set(df_local['target_label'].dropna().unique().astype(int))]
    order = present_values if present_values else df_local['target_label'].value_counts().index
    ax = sns.countplot(x='target_label', data=df_local, order=order, palette='Set2')
    for c in ax.containers:
        ax.bar_label(c, fmt='%d', padding=2, fontsize=9)
    ax.set_title('target_label 분포', fontsize=13, fontweight='bold')
    ax.set_xlabel('target_label'); ax.set_ylabel('건수')
    # target_label 한글 매핑 안내 박스 표시
    tl_kor = {-1:'생존', 0:'암 관련 사망', 1:'합병증 관련 사망', 2:'기타 질환 사망', 3:'자살/자해'}
    mapping_lines = ['target_label 매핑'] + [f'{v}: {tl_kor[v]}' for v in desired_order if v in set(order)]
    mapping_text = '\n'.join(mapping_lines)
    ax.text(0.98, 0.98, mapping_text, transform=ax.transAxes, va='top', ha='right', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    plt.tight_layout(); plt.show()


def _plot_gender_age_event(encoded_cod_df):
    # [성별/연령 분석] (셀 26) 성별별 target_label 분포(%) + 연령대별 사건확률 + 성별×연령 히트맵
    if 'target_label' not in encoded_cod_df.columns:
        return
    drop_cols = ['Vital status recode (study cutoff used)','Vital status recode (study cutoff used)__enc','Survival months flag','Survival months flag__enc','COD to site recode','COD to site recode__enc']
    df_t = encoded_cod_df.drop(columns=[c for c in drop_cols if c in encoded_cod_df.columns], errors='ignore').copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,4))
    if 'Sex' in df_t.columns:
        counts = df_t.groupby(['Sex','target_label']).size().unstack(fill_value=0)
        tmp = counts.div(counts.sum(axis=1), axis=0).reset_index().melt(id_vars='Sex', var_name='target_label', value_name='pct')
        # 성별 라벨 한글화(0/1, Female/Male 모두 대응)
        sex_map = {0: '여성', 1: '남성', 'Female': '여성', 'Male': '남성'}
        try:
            tmp['Sex_kor'] = tmp['Sex'].map(lambda v: sex_map.get(v, v))
        except Exception:
            tmp['Sex_kor'] = tmp['Sex']
        # hue 순서 고정 및 라벨 한국어 매핑
        tmp['target_label'] = pd.to_numeric(tmp['target_label'], errors='coerce')
        desired_order = [-1, 0, 1, 2, 3]
        hue_order = [v for v in desired_order if v in set(tmp['target_label'].dropna().unique().astype(int))]
        tl_kor = {-1:'생존', 0:'암 관련 사망', 1:'합병증 관련 사망', 2:'기타 질환 사망', 3:'자살/자해'}
        ax1 = sns.barplot(data=tmp, x='Sex_kor', y='pct', hue='target_label', hue_order=hue_order, ax=ax1, palette='Set3')
        # 범례 라벨을 한글로 교체
        handles, labels = ax1.get_legend_handles_labels()
        try:
            labels_int = [int(float(l)) for l in labels]
            labels_kor = [tl_kor.get(v, l) for v in labels_int]
        except Exception:
            labels_kor = labels
        leg = ax1.legend(handles, labels_kor, title='사망 클래스', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax1.set_title('성별별 target_label 분포(%)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('성별')
        ax1.set_ylabel('%')
        # target_label 매핑 안내 박스(Stages 방식)
        mapping_lines = ['target_label 매핑'] + [f'{v}: {tl_kor[v]}' for v in desired_order if v in set(hue_order)]
        mapping_text = '\n'.join(mapping_lines)
        ax1.text(0.98, 0.98, mapping_text, transform=ax1.transAxes, va='top', ha='right', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    age_col = 'Age recode with <1 year olds and 90+'
    if age_col in df_t.columns:
        age_order = DataModify.DataPreprocessing.AGE_RECODE_ORDER
        age_kor_map = {'00 years':'0','01-04 years':'1-4','05-09 years':'5-9','10-14 years':'10-14','15-19 years':'15-19','20-24 years':'20-24','25-29 years':'25-29','30-34 years':'30-34','35-39 years':'35-39','40-44 years':'40-44','45-49 years':'45-49','50-54 years':'50-54','55-59 years':'55-59','60-64 years':'60-64','65-69 years':'65-69','70-74 years':'70-74','75-79 years':'75-79','80-84 years':'80-84','85-89 years':'85-89','90+ years':'90 이상'}
        age_evt = df_t.groupby(age_col)['target_label'].apply(lambda x: (x != -1).mean() * 100).reset_index(name='Event_Rate')
        if np.issubdtype(age_evt[age_col].dtype, np.number):
            present_codes = sorted([int(c) for c in age_evt[age_col].dropna().unique() if 0 <= int(c) < len(age_order)])
            age_evt = age_evt[age_evt[age_col].isin(present_codes)].copy().sort_values(age_col)
            x_vals = np.arange(len(age_evt)); x_labels = [age_kor_map.get(age_order[int(c)], str(c)) for c in age_evt[age_col]]
            ax2.plot(x_vals, age_evt['Event_Rate'], marker='o', color='#E56B6F')
            ax2.set_xticks(x_vals); ax2.set_xticklabels(x_labels, rotation=45, ha='center')
        else:
            present_labels = [a for a in age_order if a in list(age_evt[age_col].astype(str).unique())]
            age_evt[age_col] = pd.Categorical(age_evt[age_col].astype(str), categories=present_labels, ordered=True)
            age_evt = age_evt.sort_values(age_col)
            sns.lineplot(data=age_evt, x=age_col, y='Event_Rate', marker='o', ax=ax2, color='#E56B6F')
            ax2.set_xticklabels([age_kor_map.get(str(a), str(a)) for a in age_evt[age_col]], rotation=45, ha='center')
        ax2.set_title('연령대별 사건 확률 P(target_label != -1)', fontsize=12, fontweight='bold'); ax2.set_xlabel('연령대'); ax2.set_ylabel('확률(%)'); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    # 성별×연령 사건 확률 히트맵
    if not np.issubdtype(df_t['target_label'].dtype, np.number):
        df_t['target_label'] = pd.to_numeric(df_t['target_label'], errors='coerce')
    pivot = df_t.groupby(['Sex', age_col])['target_label'].apply(lambda x: (x != -1).mean() * 100).unstack()
    if np.issubdtype(pivot.columns.dtype, np.number):
        present_codes = [c for c in range(len(DataModify.DataPreprocessing.AGE_RECODE_ORDER)) if c in pivot.columns]
        pivot = pivot.reindex(columns=present_codes) if present_codes else pivot
        x_labels = [age_kor_map.get(DataModify.DataPreprocessing.AGE_RECODE_ORDER[int(c)], str(c)) for c in pivot.columns]
    else:
        present = [a for a in DataModify.DataPreprocessing.AGE_RECODE_ORDER if a in pivot.columns]
        pivot = pivot.reindex(columns=present) if present else pivot
        x_labels = [age_kor_map.get(a, a) for a in pivot.columns]
    plt.figure(figsize=(12,3.8))
    pivot = pivot.fillna(0)
    im = sns.heatmap(pivot, cmap='RdYlGn_r', vmin=0, vmax=100, cbar=True, linewidths=0.5, linecolor='white')
    y_labels_map = {'Female':'여성','Male':'남성',0:'여성',1:'남성'}
    if np.issubdtype(pivot.index.dtype, np.number) and len(pivot.index.unique()) == 2:
        idx_sorted = sorted(pivot.index.tolist())
        y_tick = [('여성' if i == idx_sorted[0] else '남성') for i in pivot.index]
        im.set_yticklabels(y_tick)
    else:
        im.set_yticklabels([y_labels_map.get(i, i) for i in pivot.index])
    im.set_xticklabels(x_labels, rotation=45, ha='center')
    plt.title('성별-연령대별 사건 확률 (P(target_label != -1))', fontsize=13, fontweight='bold')
    plt.xlabel('연령대'); plt.ylabel('성별')
    plt.tight_layout(); plt.show()


def _plot_yearly_event_and_classes(encoded_cod_df):
    # [연도별 추이] 사건확률 변화(라인+추세선) + 연도별 사망 클래스(0/1/2/3) 구성비 변화
    if 'target_label' not in encoded_cod_df.columns or 'Year of diagnosis' not in encoded_cod_df.columns:
        return
    drop_cols = ['Vital status recode (study cutoff used)','Vital status recode (study cutoff used)__enc','Survival months flag','Survival months flag__enc','COD to site recode','COD to site recode__enc']
    df_t = encoded_cod_df.drop(columns=[c for c in drop_cols if c in encoded_cod_df.columns], errors='ignore').copy()
    # 연도 코드 → 원본 연도 라벨 매핑
    year_map = _build_year_label_map(df_t, 'Year of diagnosis')
    # 사건 확률 연도별
    year_df = df_t.groupby('Year of diagnosis')['target_label'].apply(lambda x: (x != -1).mean() * 100).reset_index(name='Event_Rate')
    year_df['Year_label'] = year_df['Year of diagnosis'].map(year_map).fillna(year_df['Year of diagnosis'])
    year_df['Year_label'] = pd.to_numeric(year_df['Year_label'], errors='coerce')
    year_df = year_df.dropna(subset=['Year_label'])
    if year_df.empty:
        return
    year_df['Year_label'] = year_df['Year_label'].astype(int)
    plt.figure(figsize=(7,4))
    sns.lineplot(data=year_df, x='Year_label', y='Event_Rate', marker='o', linewidth=2, color='#3A86FF')
    plt.title('진단 연도별 사건 확률 추이 (P(target_label != -1))', fontsize=13, fontweight='bold')
    plt.xlabel('진단 연도'); plt.ylabel('확률(%)')
    plt.xticks(sorted(year_df['Year_label'].unique()), rotation=45)
    plt.grid(True, alpha=0.3)
    try:
        x = pd.to_numeric(year_df['Year_label'], errors='coerce')
        z = np.polyfit(x, year_df['Event_Rate'], 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), '--', alpha=0.8, color='blue', linewidth=2)
    except Exception:
        ...
    plt.tight_layout(); plt.show()

    year_col = 'Year of diagnosis'
    df_t['target_label'] = pd.to_numeric(df_t['target_label'], errors='coerce')
    df_t[year_col] = pd.to_numeric(df_t[year_col], errors='coerce').round().astype('Int64')
    d = df_t[(df_t['target_label'] != -1)][[year_col, 'target_label']].dropna().copy()
    if d.empty:
        return
    # 연도 코드 → 라벨로 변환하여 x축에 실제 연도 표시
    d['Year_label'] = d[year_col].map(year_map).fillna(d[year_col])
    d['Year_label'] = pd.to_numeric(d['Year_label'], errors='coerce')
    d = d.dropna(subset=['Year_label'])
    if d.empty:
        return
    d['Year_label'] = d['Year_label'].astype(int)
    years = sorted(d['Year_label'].dropna().unique().astype(int).tolist())
    counts = d.groupby(['Year_label', 'target_label']).size().unstack(fill_value=0)
    keep = [c for c in [0,1,2,3] if c in counts.columns]
    counts = counts.reindex(columns=keep).reindex(index=years).fillna(0)
    perc = counts.div(counts.sum(axis=1), axis=0) * 100
    long = perc.reset_index().melt(id_vars='Year_label', var_name='class', value_name='pct')
    class_kor = {0:'암 관련 사망', 1:'합병증 관련 사망', 2:'기타 질환 사망', 3:'자살/자해'}
    long['class_kor'] = long['class'].map(class_kor).astype(str)
    plt.figure(figsize=(10,5))
    palette = {'암 관련 사망':'#D62839','합병증 관련 사망':'#F4A261','기타 질환 사망':'#3A86FF','자살/자해':'#8338EC'}
    sns.lineplot(data=long, x='Year_label', y='pct', hue='class_kor', marker='o', palette=palette, linewidth=2.5, markersize=7)
    plt.title('연도별 사망 클래스 구성비 변화', fontsize=13, fontweight='bold')
    plt.xlabel('진단 연도'); plt.ylabel('구성비(%)')
    plt.xticks(years, years, rotation=45)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(title='클래스', bbox_to_anchor=(1.02, 1), loc='best')
    plt.tight_layout(); plt.show()


def show_graph(df) :
    # 노트북(insight/data_insight.ipynb)의 EDA 시각화를 일괄 실행
    _set_style()
    encoded_df, encoded_cod_df = _load_encoded_data(df)
    _plot_corr_with_target(encoded_df)               # 피처-타깃 상관 히트맵
    _plot_survival_months(encoded_df)                # 생존개월 vs 사건확률 추이
    _plot_basic_distributions(encoded_cod_df)        # 성별/연령/생존상태/연도 분포
    _plot_site_survival_year(encoded_cod_df)         # 암 부위/연령/연도 관련 시각화
    _plot_stage_surgery_gender_age(encoded_cod_df)   # 병기/수술/성별×연령 교차 시각화(생존율)
    _plot_key_corr_and_impacts(encoded_cod_df)       # 상관행렬/영향도/연도 지표/요약
    _plot_cod_top_and_age_pattern(encoded_cod_df)    # COD Top15 + 연령대별 Top5 스택
    _plot_target_extras(encoded_cod_df)              # 타깃 분포
    _plot_gender_age_event(encoded_cod_df)           # 성별/연령 분석(사건확률)
    _plot_yearly_event_and_classes(encoded_cod_df)   # 연도별 사건확률 및 클래스 변화
