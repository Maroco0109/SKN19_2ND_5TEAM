# smart_csv.py
from __future__ import annotations
import os, gc, re, math
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import pandas as pd

# 전역: 메모리 안전 옵션
pd.options.mode.copy_on_write = True

# 휴리스틱 파라미터
LOW_CARDINALITY_MAX_UNIQUE_RATIO = 0.15   # 고유값 비율이 이보다 작으면 category 후보
MAX_CATEGORY_UNIQUES = 50_000             # 카테고리 최대 고유값 수(너무 크면 비효율)
DEFAULT_CHUNKSIZE = 200_000

def _guess_date_cols(sample: pd.DataFrame) -> List[str]:
    date_like_by_name = [c for c in sample.columns if re.search(r"(date|dt|ymd|time|timestamp)", c, re.I)]
    date_cols = []
    for c in date_like_by_name:
        try:
            pd.to_datetime(sample[c], errors="raise")
            date_cols.append(c)
        except Exception:
            pass
    return list(dict.fromkeys(date_cols))  # 중복 제거, 순서 유지

def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["int64", "int32", "int16", "int8"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes(include=["float64", "float32"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    return df

def _categorify(df: pd.DataFrame, candidate_cols: List[str]) -> pd.DataFrame:
    for c in candidate_cols:
        if c not in df.columns: 
            continue
        ser = df[c]
        # 문자열/오브젝트/Int64(NA있는 정수)만 카테고리화 고려
        if ser.dtype.kind in ("O", "U") or str(ser.dtype).startswith("string") or str(ser.dtype).startswith("Int"):
            nunique = ser.nunique(dropna=True)
            total = len(ser)
            ratio = (nunique / max(total, 1)) if total else 1.0
            if nunique <= MAX_CATEGORY_UNIQUES and ratio <= LOW_CARDINALITY_MAX_UNIQUE_RATIO:
                df[c] = ser.astype("category")
    return df

def infer_schema(csv_path: str, nrows: int = 50_000, usecols: Optional[List[str]] = None) -> Dict:
    """작은 샘플로 dtype/parse_dates/category 후보를 추정."""
    sample = pd.read_csv(csv_path, nrows=nrows, usecols=usecols, low_memory=True)
    # 숫자 다운캐스트 후보
    sample = _downcast_numeric(sample)
    # 카테고리 후보: object/string & 저카디널리티
    obj_cols = sample.select_dtypes(include=["object", "string"]).columns.tolist()
    cat_candidates = []
    for c in obj_cols:
        nunique = sample[c].nunique(dropna=True)
        ratio = nunique / max(len(sample), 1)
        if nunique <= MAX_CATEGORY_UNIQUES and ratio <= LOW_CARDINALITY_MAX_UNIQUE_RATIO:
            cat_candidates.append(c)
    # 날짜 후보
    date_cols = _guess_date_cols(sample)
    # dtype 맵 (가능한 한 축소된 타입을 기록)
    dtype_map = {}
    for c in sample.columns:
        dt = str(sample[c].dtype)
        if dt in ("int8","int16","int32","Int8","Int16","Int32"):
            dtype_map[c] = dt
        elif dt in ("float16","float32"):
            dtype_map[c] = dt
        # object/string은 일단 그대로 두고, 나중에 카테고리화
    return {
        "columns": sample.columns.tolist(),
        "dtype_map": dtype_map,
        "date_cols": date_cols,
        "category_candidates": cat_candidates
    }

def process_csv_stream(
    csv_path: str,
    usecols: Optional[List[str]] = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
    preprocess_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    to_parquet: Optional[str] = None,       # 출력 경로(.parquet). 주면 스트리밍 저장
    parquet_engine: str = "pyarrow",        # pyarrow 권장
    compression: str = "snappy",            # 압축
    datetime_cols_tz_utc: bool = False,     # 날짜 칼럼을 UTC로 정규화할지
) -> Iterable[pd.DataFrame]:
    """
    - 큰 CSV를 청크로 읽어 메모리 안전하게 가공.
    - preprocess_fn이 있으면 각 청크에 적용.
    - to_parquet가 주어지면 파일로 바로 append 저장(청크 반환하지 않음).
    - 반환: to_parquet=None이면 처리된 청크 제너레이터(연달아 for로 소비).
    """
    schema = infer_schema(csv_path, usecols=usecols)
    dtype_map = {c:t for c,t in schema["dtype_map"].items() if (usecols is None or c in usecols)}
    date_cols = [c for c in schema["date_cols"] if (usecols is None or c in usecols)]
    cat_candidates = [c for c in schema["category_candidates"] if (usecols is None or c in usecols)]

    # 파케이 준비 상태 체크
    write_to_parquet = bool(to_parquet)
    if write_to_parquet:
        # 첫 청크에서 스키마를 정하고 이후 append
        if os.path.exists(to_parquet):
            os.remove(to_parquet)

    reader = pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=dtype_map if dtype_map else None,
        parse_dates=date_cols if date_cols else None,
        chunksize=chunksize,
        low_memory=True
    )

    for chunk in reader:
        # 숫자 추가 다운캐스트
        chunk = _downcast_numeric(chunk)
        # 카테고리화
        chunk = _categorify(chunk, cat_candidates)
        # 날짜 타임존(옵션)
        if date_cols and datetime_cols_tz_utc:
            for c in date_cols:
                # 이미 datetime이면 tz_localize/convert
                if pd.api.types.is_datetime64_any_dtype(chunk[c]):
                    if chunk[c].dt.tz is None:
                        chunk[c] = chunk[c].dt.tz_localize("UTC")
                    else:
                        chunk[c] = chunk[c].dt.tz_convert("UTC")

        # 사용자 전처리
        if preprocess_fn is not None:
            chunk = preprocess_fn(chunk)

        if write_to_parquet:
            chunk.to_parquet(
                to_parquet,
                engine=parquet_engine,
                compression=compression,
                index=False,
                append=True
            )
            # 메모리 즉시 반환
            del chunk
            gc.collect()
            continue

        yield chunk  # 호출자가 직접 소비

def csv_to_parquet_stream(
    csv_path: str,
    out_path: Optional[str] = None,
    usecols: Optional[List[str]] = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
    preprocess_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    parquet_engine: str = "pyarrow",
    compression: str = "snappy",
):
    """CSV를 메모리 안전하게 바로 Parquet으로 변환(스트리밍)."""
    if out_path is None:
        out_path = os.path.splitext(csv_path)[0] + ".parquet"
    process_csv_stream(
        csv_path=csv_path,
        usecols=usecols,
        chunksize=chunksize,
        preprocess_fn=preprocess_fn,
        to_parquet=out_path,
        parquet_engine=parquet_engine,
        compression=compression
    )
    return out_path
